"""t-SNE comparison of learned representations on the IHA dataset.

For each architecture in the test set we extract a representation under four
predictors (and one no-model reference), reduce to 2-D with t-SNE, color by
val_loss, and report the kNN-val_loss MAE — i.e. how well each representation's
geometry aligns with perplexity.

Representations:
  • Raw       — 324-dim per-layer feature vector (no model).
  • RF        — proximity distance: 1 − fraction of trees that put two samples
                in the same leaf, then t-SNE with metric="precomputed".
  • MLP       — penultimate activation of the HW-GPT-Bench paper-baseline Net
                (the 128-dim output of fc4, after ReLU, before the regression
                head fc5).
  • ForgeFormer — pooled token embedding (d_model-dim) just before the final
                Linear head — what the paper §3.2 calls the per-layer encoder
                output after masked-mean pooling.

Reuses the saved baselines + ForgeFormer ckpt (no retraining); writes
figs/fig_embeddings_tsne.pdf.
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR))

from surrogate.data import load_raw_arch_dataset, normalize_batch  # noqa: E402
from surrogate.inference import load_surrogate  # noqa: E402
from table_a import (  # noqa: E402
    BASELINES_DIR, CSV_PATH, CUSTOM_CKPT, CUSTOM_SEED, CUSTOM_TEST_RATIO,
    N_LAYERS_CUSTOM, _PaperNet, build_raw_features_custom,
)


OUT_DIR = THIS_DIR / "figs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TSNE_PERPLEXITY = 30
TSNE_SEED = 0
KNN_K = 10

# NeurIPS standard: body text 10pt Times; figure text 8-9pt to remain
# legible at 1× column width without growing taller than ~2.5in.
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0, "axes.titlesize": 9.0, "axes.labelsize": 8.5,
    "legend.fontsize": 8.0, "legend.frameon": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.30, "grid.linewidth": 0.45,
}


# ── Embedding extractors ────────────────────────────────────────────────────

@torch.no_grad()
def forgeformer_pooled_embeddings(ckpt_path: Path) -> np.ndarray:
    """Pooled (pre-head) d_model-dim representation for every IHA architecture."""
    model, norm_stats, max_layers = load_surrogate(str(ckpt_path), DEVICE)
    raw = load_raw_arch_dataset(str(CSV_PATH), max_layers=max_layers)
    full = normalize_batch(raw, norm_stats)
    caught = []
    h = model.head.register_forward_pre_hook(
        lambda _m, inp: caught.append(inp[0].detach().cpu().clone())
    )
    bs = 128
    for i in range(0, full.x.shape[0], bs):
        model(full.x[i:i + bs].to(DEVICE),
              padding_mask=full.padding_mask[i:i + bs].to(DEVICE))
    h.remove()
    return torch.cat(caught, dim=0).numpy()


def mlp_penultimate_embeddings(X_raw_scaled: np.ndarray, ckpt_path: Path) -> np.ndarray:
    """Penultimate (post-fc4-ReLU) 128-dim activation for every architecture."""
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = _PaperNet(nfeat=ck["nfeat"], hidden=ck.get("hidden", 128)).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    caught = []
    h = model.fc5.register_forward_pre_hook(
        lambda _m, inp: caught.append(inp[0].detach().cpu().clone())
    )
    with torch.no_grad():
        bs = 1024
        x_t = torch.from_numpy(X_raw_scaled).float().to(DEVICE)
        for i in range(0, x_t.shape[0], bs):
            model(x_t[i:i + bs])
    h.remove()
    return torch.cat(caught, dim=0).numpy()


def rf_proximity_distance(rf, X: np.ndarray) -> np.ndarray:
    """Pairwise distance: 1 − fraction of trees that share a leaf for (i,j)."""
    leaves = rf.apply(X).astype(np.int32)  # [N, n_trees]
    return pairwise_distances(leaves, metric="hamming")


# ── t-SNE ───────────────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray, perplexity=TSNE_PERPLEXITY, seed=TSNE_SEED) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=seed).fit_transform(Xs)
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", max_iter=1500, random_state=seed, metric="euclidean",
    ).fit_transform(Xs)


def run_tsne_precomputed(D: np.ndarray, perplexity=TSNE_PERPLEXITY, seed=TSNE_SEED) -> np.ndarray:
    return TSNE(
        n_components=2, perplexity=perplexity, init="random",
        learning_rate="auto", max_iter=1500, random_state=seed, metric="precomputed",
    ).fit_transform(D)


# ── Quantitative geometry: kNN val_loss MAE ─────────────────────────────────

def knn_valloss_mae(emb: np.ndarray | None, vals: np.ndarray,
                    train_idx, test_idx, k: int = KNN_K,
                    D_full: np.ndarray | None = None) -> float:
    if D_full is not None:
        sub = D_full[np.ix_(test_idx, train_idx)]
        nn_idx = np.argpartition(sub, k, axis=1)[:, :k]
        preds = vals[train_idx][nn_idx].mean(axis=1)
    else:
        sc = StandardScaler().fit(emb[train_idx])
        Etr, Ete = sc.transform(emb[train_idx]), sc.transform(emb[test_idx])
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Etr)
        _, idxs = nn.kneighbors(Ete)
        preds = vals[train_idx][idxs].mean(axis=1)
    return float(np.mean(np.abs(preds - vals[test_idx])))


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_panel(ax, xy, vals, test_mask, title, knn_mae,
               vmin: float = None, vmax: float = None):
    """All points plotted; no outlier removal. vmin/vmax shared across panels
    so colors are directly comparable."""
    train_m = ~test_mask
    ax.scatter(xy[train_m, 0], xy[train_m, 1],
               c=vals[train_m], cmap="viridis_r", s=5.0, alpha=0.50,
               linewidths=0, vmin=vmin, vmax=vmax,
               rasterized=True, zorder=2)
    sc = ax.scatter(xy[test_mask, 0], xy[test_mask, 1],
                    c=vals[test_mask], cmap="viridis_r", s=18.0, alpha=1.0,
                    edgecolors="white", linewidths=0.55,
                    vmin=vmin, vmax=vmax,
                    rasterized=True, zorder=3)
    ax.set_title(title, fontsize=9.0, pad=3)
    ax.text(0.03, 0.97,
            fr"$k$NN MAE $=\;\mathbf{{{knn_mae:.3f}}}$",
            transform=ax.transAxes, fontsize=7.8, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.92, pad=1.6))
    # Square frame fitted to all points.
    pad = 0.04 * float(np.ptp(xy, axis=0).max())
    xlo, xhi = xy[:, 0].min() - pad, xy[:, 0].max() + pad
    ylo, yhi = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    span = max(xhi - xlo, yhi - ylo)
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return sc


CACHE_PATH = THIS_DIR / "figs" / "_tsne_cache.npz"


def compute_or_load_cache():
    """Returns (xy_raw, xy_rf, xy_mlp, xy_af, vals, train_idx, test_idx,
    mae_raw, mae_rf, mae_mlp, mae_af). Recomputes if cache missing."""
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=False)
        print(f"[cache hit] {CACHE_PATH}")
        return (z["xy_raw"], z["xy_rf"], z["xy_mlp"], z["xy_af"], z["vals"],
                z["train_idx"], z["test_idx"],
                float(z["mae_raw"]), float(z["mae_rf"]),
                float(z["mae_mlp"]), float(z["mae_af"]))
    df = pd.read_csv(CSV_PATH)
    df = df[np.isfinite(df["val_loss"].values)].reset_index(drop=True)
    df["_mask"] = df["global_layer_mask"].apply(ast.literal_eval).apply(np.array)
    vals = df["val_loss"].values
    n = len(vals)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=CUSTOM_TEST_RATIO,
                                           random_state=CUSTOM_SEED, shuffle=True)
    print(f"IHA: n={n}, train={len(train_idx)}, test={len(test_idx)}, seed={CUSTOM_SEED}")
    X_raw = build_raw_features_custom(df)
    af_emb = forgeformer_pooled_embeddings(CUSTOM_CKPT)
    sc_mlp = joblib.load(BASELINES_DIR / "custom_mlp_scaler.joblib")
    mlp_emb = mlp_penultimate_embeddings(sc_mlp.transform(X_raw),
                                         BASELINES_DIR / "custom_mlp.pt")
    rf = joblib.load(BASELINES_DIR / "custom_rf.joblib")
    rf_dist = rf_proximity_distance(rf, X_raw)
    print("[t-SNE] running ...")
    xy_raw = run_tsne(X_raw)
    xy_rf  = run_tsne_precomputed(rf_dist)
    xy_mlp = run_tsne(mlp_emb)
    xy_af  = run_tsne(af_emb)
    mae_raw = knn_valloss_mae(X_raw, vals, train_idx, test_idx)
    mae_rf  = knn_valloss_mae(None,  vals, train_idx, test_idx, D_full=rf_dist)
    mae_mlp = knn_valloss_mae(mlp_emb, vals, train_idx, test_idx)
    mae_af  = knn_valloss_mae(af_emb, vals, train_idx, test_idx)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH,
             xy_raw=xy_raw, xy_rf=xy_rf, xy_mlp=xy_mlp, xy_af=xy_af,
             vals=vals, train_idx=train_idx, test_idx=test_idx,
             mae_raw=mae_raw, mae_rf=mae_rf, mae_mlp=mae_mlp, mae_af=mae_af)
    print(f"[cache saved] {CACHE_PATH}")
    return (xy_raw, xy_rf, xy_mlp, xy_af, vals, train_idx, test_idx,
            mae_raw, mae_rf, mae_mlp, mae_af)


def main():
    plt.rcParams.update(STYLE)
    (xy_raw, xy_rf, xy_mlp, xy_af, vals, train_idx, test_idx,
     mae_raw, mae_rf, mae_mlp, mae_af) = compute_or_load_cache()
    n = len(vals)
    print(f"\n[kNN val-loss MAE]  raw={mae_raw:.4f}  rf={mae_rf:.4f}  "
          f"mlp={mae_mlp:.4f}  forgeformer={mae_af:.4f}")

    # ── 1×4 figure (each panel 1:1, NeurIPS-sized) ──────────────────────
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    fig, axes = plt.subplots(
        1, 4, figsize=(7.5, 2.15),
        gridspec_kw=dict(wspace=0.10, left=0.005, right=0.90, top=0.90, bottom=0.02),
    )
    titles = [
        "(a) Raw features",
        "(b) RF proximity",
        "(c) MLP penultimate",
        "(d) Forge-Former (ours)",
    ]
    # Shared colour range, [5th, 95th] percentile so the dynamic range covers
    # most architectures — the rare worst-case tail saturates rather than
    # dragging every other point to the bottom of the colormap.
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))
    last_sc = None
    for ax, xy, t, m in zip(axes, [xy_raw, xy_rf, xy_mlp, xy_af], titles,
                             [mae_raw, mae_rf, mae_mlp, mae_af]):
        last_sc = plot_panel(ax, xy, vals, test_mask, t, m, vmin=vmin, vmax=vmax)

    # Slim colorbar at the right of the rightmost panel.
    cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                        fraction=0.018, pad=0.010, shrink=0.78)
    cbar.set_label("val_loss", fontsize=8.0, labelpad=2)
    cbar.ax.tick_params(labelsize=7.0, length=2.5, pad=1.5)
    cbar.outline.set_linewidth(0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_embeddings_tsne.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
