"""Predictor-verify table — search-efficiency comparison across datasets.

Evaluates ForgeFormer vs RF vs MLP (HW-GPT-Bench paper baseline `Net`) on:
  • our custom dataset  — dataset_200M.csv  (IHA search space, 9 fields)
  • HW-GPT-Bench gpt_l  — Sukthanker et al., NeurIPS D&B 2024  (5 fields)

For each dataset, RF and MLP are fit on the SAME train rows ForgeFormer was
trained on (sklearn train_test_split, same seed, same test_ratio). Metrics
reported per method:
  k@1%, k@5%   — attempts to fully recover the true top-1% / top-5% of test
  Spearman ρ   — global rank correlation
  Kendall τ    — global rank correlation
  MAE          — mean |pred − truth| over all test rows
  MAE@5%       — same, restricted to true top-5% of test (Pareto fidelity)

Writes:
  paper_plot/predictor_verify/table_a.md   (Markdown — paper / README)
  paper_plot/predictor_verify/table_a.tex  (LaTeX  — booktabs)
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR / "hw_gpt_bench"))

from surrogate.data import load_raw_arch_dataset, normalize_batch  # noqa: E402
from surrogate.inference import load_surrogate  # noqa: E402
from surrogate.model import ArchTransformerRanker  # noqa: E402
from data_loader import (  # noqa: E402
    SCALE_TO_MAX_LAYERS, load_hwgpt_dataset, load_hwgpt_archs,
    flatten_for_baselines, one_hot_features,
    FIELD_COUNT as HWGPT_FIELD_COUNT,
)


# ── Defaults ────────────────────────────────────────────────────────────────
CSV_PATH = PROJECT / "surrogate" / "dataset" / "dataset_200M.csv"
CUSTOM_CKPT = PROJECT / "surrogate" / "ckpts" / "forgeformer.pt"
HWGPT_CKPT_DIR = THIS_DIR / "hw_gpt_bench" / "ckpts"
BASELINES_DIR = THIS_DIR / "baselines"  # trained RF / MLP baselines per dataset

CUSTOM_SEED = 100
CUSTOM_TEST_RATIO = 0.2
# HW-GPT-Bench split locked to the legacy LLMArch_Predictor V1-on-HWGPT
# protocol (seed=7, test_ratio=0.3) so V1 / Paper-Net / RF / MLP are all on
# the same split the legacy paper reports.
HWGPT_SCALE = "l"
HWGPT_SEED = 7
HWGPT_TEST_RATIO = 0.3
N_LAYERS_CUSTOM = 40

LAYER_FIELDS_CUSTOM = [
    "n_head", "n_kv_group", "mlp_size",
    "n_qk_head_dim", "n_v_head_dim", "n_cproj", "attention_variant",
]
ATTN_MAP = {"identity": 0, "infinite": 1}


# ── Custom-dataset feature builder (per-layer raw, padded to 40 layers) ─────

def build_raw_features_custom(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        mask = r["_mask"]
        feats = [
            float(r["global_n_embd"]),
            float(r["global_block_size"]),
            int(bool(r["global_use_concat_heads"])),
            int(mask.sum()),
        ]
        for i in range(N_LAYERS_CUSTOM):
            active = bool(mask[i])
            for fld in LAYER_FIELDS_CUSTOM:
                v = r[f"layer{i}_{fld}"]
                v = ATTN_MAP.get(str(v), 0) if fld == "attention_variant" else float(v)
                feats.append(v if active else 0.0)
            feats.append(1.0 if active else 0.0)
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)


# ── Baseline model definition ──────────────────────────────────────────────
# RF: sklearn RandomForestRegressor (500 trees) — fit/load via joblib.
# MLP: HW-GPT-Bench paper baseline `Net` — 5 FCs / 4 hidden of width 128 /
#      ReLU / 1-d head. Trained with Adam(1e-3), MSE, batch=1024, 4000 epochs.
#      State-dict saved via torch.save; the matching StandardScaler is saved
#      as a sidecar joblib so prediction is deterministic on reload.

class _PaperNet(nn.Module):
    def __init__(self, nfeat: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(nfeat, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)


# ── Metrics ─────────────────────────────────────────────────────────────────

def k_at_top_pct(y_true: np.ndarray, y_pred: np.ndarray, pct: float):
    """Smallest k such that all of true top-pct% lie within predicted-top-k."""
    n = len(y_true)
    M = max(1, int(round(n * pct / 100.0)))
    true_top = set(np.argsort(y_true)[:M].tolist())
    order = np.argsort(y_pred)
    seen = 0
    for k_idx, idx in enumerate(order, start=1):
        if int(idx) in true_top:
            seen += 1
            if seen == M:
                return k_idx, M
    return None, M


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rho, _ = spearmanr(y_pred, y_true)
    tau, _ = kendalltau(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    k1, M1 = k_at_top_pct(y_true, y_pred, 1.0)
    k5, M5 = k_at_top_pct(y_true, y_pred, 5.0)
    top5_idx = np.argsort(y_true)[:M5]
    mae5 = float(np.mean(np.abs(y_pred[top5_idx] - y_true[top5_idx])))
    return dict(spearman=float(rho), kendall=float(tau), mae=mae,
                k1=k1, k5=k5, mae_top5=mae5, M1=M1, M5=M5)


# ── ForgeFormer prediction (custom and HW-GPT-Bench) ─────────────────────────

@torch.no_grad()
def forgeformer_predict_custom(test_idx: np.ndarray, device: torch.device,
                              ckpt_path: Path) -> np.ndarray:
    model, norm_stats, max_layers = load_surrogate(str(ckpt_path), device)
    raw = load_raw_arch_dataset(str(CSV_PATH), max_layers=max_layers)
    batch = normalize_batch(raw, norm_stats)
    x_te = batch.x[test_idx].to(device)
    mask_te = batch.padding_mask[test_idx].to(device)
    out = []
    for i in range(0, x_te.shape[0], 128):
        out.append(model(x_te[i:i+128], padding_mask=mask_te[i:i+128]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def forgeformer_predict_hwgpt(scale: str, ckpt_path: Path,
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (preds_test, y_test, train_idx, test_idx)."""
    import json
    config_path = ckpt_path.with_suffix(".json")
    with open(config_path) as f:
        cfg = json.load(f)["model"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ArchTransformerRanker(
        max_layers=cfg["max_layers"], d_model=cfg["d_model"],
        nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], field_count=cfg["field_count"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm_batch, _, ppls, info = load_hwgpt_dataset(scale)
    n = info["n"]
    train_idx = ckpt.get("train_idx")
    test_idx = ckpt.get("test_idx")
    if train_idx is None or test_idx is None:
        # Fallback: reproduce from saved seed/test_ratio
        with open(config_path) as f:
            tr = json.load(f)["training"]
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=tr["test_ratio"],
            random_state=tr["seed"], shuffle=True,
        )
    x_te = norm_batch.x_raw[test_idx].to(device)
    pad_te = norm_batch.padding_mask[test_idx].to(device)
    out = []
    for i in range(0, x_te.shape[0], 256):
        out.append(model(x_te[i:i+256], padding_mask=pad_te[i:i+256]).cpu().numpy())
    return (np.concatenate(out), ppls[test_idx],
            np.asarray(train_idx), np.asarray(test_idx))


# ── Trained-baseline persistence ───────────────────────────────────────────
# RF and MLP baselines are expensive to refit (RF: ~1 min, paper-baseline
# MLP: 4000 epochs / 1–5 min), so we save the fitted models to disk and
# reload them on subsequent runs. Storage layout:
#
#   baselines/<dataset>_rf.joblib            # sklearn RandomForestRegressor
#   baselines/<dataset>_mlp.pt               # PaperNet state_dict + nfeat
#   baselines/<dataset>_mlp_scaler.joblib    # StandardScaler used at fit time
#
# The ForgeFormer ckpts are stored separately under surrogate/ckpts/
# (custom dataset) and hw_gpt_bench/ckpts/ (HW-GPT-Bench).

import joblib  # noqa: E402


def _baseline_path(dataset_tag: str, method_tag: str, ext: str,
                   split_seed: int | None = None) -> Path:
    """Cache path for a fitted baseline.

    The split seed is part of the filename so multi-seed sweeps each get their
    own cache slot. Without this, every seed loads whichever baseline happens
    to be on disk and silently evaluates it on the wrong test rows (most of
    which were in the cached model's train set), inflating reported accuracy.
    """
    seed_tag = f"_seed{split_seed}" if split_seed is not None else ""
    return BASELINES_DIR / f"{dataset_tag}{seed_tag}_{method_tag}.{ext}"


def fit_or_load_rf(dataset_tag: str, X: np.ndarray, train_idx: np.ndarray,
                   y: np.ndarray, refresh: bool,
                   split_seed: int | None = None) -> RandomForestRegressor:
    path = _baseline_path(dataset_tag, "rf", "joblib", split_seed)
    if path.exists() and not refresh:
        print(f"  [load   ] RF → {path.name}")
        return joblib.load(path)
    print(f"  [fit    ] RF on {len(train_idx)} train rows ...")
    rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, path)
    print(f"            saved → {path.name}")
    return rf


def fit_or_load_mlp(dataset_tag: str, X: np.ndarray, train_idx: np.ndarray,
                    y: np.ndarray, device: torch.device, refresh: bool,
                    epochs: int = 4000, batch_size: int = 1024,
                    lr: float = 1e-3, hidden: int = 128, seed: int = 0,
                    standardize: bool = True,
                    split_seed: int | None = None,
                    ) -> Tuple["_PaperNet", "StandardScaler | None"]:
    """Fit (or load) the paper-baseline Net.

    standardize=True applies StandardScaler to the input (correct for raw-flat
    continuous features). standardize=False feeds the input verbatim — required
    for sparse one-hot encodings, where standardization destroys the 0/1
    structure and cripples the MLP (turns 0→−0.7, 1→+1.4 and the first ReLU
    layer kills the negative half before training can recover).
    """
    pt_path = _baseline_path(dataset_tag, "mlp", "pt", split_seed)
    sc_path = _baseline_path(dataset_tag, "mlp_scaler", "joblib", split_seed)
    if pt_path.exists() and not refresh:
        print(f"  [load   ] MLP → {pt_path.name}")
        scaler = joblib.load(sc_path) if sc_path.exists() else None
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model = _PaperNet(nfeat=ckpt["nfeat"], hidden=ckpt.get("hidden", hidden)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, scaler

    print(f"  [fit    ] MLP on {len(train_idx)} train rows "
          f"(epochs={epochs}, batch={batch_size}, lr={lr}, "
          f"standardize={standardize}) ...")
    if standardize:
        scaler = StandardScaler().fit(X[train_idx])
        Xtr_np = scaler.transform(X[train_idx])
    else:
        scaler = None
        Xtr_np = X[train_idx]
    Xtr = torch.from_numpy(Xtr_np).float().to(device)
    ytr = torch.from_numpy(y[train_idx]).float().to(device)
    torch.manual_seed(seed)
    model = _PaperNet(nfeat=X.shape[1], hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
    model.eval()
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "nfeat": X.shape[1], "hidden": hidden,
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "seed": seed, "standardize": standardize}, pt_path)
    if scaler is not None:
        joblib.dump(scaler, sc_path)
    elif sc_path.exists():
        sc_path.unlink()
    print(f"            saved → {pt_path.name}" + (f", {sc_path.name}" if scaler is not None else ""))
    return model, scaler


def predict_mlp(model: "_PaperNet", scaler: "StandardScaler | None",
                X: np.ndarray, idx: np.ndarray, device: torch.device) -> np.ndarray:
    Xs = scaler.transform(X[idx]) if scaler is not None else X[idx]
    Xte = torch.from_numpy(np.asarray(Xs)).float().to(device)
    model.eval()
    with torch.no_grad():
        return model(Xte).squeeze(-1).cpu().numpy()


# ── Per-dataset evaluation (loads or trains baselines as needed) ───────────

def evaluate_custom(device: torch.device, ckpt_path: Path,
                    seed: int, test_ratio: float, refresh: bool = False) -> Dict:
    df = pd.read_csv(CSV_PATH)
    df = df[np.isfinite(df["val_loss"].values)].reset_index(drop=True)
    df["_mask"] = df["global_layer_mask"].apply(ast.literal_eval).apply(np.array)
    vals = df["val_loss"].values
    n = len(vals)

    # Sanity: panel row count matches the surrogate trainer's loader, so the
    # train_test_split below reproduces the indices ForgeFormer trained on.
    raw_check = load_raw_arch_dataset(str(CSV_PATH), max_layers=N_LAYERS_CUSTOM)
    assert len(raw_check.x_raw) == n, "row-count drift between filters"

    train_idx, test_idx = train_test_split(np.arange(n), test_size=test_ratio,
                                           random_state=seed, shuffle=True)
    y_te = vals[test_idx]
    print(f"[custom] n={n}, train={len(train_idx)}, test={len(test_idx)}, seed={seed}")

    print("[custom] ForgeFormer ...")
    our_preds = forgeformer_predict_custom(test_idx, device, ckpt_path)

    X_raw = build_raw_features_custom(df)
    rf = fit_or_load_rf("custom", X_raw, train_idx, vals, refresh,
                        split_seed=seed)
    rf_preds = rf.predict(X_raw[test_idx])
    mlp, scaler = fit_or_load_mlp("custom", X_raw, train_idx, vals, device,
                                  refresh, split_seed=seed)
    mlp_preds = predict_mlp(mlp, scaler, X_raw, test_idx, device)

    return dict(
        n=n, n_test=len(test_idx),
        Ours=compute_metrics(y_te, our_preds),
        RF=compute_metrics(y_te, rf_preds),
        MLP=compute_metrics(y_te, mlp_preds),
    )


def evaluate_hwgpt(device: torch.device, ckpt_path: Path, scale: str,
                   refresh: bool = False) -> Dict:
    # Pull train_idx/test_idx from the ForgeFormer ckpt so baselines see the
    # same split. RF uses flat features; MLP (Paper-Net) uses the one-hot
    # encoding the HW-GPT-Bench paper proposes — that's what their `Net` was
    # designed for, and the gap to ForgeFormer collapses to a fair head-to-head.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_idx = np.asarray(ckpt.get("train_idx"))
    test_idx = np.asarray(ckpt.get("test_idx"))
    split_seed = ckpt.get("seed")
    n = train_idx.shape[0] + test_idx.shape[0]
    print(f"[hwgpt] gpt_{scale}: n={n}, train={len(train_idx)}, "
          f"test={len(test_idx)}, seed={split_seed}")

    archs, ppls = load_hwgpt_archs(scale)
    X_flat = flatten_for_baselines(archs, scale)
    X_oh = one_hot_features(archs, scale)
    y_te = ppls[test_idx]

    print(f"[hwgpt] ForgeFormer (gpt_{scale}) ...")
    our_preds, _, _, _ = forgeformer_predict_hwgpt(scale, ckpt_path, device)

    tag = f"hwgpt_{scale}"
    print(f"[hwgpt] RF (raw-flat features, dim={X_flat.shape[1]})")
    rf = fit_or_load_rf(tag, X_flat, train_idx, ppls, refresh,
                        split_seed=split_seed)
    rf_preds = rf.predict(X_flat[test_idx])
    print(f"[hwgpt] MLP (Paper-Net on one-hot input, dim={X_oh.shape[1]})")
    mlp, scaler = fit_or_load_mlp(tag, X_oh, train_idx, ppls, device, refresh,
                                  standardize=False, split_seed=split_seed)
    mlp_preds = predict_mlp(mlp, scaler, X_oh, test_idx, device)

    return dict(
        n=n, n_test=len(test_idx),
        Ours=compute_metrics(y_te, our_preds),
        RF=compute_metrics(y_te, rf_preds),
        MLP=compute_metrics(y_te, mlp_preds),
    )


# ── Table rendering ─────────────────────────────────────────────────────────

# Methods within each dataset block. Order matches the user's table sketch.
METHOD_ORDER = ("Ours", "MLP", "RF")
METHOD_LABEL = {
    "Ours": "ForgeFormer",
    "MLP":  "MLP",
    "RF":   "RF",
}
# Metric row order (top → bottom).
ROW_ORDER = ("mae", "mae_top5", "spearman", "kendall", "k1", "k5")
ROW_LABEL_MD = {
    "mae":      "MAE",
    "mae_top5": "MAE@5%",
    "spearman": r"Spearman $\rho$",
    "kendall":  r"Kendall $\tau$",
    "k1":       r"$k$@1%",
    "k5":       r"$k$@5%",
}
ROW_LABEL_TEX = {
    "mae":      "MAE",
    "mae_top5": r"MAE@5\%",
    "spearman": r"Spearman $\rho$",
    "kendall":  r"Kendall $\tau$",
    "k1":       r"$k$@1\%",
    "k5":       r"$k$@5\%",
}
ROW_FMT = {
    "mae":      lambda v: f"{v:.4f}",
    "mae_top5": lambda v: f"{v:.4f}",
    "spearman": lambda v: f"{v:+.4f}",
    "kendall":  lambda v: f"{v:+.4f}",
    "k1":       lambda v: f"{v}" if v is not None else "—",
    "k5":       lambda v: f"{v}" if v is not None else "—",
}


def _row_values(custom_res: Dict, hwgpt_res: Dict, row_key: str) -> List[str]:
    """Return formatted cell values for one metric row, across all method × dataset columns."""
    out = []
    for ds in (custom_res, hwgpt_res):
        for m in METHOD_ORDER:
            out.append(ROW_FMT[row_key](ds[m][row_key]))
    return out


def render_markdown(custom_res: Dict, hwgpt_res: Dict, scale: str) -> str:
    n_methods = len(METHOD_ORDER)
    n_cols = 1 + 2 * n_methods

    # Method names get the dataset suffix only on the first column of each group
    # (Markdown doesn't support colspan; this keeps the table readable).
    h_methods = ["Metric"]
    h_methods.append(f"{METHOD_LABEL[METHOD_ORDER[0]]} (IHA)")
    for m in METHOD_ORDER[1:]:
        h_methods.append(METHOD_LABEL[m])
    h_methods.append(f"{METHOD_LABEL[METHOD_ORDER[0]]} (HW-GPT-Bench)")
    for m in METHOD_ORDER[1:]:
        h_methods.append(METHOD_LABEL[m])

    rows = [h_methods]
    for r in ROW_ORDER:
        rows.append([ROW_LABEL_MD[r]] + _row_values(custom_res, hwgpt_res, r))

    widths = [max(len(row[i]) for row in rows) for i in range(n_cols)]

    def fmt_row(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(n_cols)) + " |"

    lines = [
        fmt_row(rows[0]),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows[1:]:
        lines.append(fmt_row(row))

    custom_info = (f"IHA dataset: n={custom_res['n']}, n_test={custom_res['n_test']}, "
                   f"M_1%={custom_res['Ours']['M1']}, M_5%={custom_res['Ours']['M5']}, "
                   f"seed={CUSTOM_SEED}")
    hwgpt_info = (f"HW-GPT-Bench gpt_{scale}: n={hwgpt_res['n']}, n_test={hwgpt_res['n_test']}, "
                  f"M_1%={hwgpt_res['Ours']['M1']}, M_5%={hwgpt_res['Ours']['M5']}, "
                  f"seed={HWGPT_SEED}")

    return (
        "# Predictor accuracy on held-out test\n\n"
        f"_{custom_info}; {hwgpt_info}; both 80/20 random splits._\n\n"
        "RF and MLP baselines are fit on the **same train rows** ForgeFormer was "
        "trained on. MLP follows the HW-GPT-Bench paper baseline `Net` "
        "(5 FCs / hidden 128 / Adam(1e-3) / MSE / batch 1024 / 4000 epochs). "
        "$k$@$X$% is the smallest $k$ such that all of true-top-$X$% sit within "
        "predicted-top-$k$. MAE@5% is over the true top-5% subset (Pareto fidelity).\n\n"
        + "\n".join(lines) + "\n"
    )


def render_latex(custom_res: Dict, hwgpt_res: Dict, scale: str) -> str:
    n_methods = len(METHOD_ORDER)
    col_spec = "l" + "c" * (2 * n_methods)
    cmid_a = f"\\cmidrule(lr){{2-{1 + n_methods}}}"
    cmid_b = f"\\cmidrule(lr){{{2 + n_methods}-{1 + 2 * n_methods}}}"

    method_header = (
        " & " + " & ".join(METHOD_LABEL[m] for m in METHOD_ORDER)
        + " & " + " & ".join(METHOD_LABEL[m] for m in METHOD_ORDER)
    )

    body_rows = []
    for r in ROW_ORDER:
        cells = [ROW_LABEL_TEX[r]] + _row_values(custom_res, hwgpt_res, r)
        cells = [c.replace("&", r"\&") for c in cells]
        body_rows.append(" & ".join(cells) + r" \\")

    return (
        "% Auto-generated by paper_plot/predictor_verify/table_a.py — do not hand-edit.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{Predictor accuracy on held-out test (80/20 split). "
        f"RF and MLP baselines are fit on the same train rows ForgeFormer was "
        f"trained on; the MLP is the HW-GPT-Bench paper baseline \\texttt{{Net}}. "
        f"$k$@$X$\\% is the smallest $k$ such that all of true-top-$X$\\% sit "
        f"within predicted-top-$k$. MAE on the entire test; MAE@5\\% on the true "
        f"top-5\\% subset (Pareto fidelity). IHA dataset: $n$={custom_res['n']}, "
        f"$n_\\text{{test}}$={custom_res['n_test']}; HW-GPT-Bench gpt\\_{scale}: "
        f"$n$={hwgpt_res['n']}, $n_\\text{{test}}$={hwgpt_res['n_test']}.}}\n"
        "\\label{tab:predictor_accuracy}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f" & \\multicolumn{{{n_methods}}}{{c}}{{IHA dataset}}"
        f" & \\multicolumn{{{n_methods}}}{{c}}{{HW-GPT-Bench gpt\\_{scale}}} \\\\\n"
        f"{cmid_a} {cmid_b}\n"
        f"{method_header} \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--custom_seed", type=int, default=CUSTOM_SEED)
    p.add_argument("--custom_test_ratio", type=float, default=CUSTOM_TEST_RATIO)
    p.add_argument("--custom_ckpt", type=Path, default=CUSTOM_CKPT)
    p.add_argument("--hwgpt_scale", default=HWGPT_SCALE, choices=("s", "m", "l"))
    p.add_argument("--hwgpt_ckpt", type=Path, default=None)
    p.add_argument("--out_md",  type=Path, default=THIS_DIR / "table_a.md")
    p.add_argument("--out_tex", type=Path, default=THIS_DIR / "table_a.tex")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore the cache and recompute all predictions.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.hwgpt_ckpt is None:
        args.hwgpt_ckpt = HWGPT_CKPT_DIR / f"forgeformer_hwgpt_{args.hwgpt_scale}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.refresh:
        print("--refresh: ignoring prediction cache")

    print("=" * 60)
    print("Custom dataset")
    print("=" * 60)
    custom = evaluate_custom(device, args.custom_ckpt,
                             args.custom_seed, args.custom_test_ratio,
                             refresh=args.refresh)

    print()
    print("=" * 60)
    print(f"HW-GPT-Bench gpt_{args.hwgpt_scale}")
    print("=" * 60)
    hwgpt = evaluate_hwgpt(device, args.hwgpt_ckpt, args.hwgpt_scale,
                           refresh=args.refresh)

    md = render_markdown(custom, hwgpt, args.hwgpt_scale)
    tex = render_latex(custom, hwgpt, args.hwgpt_scale)
    args.out_md.write_text(md)
    args.out_tex.write_text(tex)

    print()
    print(md)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
