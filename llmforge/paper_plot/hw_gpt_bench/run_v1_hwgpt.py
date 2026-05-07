"""Apply our V1 predictor (PerFieldMLPRanker) to HW-GPT-Bench (gpt_s scale).

HW-GPT-Bench (Sukthanker et al., NeurIPS D&B 2024) releases 10,000
(architecture → measured perplexity) pairs per scale in
`data_collection/gpt_datasets/gpt_{s,m,l}/stats.pkl` — a dict keyed by a
dash-separated architecture string.

We port the dataset into our `ArchBatchRaw` layout (per-layer token with a
padding mask) and run our V1 tokeniser against two baselines on the same
train/test split, reporting Spearman, Kendall, L1 MAE, and top-k discovery
metrics.

Scale: "s"  →  12 layers max, search space:
  embed_dim ∈ {192, 384, 768}   (global)
  n_layer   ∈ {10, 11, 12}       (global; controls padding)
  mlp_ratio ∈ {2, 3, 4}           (per layer)
  n_head    ∈ {4, 8, 12}          (per layer)
  bias      ∈ {True, False}       (global)

Per-token feature vector (FIELD_COUNT=5):
    [ mlp_ratio, n_head, is_active, embed_dim, bias ]
with is_active=1 for real layers, 0 for padding; embed_dim and bias are
broadcast across all max_layers positions.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("$HOME/hw-gpt-bench")
OURS = Path("$HOME/LLMArch_Predictor")
HERE = OURS / "final_paper_plots" / "predictor_verify" / "hw_gpt_bench"
sys.path.insert(0, str(OURS))
sys.path.insert(0, str(OURS / "final_paper_plots" / "predictor_verify"))
sys.path.insert(0, str(OURS / "final_paper_plots" / "predictor_verify" / "embedding_ablation"))

# Reuse our V1 model verbatim
from models import PerFieldMLPRanker, absolute_loss  # noqa: E402

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("scale", nargs="?", default="s")
_p.add_argument("--epochs", type=int, default=200)
_p.add_argument("--batch", type=int, default=64)
_p.add_argument("--lr", type=float, default=1e-4)
_p.add_argument("--tag", type=str, default="")
_p.add_argument("--d_model", type=int, default=64)
_p.add_argument("--n_layer", type=int, default=4)
_p.add_argument("--n_head", type=int, default=4)
_args = _p.parse_args()
SCALE = _args.scale
MAX_LAYERS = {"s": 12, "m": 24, "l": 36}[SCALE]
FIELD_COUNT = 5  # mlp_ratio, n_head, is_active, embed_dim, bias
SEED = 7
TEST_RATIO = 0.3
EPOCHS = _args.epochs
BATCH = _args.batch
LR = _args.lr
TAG = _args.tag
D_MODEL = _args.d_model
N_LAYER = _args.n_layer
N_HEAD = _args.n_head
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Parse an HW-GPT-Bench architecture string
# Format (scale s):  "gpt-s-<n_layer>-<embed_dim>-<mlp*n_layer>-<head*n_layer>-<bias>"
# --------------------------------------------------------------------------- #
def parse_arch_str(arch_str: str):
    parts = arch_str.split("-")
    assert parts[0] == "gpt"
    assert parts[1] == SCALE
    n_layer = int(parts[2])
    embed_dim = int(parts[3])
    mlp_ratios = [int(x) for x in parts[4:4 + n_layer]]
    n_heads = [int(x) for x in parts[4 + n_layer:4 + 2 * n_layer]]
    bias = 1 if parts[-1] == "True" else 0
    return dict(n_layer=n_layer, embed_dim=embed_dim,
                mlp_ratios=mlp_ratios, n_heads=n_heads, bias=bias)


def load_dataset():
    with open(REPO / f"data_collection/gpt_datasets/gpt_{SCALE}/stats.pkl", "rb") as f:
        data = pickle.load(f)
    archs, ppls = [], []
    for arch_str, metrics in data.items():
        ppl = metrics.get("perplexity")
        if ppl is None or not np.isfinite(float(ppl)):
            continue
        archs.append(parse_arch_str(arch_str))
        ppls.append(float(ppl))
    return archs, np.asarray(ppls, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Tensorise: (N, L, 5) raw, plus padding mask
# --------------------------------------------------------------------------- #
def build_tensors(archs):
    n = len(archs)
    x = np.zeros((n, MAX_LAYERS, FIELD_COUNT), dtype=np.float32)
    pad = np.ones((n, MAX_LAYERS), dtype=bool)          # True = padding
    for i, a in enumerate(archs):
        L = a["n_layer"]
        for j in range(L):
            x[i, j, 0] = a["mlp_ratios"][j]
            x[i, j, 1] = a["n_heads"][j]
            x[i, j, 2] = 1.0                             # is_active
            x[i, j, 3] = a["embed_dim"]                  # broadcast global
            x[i, j, 4] = a["bias"]                       # broadcast global
        pad[i, :L] = False
    return x, pad


def normalize_per_field(x_raw: np.ndarray, pad: np.ndarray):
    """Min-max normalize each field across *active* positions only."""
    mask_flat = (~pad).reshape(-1)
    x_flat = x_raw.reshape(-1, FIELD_COUNT)
    mins = x_flat[mask_flat].min(axis=0)
    maxs = x_flat[mask_flat].max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
    x_norm = (x_raw - mins) / span
    x_norm = np.where(pad[..., None], 0.0, x_norm).astype(np.float32)
    return x_norm, mins, span


# --------------------------------------------------------------------------- #
# Flattened feature vector for RF / MLP baselines
# layout: [embed_dim, bias, n_layer, *mlp(12 padded -1), *head(12 padded -1)]
# --------------------------------------------------------------------------- #
def flatten_for_baselines(archs):
    rows = []
    for a in archs:
        L = a["n_layer"]
        v = [a["embed_dim"], a["bias"], L]
        v += a["mlp_ratios"] + [-1] * (MAX_LAYERS - L)
        v += a["n_heads"]    + [-1] * (MAX_LAYERS - L)
        rows.append(v)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def k_recover_pct(y_true, y_pred, pct):
    """Smallest k such that all of the true top-pct% test archs appear in the
    predictor's top-k ranking. N=len(y_true); M=ceil(pct/100 * N)."""
    n = len(y_true)
    m = max(1, int(np.ceil(pct / 100.0 * n)))
    true_top = set(np.argsort(y_true)[:m].tolist())
    order = np.argsort(y_pred)
    seen = 0
    for k, idx in enumerate(order):
        if int(idx) in true_top:
            seen += 1
            if seen == m:
                return k + 1
    return None


def report(name, y_true, y_pred):
    rho, _ = spearmanr(y_pred, y_true)
    tau, _ = kendalltau(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    k1 = k_recover_pct(y_true, y_pred, 1.0)
    k2 = k_recover_pct(y_true, y_pred, 2.0)
    k5 = k_recover_pct(y_true, y_pred, 5.0)
    # MAE on the true top-1% (Pareto-frontier fidelity)
    m = max(1, int(np.ceil(0.01 * len(y_true))))
    top_idx = np.argsort(y_true)[:m]
    top_mae = float(np.mean(np.abs(y_pred[top_idx] - y_true[top_idx])))
    print(f"  {name:<22}  ρ={rho:.4f}  τ={tau:.4f}  MAE={mae:.4f}  "
          f"k@1%={k1 or '—':>4}  k@2%={k2 or '—':>4}  k@5%={k5 or '—':>5}  "
          f"top1%-MAE={top_mae:.4f}")
    return dict(method=name, spearman=float(rho), kendall=float(tau), mae=mae,
                k1pct=k1, k2pct=k2, k5pct=k5, top1pct_mae=top_mae)


# --------------------------------------------------------------------------- #
# Train / eval V1
# --------------------------------------------------------------------------- #
def train_v1(x_tr, pad_tr, y_tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PerFieldMLPRanker(
        max_layers=MAX_LAYERS, d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYER, dropout=0.2
    ).to(DEVICE)
    model.field_proj = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(1, D_MODEL), torch.nn.GELU(),
                            torch.nn.Linear(D_MODEL, D_MODEL))
        for _ in range(FIELD_COUNT)
    ]).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(pad_tr),
                      torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True,
    )
    model.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        running = 0.0
        for x, m, y in loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            pred = model(x, padding_mask=m)
            loss = absolute_loss(pred, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.shape[0]
        log_every = max(25, EPOCHS // 16)
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    epoch {ep+1:>3}/{EPOCHS}  train L1 = {running/len(y_tr):.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def predict_v1(model, x, pad):
    model.eval()
    outs = []
    for i in range(0, x.shape[0], 256):
        xb = torch.from_numpy(x[i:i+256]).to(DEVICE)
        pb = torch.from_numpy(pad[i:i+256]).to(DEVICE)
        outs.append(model(xb, padding_mask=pb).cpu().numpy())
    return np.concatenate(outs)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    print(f"[HW-GPT-Bench] scale={SCALE}  device={DEVICE}")
    archs, ppls = load_dataset()
    print(f"Loaded {len(archs)} architectures. ppl min={ppls.min():.2f} "
          f"max={ppls.max():.2f} mean={ppls.mean():.2f}")

    x_raw, pad = build_tensors(archs)
    x_norm, mins, span = normalize_per_field(x_raw, pad)
    X_flat = flatten_for_baselines(archs)

    n = len(archs)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=TEST_RATIO, random_state=SEED, shuffle=True)
    print(f"split: n_train={len(train_idx)}  n_test={len(test_idx)}")

    y = ppls
    y_tr, y_te = y[train_idx], y[test_idx]

    # ---- our V1 ----
    print("\n[V1] training PerFieldMLPRanker…")
    suffix = f"_{TAG}" if TAG else ""
    ckpt_path = HERE / f"hwgpt_{SCALE}_v1{suffix}.pt"
    model = train_v1(x_norm[train_idx], pad[train_idx], y_tr, SEED)
    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(max_layers=MAX_LAYERS, d_model=D_MODEL,
                       nhead=N_HEAD, num_layers=N_LAYER, field_count=FIELD_COUNT),
        "norm": dict(mins=mins, span=span),
        "train_idx": train_idx, "test_idx": test_idx, "seed": SEED,
    }, ckpt_path)
    print(f"[V1] saved checkpoint → {ckpt_path}")
    pred_v1 = predict_v1(model, x_norm, pad)[test_idx]

    # ---- RF on flattened ----
    print("\n[RF] fitting RandomForestRegressor(500)…")
    rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(X_flat[train_idx], y_tr)
    pred_rf = rf.predict(X_flat[test_idx])

    # ---- MLP on flattened ----
    print("\n[MLP] fitting MLPRegressor((128, 64))…")
    sc = StandardScaler().fit(X_flat[train_idx])
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                       random_state=0, alpha=1e-3, early_stopping=True)
    mlp.fit(sc.transform(X_flat[train_idx]), y_tr)
    pred_mlp = mlp.predict(sc.transform(X_flat[test_idx]))

    # ---- report ----
    print(f"\n=== Held-out metrics on HW-GPT-Bench gpt_{SCALE}  "
          f"(n_train={len(train_idx)}, n_test={len(test_idx)}) ===")
    rows = [
        report("V1 (Ours)", y_te, pred_v1),
        report("Random Forest",  y_te, pred_rf),
        report("MLP (sklearn)",  y_te, pred_mlp),
    ]
    pd.DataFrame(rows).to_csv(HERE / f"hwgpt_{SCALE}_results{suffix}.csv", index=False)
    np.savez(HERE / f"hwgpt_{SCALE}_preds{suffix}.npz",
             y_true=y_te, v1=pred_v1, rf=pred_rf, mlp=pred_mlp)
    print(f"\nWrote {HERE}/hwgpt_{SCALE}_results{suffix}.csv")
    print(f"Wrote {HERE}/hwgpt_{SCALE}_preds{suffix}.npz")


if __name__ == "__main__":
    main()
