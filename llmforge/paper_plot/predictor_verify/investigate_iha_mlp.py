"""Investigate why Paper-Net MLP underperforms on the IHA dataset.

Hypothesis (confirmed for HW-GPT-Bench): the gap is mostly *input encoding*.
Raw flat numeric features force a linear ordering on categorical fields
(`n_head=8` ≠ "twice as good as `n_head=4`"); one-hot makes each value an
independent feature.

This script trains Paper-Net under several IHA input variants and reports
side-by-side metrics so we can attribute the gap to encoding vs. capacity vs.
data scale.
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR))

from table_a import (  # noqa: E402
    CSV_PATH, build_raw_features_custom, compute_metrics, _PaperNet,
)

CSV_PATH = PROJECT / "surrogate" / "dataset" / "dataset_200M.csv"
SEED = 100
TEST_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAYERS = 40

# IHA field domains (observed in dataset_200M.csv).
N_HEAD_VALS = [1, 2, 3, 4, 6, 8, 12, 16]
N_KV_VALS = [1, 2, 3, 4, 6, 8, 12, 16]
MLP_SIZE_VALS = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
                 2816, 3072, 3328, 3584, 3840, 4096]
QK_DIM_VALS = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
V_DIM_VALS = QK_DIM_VALS
ATTN_VALS = ["identity", "infinite"]


def build_iha_one_hot(df: pd.DataFrame) -> np.ndarray:
    """Per-layer one-hot for the IHA dataset.

    Per active layer: n_head (8) + n_kv (8) + mlp_size (15) + qk_dim (15)
    + v_dim (15) + attn_variant (2) + is_active (1) = 64 dims.
    Plus n_active_layers one-hot (40 dims). Padded layers contribute zeros.

    Total dim ≈ 64 × 40 + 40 = 2600.
    """
    rows = []
    per_layer_dim = (len(N_HEAD_VALS) + len(N_KV_VALS) + len(MLP_SIZE_VALS)
                     + len(QK_DIM_VALS) + len(V_DIM_VALS) + len(ATTN_VALS) + 1)
    n_active_dim = N_LAYERS  # one-hot over active-layer count 1..40

    for _, r in df.iterrows():
        mask = r["_mask"]
        n_active = int(mask.sum())
        feats = np.zeros(per_layer_dim * N_LAYERS + n_active_dim, dtype=np.float32)
        # n_active_layers one-hot
        if 1 <= n_active <= N_LAYERS:
            feats[per_layer_dim * N_LAYERS + n_active - 1] = 1.0
        for li in range(N_LAYERS):
            if not mask[li]:
                continue
            base = per_layer_dim * li
            o = 0
            try:
                feats[base + o + N_HEAD_VALS.index(int(r[f"layer{li}_n_head"]))] = 1.0
            except ValueError:
                pass
            o += len(N_HEAD_VALS)
            try:
                feats[base + o + N_KV_VALS.index(int(r[f"layer{li}_n_kv_group"]))] = 1.0
            except ValueError:
                pass
            o += len(N_KV_VALS)
            try:
                feats[base + o + MLP_SIZE_VALS.index(int(r[f"layer{li}_mlp_size"]))] = 1.0
            except ValueError:
                pass
            o += len(MLP_SIZE_VALS)
            try:
                feats[base + o + QK_DIM_VALS.index(int(r[f"layer{li}_n_qk_head_dim"]))] = 1.0
            except ValueError:
                pass
            o += len(QK_DIM_VALS)
            try:
                feats[base + o + V_DIM_VALS.index(int(r[f"layer{li}_n_v_head_dim"]))] = 1.0
            except ValueError:
                pass
            o += len(V_DIM_VALS)
            attn = str(r[f"layer{li}_attention_variant"])
            if attn in ATTN_VALS:
                feats[base + o + ATTN_VALS.index(attn)] = 1.0
            o += len(ATTN_VALS)
            feats[base + o] = 1.0  # is_active
        rows.append(feats)
    return np.stack(rows)


def fit_papernet(X, train_idx, test_idx, y, *, standardize: bool,
                 weight_decay: float = 0.0, epochs: int = 4000,
                 batch_size: int = 1024, lr: float = 1e-3,
                 hidden: int = 128, seed: int = 0) -> np.ndarray:
    if standardize:
        sc = StandardScaler().fit(X[train_idx])
        Xtr_np = sc.transform(X[train_idx])
        Xte_np = sc.transform(X[test_idx])
    else:
        Xtr_np = X[train_idx]
        Xte_np = X[test_idx]
    Xtr = torch.from_numpy(np.asarray(Xtr_np)).float().to(DEVICE)
    Xte = torch.from_numpy(np.asarray(Xte_np)).float().to(DEVICE)
    ytr = torch.from_numpy(y[train_idx]).float().to(DEVICE)

    torch.manual_seed(seed)
    model = _PaperNet(nfeat=X.shape[1], hidden=hidden).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    model.train()
    t0 = time.time()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
    print(f"  fit took {time.time()-t0:.0f}s")
    model.eval()
    with torch.no_grad():
        return model(Xte).squeeze(-1).cpu().numpy(), \
               model(Xtr).squeeze(-1).cpu().numpy()


def report(label, y_test, preds_test, y_train, preds_train):
    m = compute_metrics(y_test, preds_test)
    train_mae = float(np.mean(np.abs(preds_train - y_train)))
    print(f"{label:<40}  train MAE={train_mae:.4f}  test MAE={m['mae']:.4f}  "
          f"ρ={m['spearman']:+.4f}  τ={m['kendall']:+.4f}  "
          f"k@1%={m['k1']}  k@5%={m['k5']}  MAE@5%={m['mae_top5']:.4f}")


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[np.isfinite(df["val_loss"].values)].reset_index(drop=True)
    df["_mask"] = df["global_layer_mask"].apply(ast.literal_eval).apply(np.array)
    vals = df["val_loss"].values
    n = len(vals)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=TEST_RATIO,
                                           random_state=SEED, shuffle=True)
    print(f"IHA: n={n}, train={len(train_idx)}, test={len(test_idx)}, seed={SEED}\n")

    print("Building features ...")
    X_raw = build_raw_features_custom(df)
    X_oh = build_iha_one_hot(df)
    print(f"  raw flat dim   = {X_raw.shape[1]}")
    print(f"  one-hot dim    = {X_oh.shape[1]}")
    print(f"  train rows     = {len(train_idx)}\n")

    y_tr = vals[train_idx]
    y_te = vals[test_idx]

    print("=== Paper-Net variants on IHA ===")

    # Variant 1: raw flat + StandardScaler (current setup in the table)
    pte, ptr = fit_papernet(X_raw, train_idx, test_idx, vals, standardize=True)
    report("Paper-Net | raw flat + StandardScaler", y_te, pte, y_tr, ptr)

    # Variant 2: raw flat + StandardScaler + weight decay (regularize)
    pte, ptr = fit_papernet(X_raw, train_idx, test_idx, vals, standardize=True, weight_decay=1e-3)
    report("Paper-Net | raw flat + StdScaler + WD=1e-3", y_te, pte, y_tr, ptr)

    # Variant 3: one-hot + no scaler (legacy paper baseline recipe)
    pte, ptr = fit_papernet(X_oh, train_idx, test_idx, vals, standardize=False)
    report("Paper-Net | one-hot (no scaler)", y_te, pte, y_tr, ptr)

    # Variant 4: one-hot + weight decay (helps with the high input dim)
    pte, ptr = fit_papernet(X_oh, train_idx, test_idx, vals, standardize=False, weight_decay=1e-3)
    report("Paper-Net | one-hot + WD=1e-3", y_te, pte, y_tr, ptr)


if __name__ == "__main__":
    main()
