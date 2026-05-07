"""Sanity check baselines vs ForgeFormer.

Two checks:
  (a) Diagnostic on the saved baselines: train-fit quality vs test metrics. If
      a baseline is overfit / under-trained, train_L1 will diverge from test_L1
      in a tell-tale way.
  (b) Train HW-GPT-Bench MLP with the *paper's native one-hot encoding*
      (instead of our raw-flat features) and compare. Legacy LLMArch_Predictor
      reported k@1%=45 with one-hot vs k@1%=801 with raw flat — i.e., the
      input encoding accounts for most of the MLP's gap to ForgeFormer.

This script does NOT touch the saved baselines or the table outputs; it
prints a side-by-side report so you can decide whether to swap the MLP
input encoding for a fairer head-to-head.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR / "hw_gpt_bench"))
sys.path.insert(0, str(THIS_DIR))

import joblib  # noqa: E402

from data_loader import (  # noqa: E402
    SCALE_TO_MAX_LAYERS, load_hwgpt_archs, load_hwgpt_dataset, build_batch,
    flatten_for_baselines,
)
from table_a import (  # noqa: E402
    BASELINES_DIR, CSV_PATH, build_raw_features_custom, compute_metrics,
    _PaperNet, fit_or_load_rf, fit_or_load_mlp,  # noqa: E402
)
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── HW-GPT-Bench native one-hot encoding (verbatim from paper_baseline.py) ──
SEARCH_SPACES = {
    "s": dict(embed_dim_choices=[192, 384, 768], n_layer_choices=[10, 11, 12],
              mlp_ratio_choices=[2, 3, 4], n_head_choices=[4, 8, 12],
              bias_choices=["True", "False"]),
    "m": dict(embed_dim_choices=[256, 512, 1024], n_layer_choices=[22, 23, 24],
              mlp_ratio_choices=[2, 3, 4], n_head_choices=[8, 12, 16],
              bias_choices=["True", "False"]),
    "l": dict(embed_dim_choices=[320, 640, 1280], n_layer_choices=[34, 35, 36],
              mlp_ratio_choices=[2, 3, 4], n_head_choices=[8, 16, 20],
              bias_choices=["True", "False"]),
}


def hwgpt_one_hot(arch: dict, scale: str) -> torch.Tensor:
    cd = SEARCH_SPACES[scale]
    max_layers = max(cd["n_layer_choices"])
    e = torch.zeros(len(cd["embed_dim_choices"]))
    l = torch.zeros(len(cd["n_layer_choices"]))
    m = torch.zeros(max_layers, len(cd["mlp_ratio_choices"]))
    h = torch.zeros(max_layers, len(cd["n_head_choices"]))
    b = torch.zeros(len(cd["bias_choices"]))
    e[cd["embed_dim_choices"].index(arch["embed_dim"])] = 1
    l[cd["n_layer_choices"].index(arch["n_layer"])] = 1
    for i in range(arch["n_layer"]):
        m[i][cd["mlp_ratio_choices"].index(arch["mlp_ratios"][i])] = 1
        h[i][cd["n_head_choices"].index(arch["n_heads"][i])] = 1
    bias_str = "True" if arch["bias"] == 1 else "False"
    b[cd["bias_choices"].index(bias_str)] = 1
    return torch.cat([e, l, m.view(-1), h.view(-1), b])


# ── Diagnostic: train-fit quality on the saved baselines ────────────────────

def _train_fit_quality(model_predict_fn, X, train_idx, y_true_train) -> Tuple[float, float]:
    preds = model_predict_fn(X[train_idx])
    err = preds - y_true_train
    return float(np.mean(np.abs(err))), float(np.std(err))


def diagnose_custom():
    print("\n=== (a) Diagnostic: custom (IHA) baselines on TRAINING fit ===")
    df = pd.read_csv(CSV_PATH)
    df = df[np.isfinite(df["val_loss"].values)].reset_index(drop=True)
    import ast
    df["_mask"] = df["global_layer_mask"].apply(ast.literal_eval).apply(np.array)
    vals = df["val_loss"].values
    n = len(vals)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=100, shuffle=True)
    X_raw = build_raw_features_custom(df)

    rf = joblib.load(BASELINES_DIR / "custom_rf.joblib")
    rf_train_mae = float(np.mean(np.abs(rf.predict(X_raw[train_idx]) - vals[train_idx])))
    rf_test_mae = float(np.mean(np.abs(rf.predict(X_raw[test_idx]) - vals[test_idx])))
    print(f"  RF  train MAE = {rf_train_mae:.4f}  |  test MAE = {rf_test_mae:.4f}")

    sc = joblib.load(BASELINES_DIR / "custom_mlp_scaler.joblib")
    ckpt = torch.load(BASELINES_DIR / "custom_mlp.pt", map_location=DEVICE, weights_only=False)
    mlp = _PaperNet(nfeat=ckpt["nfeat"], hidden=ckpt.get("hidden", 128)).to(DEVICE)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()
    with torch.no_grad():
        Xtr = torch.from_numpy(sc.transform(X_raw[train_idx])).float().to(DEVICE)
        Xte = torch.from_numpy(sc.transform(X_raw[test_idx])).float().to(DEVICE)
        mlp_train = mlp(Xtr).squeeze(-1).cpu().numpy()
        mlp_test = mlp(Xte).squeeze(-1).cpu().numpy()
    print(f"  MLP train MAE = {float(np.mean(np.abs(mlp_train - vals[train_idx]))):.4f}  "
          f"|  test MAE = {float(np.mean(np.abs(mlp_test - vals[test_idx]))):.4f}")


def diagnose_hwgpt():
    print("\n=== (a) Diagnostic: HW-GPT-Bench baselines on TRAINING fit ===")
    archs, ppls = load_hwgpt_archs("l")
    X_flat = flatten_for_baselines(archs, "l")
    n = len(ppls)
    ckpt_path = THIS_DIR / "hw_gpt_bench" / "ckpts" / "forgeformer_hwgpt_l.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_idx = np.asarray(ck["train_idx"]); test_idx = np.asarray(ck["test_idx"])

    rf = joblib.load(BASELINES_DIR / "hwgpt_l_rf.joblib")
    rf_train_mae = float(np.mean(np.abs(rf.predict(X_flat[train_idx]) - ppls[train_idx])))
    rf_test_mae = float(np.mean(np.abs(rf.predict(X_flat[test_idx]) - ppls[test_idx])))
    print(f"  RF  train MAE = {rf_train_mae:.4f}  |  test MAE = {rf_test_mae:.4f}")

    sc = joblib.load(BASELINES_DIR / "hwgpt_l_mlp_scaler.joblib")
    ckpt = torch.load(BASELINES_DIR / "hwgpt_l_mlp.pt", map_location=DEVICE, weights_only=False)
    mlp = _PaperNet(nfeat=ckpt["nfeat"], hidden=ckpt.get("hidden", 128)).to(DEVICE)
    mlp.load_state_dict(ckpt["state_dict"]); mlp.eval()
    with torch.no_grad():
        Xtr = torch.from_numpy(sc.transform(X_flat[train_idx])).float().to(DEVICE)
        Xte = torch.from_numpy(sc.transform(X_flat[test_idx])).float().to(DEVICE)
        mlp_train = mlp(Xtr).squeeze(-1).cpu().numpy()
        mlp_test = mlp(Xte).squeeze(-1).cpu().numpy()
    print(f"  MLP (raw flat) train MAE = {float(np.mean(np.abs(mlp_train - ppls[train_idx]))):.4f}  "
          f"|  test MAE = {float(np.mean(np.abs(mlp_test - ppls[test_idx]))):.4f}")


# ── (b) HW-GPT-Bench MLP with native one-hot encoding ──────────────────────

def fit_mlp_hwgpt_onehot(scale: str = "l", epochs: int = 4000,
                        batch_size: int = 1024, lr: float = 1e-3,
                        seed: int = 42) -> dict:
    print(f"\n=== (b) Re-fit MLP on HW-GPT-Bench gpt_{scale} with native one-hot input ===")
    archs, ppls = load_hwgpt_archs(scale)
    X_oh = torch.stack([hwgpt_one_hot(a, scale) for a in archs]).numpy().astype(np.float32)
    print(f"  one-hot dim = {X_oh.shape[1]}  (vs {flatten_for_baselines(archs, scale).shape[1]} raw-flat)")

    n = len(ppls)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=seed, shuffle=True)
    Xtr = torch.from_numpy(X_oh[train_idx]).float().to(DEVICE)
    Xte = torch.from_numpy(X_oh[test_idx]).float().to(DEVICE)
    ytr = torch.from_numpy(ppls[train_idx]).float().to(DEVICE)
    yte = ppls[test_idx]

    torch.manual_seed(seed)
    model = _PaperNet(nfeat=X_oh.shape[1], hidden=128).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    print(f"  fitting (epochs={epochs}, batch={batch_size}, lr={lr})...")
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
    print(f"  done in {time.time()-t0:.0f}s")
    model.eval()
    with torch.no_grad():
        preds = model(Xte).squeeze(-1).cpu().numpy()

    metrics = compute_metrics(yte, preds)
    print(f"  one-hot MLP: Spearman={metrics['spearman']:+.4f}  Kendall={metrics['kendall']:+.4f}  "
          f"MAE={metrics['mae']:.4f}  k@1%={metrics['k1']}  k@5%={metrics['k5']}  "
          f"MAE@5%={metrics['mae_top5']:.4f}")
    return metrics


def main():
    diagnose_custom()
    diagnose_hwgpt()
    onehot_metrics = fit_mlp_hwgpt_onehot()

    print("\n=== Summary: HW-GPT-Bench gpt_l, MLP comparison ===")
    print("  current table (raw flat features, 75-dim input):")
    print("    Spearman=+0.9244  Kendall=+0.7401  MAE=0.4686  k@1%=416  k@5%=584  MAE@5%=0.3661")
    print("  one-hot encoding (HW-GPT-Bench paper's native input, ~226-dim):")
    print(f"    Spearman={onehot_metrics['spearman']:+.4f}  Kendall={onehot_metrics['kendall']:+.4f}  "
          f"MAE={onehot_metrics['mae']:.4f}  k@1%={onehot_metrics['k1']}  k@5%={onehot_metrics['k5']}  "
          f"MAE@5%={onehot_metrics['mae_top5']:.4f}")
    print("  legacy LLMArch_Predictor (Paper-Net one-hot, 70/30 split):")
    print("    Spearman=+0.9993  Kendall=+0.9774  MAE=0.0508  k@1%=45  k@5%=271  top1%-MAE=0.0443")


if __name__ == "__main__":
    main()
