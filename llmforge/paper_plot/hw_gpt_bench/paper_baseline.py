"""Train the HW-GPT-Bench paper's own perplexity predictor on our seed=7 split.

Reproduces the paper's `hwgpt/predictors/metric/net.py:Net` architecture and
training recipe:
  input        = one-hot(embed_dim) ⊕ one-hot(n_layer) ⊕ one-hot(mlp_ratio×L) ⊕
                 one-hot(n_head×L) ⊕ one-hot(bias)          # 6 + 6·L + 2 dims
  network      = FC→ReLU → FC→ReLU → FC→ReLU → FC→ReLU → FC   (5 FC, hidden 128)
  loss         = MSE
  optimizer    = Adam, lr=1e-3, batch=1024, 4000 epochs

We then evaluate on the *same* 30% held-out test set that V1/RF/sklearn-MLP
were evaluated on so the numbers stack in a single fair comparison.

Writes:
  hwgpt_{scale}_paper_baseline.csv  (single-row metric table)
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("$HOME/hw-gpt-bench")
OURS = Path("$HOME/LLMArch_Predictor")
HERE = OURS / "final_paper_plots" / "predictor_verify" / "hw_gpt_bench"


# -- Inlined verbatim from hw-gpt-bench/lib/utils.py (to avoid pulling in ------
# -- the full hwgpt package which requires `lightning`, `syne-tune`, etc). -----
search_spaces = {
    "s": dict(embed_dim_choices=[192, 384, 768],
              n_layer_choices=[10, 11, 12],
              mlp_ratio_choices=[2, 3, 4],
              n_head_choices=[4, 8, 12],
              bias_choices=["True", "False"]),
    "m": dict(embed_dim_choices=[256, 512, 1024],
              n_layer_choices=[22, 23, 24],
              mlp_ratio_choices=[2, 3, 4],
              n_head_choices=[8, 12, 16],
              bias_choices=["True", "False"]),
    "l": dict(embed_dim_choices=[320, 640, 1280],
              n_layer_choices=[34, 35, 36],
              mlp_ratio_choices=[2, 3, 4],
              n_head_choices=[8, 16, 20],
              bias_choices=["True", "False"]),
}


def convert_str_to_arch(arch_str):
    parts = arch_str.split("-")
    n_layer = int(parts[2])
    embed_dim = int(parts[3])
    mlp_ratios = [int(x) for x in parts[4:4 + n_layer]]
    n_heads = [int(x) for x in parts[4 + n_layer:4 + 2 * n_layer]]
    bias = parts[-1]
    return dict(sample_n_layer=n_layer, sample_embed_dim=embed_dim,
                sample_mlp_ratio=mlp_ratios, sample_n_head=n_heads,
                sample_bias=bias)


def convert_config_to_one_hot(cfg, scale):
    cd = search_spaces[scale]
    max_layers = max(cd["n_layer_choices"])
    e = torch.zeros(len(cd["embed_dim_choices"]))
    l = torch.zeros(len(cd["n_layer_choices"]))
    m = torch.zeros(max_layers, len(cd["mlp_ratio_choices"]))
    h = torch.zeros(max_layers, len(cd["n_head_choices"]))
    b = torch.zeros(len(cd["bias_choices"]))
    e[cd["embed_dim_choices"].index(cfg["sample_embed_dim"])] = 1
    l[cd["n_layer_choices"].index(cfg["sample_n_layer"])] = 1
    for i in range(cfg["sample_n_layer"]):
        m[i][cd["mlp_ratio_choices"].index(cfg["sample_mlp_ratio"][i])] = 1
        h[i][cd["n_head_choices"].index(cfg["sample_n_head"][i])] = 1
    b[cd["bias_choices"].index(cfg["sample_bias"])] = 1
    return torch.cat([e, l, m.view(-1), h.view(-1), b])


# -- Inlined verbatim from hw-gpt-bench/hwgpt/predictors/metric/net.py ---------
class Net(nn.Module):
    def __init__(self, num_layers: int, layer_size: int):
        super().__init__()
        nfeat = 6 + 6 * num_layers + 2
        self.fc1 = nn.Linear(nfeat, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.fc4 = nn.Linear(layer_size, layer_size)
        self.fc5 = nn.Linear(layer_size, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

SCALE = sys.argv[1] if len(sys.argv) > 1 else "s"
SEED = 7
TEST_RATIO = 0.3
EPOCHS = 4000
BATCH = 1024
LR = 1e-3
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def k_recover_pct(y_true, y_pred, pct):
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


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    print(f"[Paper baseline] scale={SCALE}  device={DEVICE}  epochs={EPOCHS}")

    # ---- Load raw arch → ppl pairs ----
    with open(REPO / f"data_collection/gpt_datasets/gpt_{SCALE}/stats.pkl", "rb") as f:
        data = pickle.load(f)
    arch_strs, ppls = [], []
    for s, m in data.items():
        p = m.get("perplexity")
        if p is None or not np.isfinite(float(p)):
            continue
        arch_strs.append(s); ppls.append(float(p))
    ppls = np.asarray(ppls, dtype=np.float32)
    n = len(arch_strs)
    print(f"Loaded {n} architectures. ppl min={ppls.min():.2f} max={ppls.max():.2f}")

    # ---- Encode via paper's own one-hot converter ----
    X = torch.stack([
        convert_config_to_one_hot(convert_str_to_arch(s), SCALE) for s in arch_strs
    ]).float()
    print(f"One-hot feature dim = {X.shape[1]}")

    # ---- Reproduce our seed=7 split ----
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=TEST_RATIO,
                                           random_state=SEED, shuffle=True)
    y = torch.from_numpy(ppls).float()
    Xtr, ytr = X[train_idx].to(DEVICE), y[train_idx].to(DEVICE)
    Xte, yte = X[test_idx].to(DEVICE), y[test_idx].to(DEVICE)
    print(f"split: n_train={len(train_idx)}  n_test={len(test_idx)}")

    # ---- Train their Net ----
    torch.manual_seed(SEED)
    max_layers = max(search_spaces[SCALE]["n_layer_choices"])
    model = Net(num_layers=max_layers, layer_size=HIDDEN).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
        if (ep + 1) % 200 == 0 or ep == 0:
            print(f"  epoch {ep+1:>4}/{EPOCHS}  train MSE = {tot/len(ytr):.4f}  "
                  f"({time.time()-t0:.0f}s)")

    # ---- Save checkpoint ----
    ckpt_path = HERE / f"hwgpt_{SCALE}_paper_net.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(num_layers=max_layers, layer_size=HIDDEN, scale=SCALE),
        "train_idx": train_idx, "test_idx": test_idx, "seed": SEED,
    }, ckpt_path)
    print(f"Saved Paper-Net checkpoint → {ckpt_path}")

    # ---- Evaluate on our held-out set ----
    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).squeeze(-1).cpu().numpy()
    y_true = yte.cpu().numpy()

    rho, _ = spearmanr(pred_te, y_true)
    tau, _ = kendalltau(pred_te, y_true)
    mae = float(np.mean(np.abs(pred_te - y_true)))
    k1 = k_recover_pct(y_true, pred_te, 1.0)
    k2 = k_recover_pct(y_true, pred_te, 2.0)
    k5 = k_recover_pct(y_true, pred_te, 5.0)
    m1 = max(1, int(np.ceil(0.01 * len(y_true))))
    top_idx = np.argsort(y_true)[:m1]
    top_mae = float(np.mean(np.abs(pred_te[top_idx] - y_true[top_idx])))

    print(f"\n=== Paper baseline on gpt_{SCALE}  "
          f"(n_train={len(train_idx)}, n_test={len(test_idx)}) ===")
    print(f"  Paper-Net (MLP)        ρ={rho:.4f}  τ={tau:.4f}  MAE={mae:.4f}  "
          f"k@1%={k1 or '—':>4}  k@2%={k2 or '—':>4}  k@5%={k5 or '—':>5}  "
          f"top1%-MAE={top_mae:.4f}")

    pd.DataFrame([dict(method="Paper-Net (MLP)",
                       spearman=float(rho), kendall=float(tau), mae=mae,
                       k1pct=k1, k2pct=k2, k5pct=k5, top1pct_mae=top_mae)]).to_csv(
        HERE / f"hwgpt_{SCALE}_paper_baseline.csv", index=False)
    np.savez(HERE / f"hwgpt_{SCALE}_paper_preds.npz",
             y_true=y_true, paper=pred_te)
    print(f"\nWrote {HERE}/hwgpt_{SCALE}_paper_baseline.csv")
    print(f"Wrote {HERE}/hwgpt_{SCALE}_paper_preds.npz")


if __name__ == "__main__":
    main()
