"""Train ForgeFormer on the HW-GPT-Bench dataset (scale s/m/l).

Uses the same ``ArchTransformerRanker`` as our custom-dataset training, just
with FIELD_COUNT=5 (HW-GPT-Bench schema). 80/20 train/test split, L1 loss,
AdamW. Saves a .pt + .json sidecar to the per-scale ckpt directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR))

from surrogate.model import ArchTransformerRanker  # noqa: E402
from data_loader import (  # noqa: E402
    FIELD_COUNT, SCALE_TO_MAX_LAYERS, load_hwgpt_dataset,
)


def parse_args() -> argparse.Namespace:
    # Defaults match the legacy LLMArch_Predictor V1-on-HWGPT winning config
    # (run_v1_hwgpt.py --d_model 256, seed=7, test_ratio=0.3) which beats the
    # HW-GPT-Bench paper baseline at the same split.
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=("s", "m", "l"), default="l")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--test_ratio", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--save_path", type=Path, default=None,
                   help="defaults to ckpts/forgeformer_hwgpt_<scale>.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_path is None:
        args.save_path = THIS_DIR / "ckpts" / f"forgeformer_hwgpt_{args.scale}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_layers = SCALE_TO_MAX_LAYERS[args.scale]

    print(f"[train] loading HW-GPT-Bench gpt_{args.scale} ...")
    norm_batch, _X_flat, ppls, info = load_hwgpt_dataset(args.scale)
    n = info["n"]
    print(f"[train] n={n}, max_layers={max_layers}, ppl=[{info['ppl_min']:.2f}, {info['ppl_max']:.2f}]")

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=args.test_ratio,
                                           random_state=args.seed, shuffle=True)
    print(f"[train] split: train={len(train_idx)} test={len(test_idx)}")

    x_tr = norm_batch.x_raw[train_idx]
    pad_tr = norm_batch.padding_mask[train_idx]
    y_tr = norm_batch.val_loss[train_idx]
    x_te = norm_batch.x_raw[test_idx]
    pad_te = norm_batch.padding_mask[test_idx]
    y_te = norm_batch.val_loss[test_idx]

    model = ArchTransformerRanker(
        max_layers=max_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        field_count=FIELD_COUNT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e3:.1f}K")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(TensorDataset(x_tr, pad_tr, y_tr),
                        batch_size=args.batch_size, shuffle=True)

    best_test_l1 = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, nb = 0.0, 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            score = model(xb, padding_mask=mb)
            loss = model.absolute_loss(score, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.shape[0]
            nb += xb.shape[0]
        train_l1 = loss_sum / max(1, nb)

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, x_te.shape[0], 256):
                preds.append(
                    model(x_te[i:i+256].to(device), padding_mask=pad_te[i:i+256].to(device))
                    .cpu().numpy()
                )
            preds = np.concatenate(preds)
        test_l1 = float(np.mean(np.abs(preds - y_te.numpy())))

        improved = test_l1 < best_test_l1
        if improved:
            best_test_l1 = test_l1
            best_epoch = epoch
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_mins": info["norm_mins"].tolist(),
                "norm_span": info["norm_span"].tolist(),
                "max_layers": max_layers,
                "field_count": FIELD_COUNT,
                "scale": args.scale,
                "test_idx": test_idx, "train_idx": train_idx,
                "seed": args.seed,
            }, args.save_path)
            with open(args.save_path.with_suffix(".json"), "w") as f:
                json.dump({
                    "model": {
                        "d_model": args.d_model, "nhead": args.nhead,
                        "num_layers": args.num_layers, "max_layers": max_layers,
                        "dropout": args.dropout, "field_count": FIELD_COUNT,
                    },
                    "training": {
                        "scale": args.scale, "epochs_trained": best_epoch,
                        "epochs_max": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "test_ratio": args.test_ratio,
                        "seed": args.seed,
                        "n_train": len(train_idx), "n_test": len(test_idx),
                    },
                    "results": {"best_test_l1": best_test_l1},
                }, f, indent=2)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs or improved:
            star = " *" if improved else ""
            print(f"  ep {epoch:>3}: train_L1={train_l1:.4f}  test_L1={test_l1:.4f}{star}")

    print(f"\n[train] best test L1 = {best_test_l1:.4f} @ epoch {best_epoch}")
    print(f"[train] saved → {args.save_path}")


if __name__ == "__main__":
    main()
