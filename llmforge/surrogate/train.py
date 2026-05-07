"""Train ForgeFormer (the architecture-perplexity surrogate) from scratch.

Loads one or more architecture-perplexity CSVs, splits 80/20 train/test,
trains ``ArchTransformerRanker`` with L1 regression, and writes a
checkpoint pair (``.pt`` + ``.json`` sidecar) compatible with
``surrogate.inference.load_surrogate``.

Example:
    python -m surrogate.train \\
        --csv_paths surrogate/dataset/dataset_200M.csv \\
        --save_path surrogate/ckpts/forgeformer_200M.pt \\
        --epochs 200 --d_model 64 --nhead 4 --num_layers 4 --dropout 0.25
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ArchBatchRaw,
    ArchDataset,
    compute_norm_stats,
    load_raw_arch_dataset,
    norm_stats_to_dict,
    normalize_batch,
)
from .evaluate import compute_metrics, evaluate_model, print_metrics_report
from .model import ArchTransformerRanker


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subset(batch: ArchBatchRaw, indices: np.ndarray) -> ArchBatchRaw:
    return ArchBatchRaw(
        x_raw=batch.x_raw[indices],
        padding_mask=batch.padding_mask[indices],
        val_loss=batch.val_loss[indices],
    )


def _merge(batches: List[ArchBatchRaw]) -> ArchBatchRaw:
    if len(batches) == 1:
        return batches[0]
    return ArchBatchRaw(
        x_raw=torch.cat([b.x_raw for b in batches], dim=0),
        padding_mask=torch.cat([b.padding_mask for b in batches], dim=0),
        val_loss=torch.cat([b.val_loss for b in batches], dim=0),
    )


def _save_checkpoint(
    save_path: Path,
    model: ArchTransformerRanker,
    norm_stats,
    args: argparse.Namespace,
    n_train: int,
    n_test: int,
    best_test_l1: float,
    best_epoch: int,
) -> None:
    """Write .pt + .json sidecar in the layout expected by load_surrogate()."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats_to_dict(norm_stats),
            "max_layers": args.max_layers,
            "val_loss": best_test_l1,
        },
        save_path,
    )
    sidecar = save_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump(
            {
                "model": {
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "max_layers": args.max_layers,
                    "dropout": args.dropout,
                },
                "training": {
                    "csv_paths": [str(p) for p in args.csv_paths],
                    "epochs_trained": best_epoch,
                    "epochs_max": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "test_ratio": args.test_ratio,
                    "seed": args.seed,
                    "n_train": n_train,
                    "n_test": n_test,
                },
                "results": {"best_test_l1": best_test_l1},
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ForgeFormer surrogate from scratch.")
    p.add_argument("--csv_paths", type=Path, nargs="+", required=True,
                   help="One or more arch-perplexity CSVs (will be padded to --max_layers and merged)")
    p.add_argument("--max_layers", type=int, default=40)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2,
                   help="Fraction held out as test split (paper: 80/20).")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop if test L1 hasn't improved for N epochs (0 = disabled).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=Path, required=True,
                   help="Path for best checkpoint .pt; sidecar .json written next to it.")
    p.add_argument("--plot_path", type=Path, default=None)
    p.add_argument("--log_path", type=Path, default=None,
                   help="Optional path for per-epoch metrics JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    raw_batches: List[ArchBatchRaw] = []
    for csv_path in args.csv_paths:
        print(f"loading {csv_path} ...")
        batch = load_raw_arch_dataset(str(csv_path), max_layers=args.max_layers)
        n = len(batch.x_raw)
        active_first = (~batch.padding_mask[0]).sum().item() if n else 0
        print(f"  -> {n} samples (row-0 active layers: {active_first})")
        raw_batches.append(batch)

    merged = _merge(raw_batches)
    total = len(merged.x_raw)
    if total == 0:
        raise ValueError("No valid samples loaded (check CSVs / val_loss column).")

    idx = np.arange(total)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_ratio, random_state=args.seed, shuffle=True,
    )
    train_raw = _subset(merged, train_idx)
    test_raw = _subset(merged, test_idx)
    print(f"total={total}  train={len(train_idx)}  test={len(test_idx)}")

    norm_stats = compute_norm_stats(train_raw)
    train_batch = normalize_batch(train_raw, norm_stats)
    test_batch = normalize_batch(test_raw, norm_stats)
    print(
        f"train targets: min={train_batch.val_loss.min():.4f} mean={train_batch.val_loss.mean():.4f} "
        f"max={train_batch.val_loss.max():.4f}"
    )
    print(
        f"test  targets: min={test_batch.val_loss.min():.4f} mean={test_batch.val_loss.mean():.4f} "
        f"max={test_batch.val_loss.max():.4f}"
    )

    model = ArchTransformerRanker(
        max_layers=args.max_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = ArchDataset(train_batch)
    test_ds = ArchDataset(test_batch)

    best_test_l1 = float("inf")
    best_epoch = 0
    patience_left = args.early_stop_patience if args.early_stop_patience > 0 else None
    train_losses: List[float] = []
    test_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        loss_sum, n_batch = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch:3d}/{args.epochs}", leave=False)
        for x, padding_mask, target in pbar:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            target = target.to(device)

            score = model(x, padding_mask=padding_mask)
            loss = model.absolute_loss(score, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_l1 = loss_sum / max(1, n_batch)
        train_losses.append(train_l1)

        preds, targets = evaluate_model(model, test_ds, device, args.batch_size)
        test_l1 = float(np.mean(np.abs(preds - targets)))
        test_losses.append(test_l1)

        improved = test_l1 < best_test_l1
        if improved:
            best_test_l1 = test_l1
            best_epoch = epoch
            _save_checkpoint(
                args.save_path, model, norm_stats, args,
                n_train=len(train_idx), n_test=len(test_idx),
                best_test_l1=best_test_l1, best_epoch=best_epoch,
            )
            patience_left = args.early_stop_patience if args.early_stop_patience > 0 else None
        elif patience_left is not None:
            patience_left -= 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs or improved:
            star = " *" if improved else ""
            print(f"  ep {epoch:>3}: train_L1={train_l1:.4f}  test_L1={test_l1:.4f}{star}")

        if patience_left is not None and patience_left <= 0:
            print(f"  early stop at epoch {epoch} (best @ {best_epoch}, test L1={best_test_l1:.4f})")
            break

    # ── Save per-epoch log ───────────────────────────────────────────────
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_path, "w") as f:
            json.dump(
                {
                    "csv_paths": [str(p) for p in args.csv_paths],
                    "args": vars(args),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "best_test_l1": best_test_l1,
                    "best_epoch": best_epoch,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"per-epoch log -> {args.log_path}")

    # ── Final report on best checkpoint ──────────────────────────────────
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    preds, targets = evaluate_model(model, test_ds, device, args.batch_size)
    metrics = compute_metrics(preds, targets)
    errors = np.abs(preds - targets)

    print()
    print("=" * 60)
    print(f"FINAL TEST METRICS (best ckpt @ epoch {best_epoch})")
    print("=" * 60)
    print_metrics_report(metrics, errors)

    # ── Plot ─────────────────────────────────────────────────────────────
    if args.plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
            epochs_arr = np.arange(1, len(train_losses) + 1)

            ax = axes[0]
            ax.plot(epochs_arr, train_losses, label="train L1", alpha=0.8)
            ax.plot(epochs_arr, test_losses, label="test L1", alpha=0.8)
            ax.set_xlabel("epoch")
            ax.set_ylabel("L1")
            ax.set_yscale("log")
            ax.set_title("training curves")
            ax.legend()

            ax = axes[1]
            ax.scatter(targets, preds, alpha=0.4, s=10)
            lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
            ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
            ax.set_xlabel("actual val_loss")
            ax.set_ylabel("predicted val_loss")
            ax.set_title(f"pred vs actual (Pearson r={metrics['pearson_r']:.3f})")
            ax.legend()

            ax = axes[2]
            ax.hist(errors, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(metrics["l1"], color="r", linestyle="--", label=f"MAE={metrics['l1']:.4f}")
            ax.set_xlabel("|error|")
            ax.set_ylabel("count")
            ax.set_title("error distribution")
            ax.legend()

            plt.tight_layout()
            args.plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot_path)
            print(f"plot -> {args.plot_path}")
        except Exception as e:
            print(f"failed to create plot: {e}")

    print(f"\nbest checkpoint -> {args.save_path}")
    print(f"sidecar config  -> {args.save_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
