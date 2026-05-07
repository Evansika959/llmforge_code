"""Evaluate a trained ForgeFormer checkpoint on a CSV dataset.

Also exposes ``compute_metrics`` and ``evaluate_model`` as a shared library
used by ``surrogate.train`` for end-of-training reporting.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ArchDataset,
    load_raw_arch_dataset,
    normalize_batch,
)
from .inference import load_surrogate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: torch.nn.Module,
    dataset: ArchDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, padding_mask, target in loader:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            out = model(x, padding_mask=padding_mask)
            preds.append(out.cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    err = preds - targets
    l1 = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    spearman_r, spearman_p = scipy_stats.spearmanr(preds, targets)
    pearson_r, pearson_p = scipy_stats.pearsonr(preds, targets)

    n = len(preds)
    n_pairs = min(n * (n - 1) // 2, 50000)
    rng = np.random.default_rng(42)
    correct = ties = 0
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        if abs(targets[i] - targets[j]) < 1e-6:
            ties += 1
            continue
        if (preds[i] < preds[j]) == (targets[i] < targets[j]):
            correct += 1
    pairwise_acc = correct / max(n_pairs - ties, 1)

    return {
        "l1": l1,
        "rmse": rmse,
        "bias": bias,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pairwise_acc": float(pairwise_acc),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
    }


def print_metrics_report(metrics: Dict[str, float], errors: np.ndarray) -> None:
    print(f"  L1 (MAE):              {metrics['l1']:.4f}")
    print(f"  RMSE:                  {metrics['rmse']:.4f}")
    print(f"  Bias:                  {metrics['bias']:+.4f}")
    print(f"  Pearson r:             {metrics['pearson_r']:+.4f}  (p={metrics['pearson_p']:.2e})")
    print(f"  Spearman ρ:            {metrics['spearman_r']:+.4f}  (p={metrics['spearman_p']:.2e})")
    print(f"  Pairwise rank acc:     {metrics['pairwise_acc']:.2%}")
    print(f"  Prediction range:      [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")
    print(f"  Target range:          [{metrics['target_min']:.4f}, {metrics['target_max']:.4f}]")
    print("  Error percentiles:")
    for pct in (50, 75, 90, 95, 99):
        print(f"    P{pct:02d}: {np.percentile(errors, pct):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained ForgeFormer checkpoint on a CSV.")
    p.add_argument("--csv_path", type=str, required=True, help="Dataset CSV to evaluate against")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint (sidecar .json must exist)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, norm_stats, max_layers = load_surrogate(str(args.checkpoint), device)
    raw = load_raw_arch_dataset(args.csv_path, max_layers=max_layers)
    if len(raw.x_raw) == 0:
        raise ValueError(f"No valid samples found in {args.csv_path}")

    batch = normalize_batch(raw, norm_stats)
    dataset = ArchDataset(batch)
    print(f"Evaluating {len(dataset)} samples on {device} ...")

    preds, targets = evaluate_model(model, dataset, device, args.batch_size)
    metrics = compute_metrics(preds, targets)
    errors = np.abs(preds - targets)

    print()
    print("=" * 60)
    print(f"EVALUATION  ({args.checkpoint.name} on {Path(args.csv_path).name})")
    print("=" * 60)
    print_metrics_report(metrics, errors)


if __name__ == "__main__":
    main()
