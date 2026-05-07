"""Online finetuning and accuracy tracking for the surrogate predictor.

Provides:
- RealDataBuffer: accumulates (architecture, real_val_loss) pairs across generations
- compute_accuracy_metrics: compares predicted vs real val_loss
- select_for_real_eval: picks a subset of individuals for real training
- finetune_surrogate: fine-tunes the surrogate on accumulated real data
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    ArchBatchRaw,
    ArchDataset,
    NormStats,
    FieldStats,
    ALL_FIELDS,
    compute_norm_stats,
    normalize_batch,
    norm_stats_to_dict,
    load_raw_arch_dataset,
)
from .model import ArchTransformerRanker
from .inference import (
    _individuals_to_df,
    _build_tensor_from_individuals_df,
)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search_space import Individual


class RealDataBuffer:
    """Accumulates (architecture_tensor, real_val_loss) pairs across generations."""

    def __init__(self):
        self._x_raw_parts: List[torch.FloatTensor] = []
        self._mask_parts: List[torch.BoolTensor] = []
        self._loss_parts: List[torch.FloatTensor] = []

    def add(
        self,
        individuals: List[Individual],
        real_losses: List[float],
        max_layers: int = 40,
    ) -> None:
        """Add a batch of (individual, real_val_loss) pairs to the buffer."""
        if not individuals or not real_losses:
            return
        df = _individuals_to_df(individuals, max_layers=max_layers)
        x_raw, padding_mask = _build_tensor_from_individuals_df(df, max_layers)
        losses = torch.tensor(real_losses, dtype=torch.float32)
        self._x_raw_parts.append(x_raw)
        self._mask_parts.append(padding_mask)
        self._loss_parts.append(losses)

    def to_raw_batch(self) -> ArchBatchRaw:
        """Concatenate all accumulated data into a single ArchBatchRaw."""
        if not self._x_raw_parts:
            raise ValueError("Buffer is empty")
        return ArchBatchRaw(
            x_raw=torch.cat(self._x_raw_parts, dim=0),
            padding_mask=torch.cat(self._mask_parts, dim=0),
            val_loss=torch.cat(self._loss_parts, dim=0),
        )

    @property
    def size(self) -> int:
        return sum(p.shape[0] for p in self._x_raw_parts)


def compute_accuracy_metrics(
    pred_losses: List[float],
    real_losses: List[float],
) -> Dict[str, float]:
    """Compare predicted vs real validation losses.

    Returns dict with: l1, spearman_r, pairwise_acc, per_error (list).
    """
    pred = np.array(pred_losses, dtype=np.float64)
    real = np.array(real_losses, dtype=np.float64)
    assert len(pred) == len(real), "pred and real must have same length"

    errors = np.abs(pred - real)
    l1 = float(errors.mean())

    # Spearman rank correlation
    spearman_r = 0.0
    if len(pred) > 1:
        from scipy import stats as scipy_stats
        sp, _ = scipy_stats.spearmanr(pred, real)
        spearman_r = float(sp) if np.isfinite(sp) else 0.0

    # Pairwise ranking accuracy
    n = len(pred)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(real[i] - real[j]) < 1e-6:
                continue
            total += 1
            if (pred[i] < pred[j]) == (real[i] < real[j]):
                correct += 1
    pairwise_acc = correct / max(total, 1)

    return {
        "l1": l1,
        "spearman_r": spearman_r,
        "pairwise_acc": pairwise_acc,
        "per_error": errors.tolist(),
    }


def select_for_real_eval(
    offspring_evaluations: list,
    pred_list: List[float],
    K: int,
    strategy: str = "full_population",
    population=None,
) -> List[int]:
    """Pick offspring/individual indices for real training verification.

    Args:
        offspring_evaluations: list of EvaluationResult for the offspring
        pred_list: surrogate-predicted val_loss for each offspring
        K: max number of individuals to select (ignored for "full_population" and "pareto_front")
        strategy: "full_population", "pareto_front", "pareto_and_random", "random", or "top_k"
        population: Population object (required for "full_population" and "pareto_front")

    Returns:
        List of indices to real-train.
        For "full_population"/"pareto_front": indices into population.individuals.
        For others: indices into offspring.
    """
    n = len(pred_list)
    K = min(K, n)

    if strategy == "full_population":
        # Real-train the entire current population
        if population is None or not population.individuals:
            raise ValueError("full_population strategy requires a population with individuals")
        return list(range(len(population.individuals)))

    if strategy == "pareto_front":
        # Select the Pareto front (F0) from the current population
        if population is None or not population.evaluations:
            raise ValueError("pareto_front strategy requires a population with evaluations")
        objs = [e.objs for e in population.evaluations]
        cons = [e.cons for e in population.evaluations]
        from nsga2 import fast_non_dominated_sort
        fronts = fast_non_dominated_sort(objs, cons)
        f0 = fronts[0] if fronts else []
        return f0

    if K <= 0:
        return []

    if strategy == "random":
        return random.sample(range(n), K)

    elif strategy == "top_k":
        sorted_indices = sorted(range(n), key=lambda i: pred_list[i])
        return sorted_indices[:K]

    elif strategy == "pareto_and_random":
        if offspring_evaluations:
            ranked = sorted(range(n), key=lambda i: offspring_evaluations[i].objs[0]
                            if i < len(offspring_evaluations) else float("inf"))
        else:
            ranked = sorted(range(n), key=lambda i: pred_list[i])

        n_pareto = min(K // 2, n)
        pareto_picks = ranked[:n_pareto]
        remaining = [i for i in range(n) if i not in set(pareto_picks)]
        n_random = min(K - n_pareto, len(remaining))
        random_picks = random.sample(remaining, n_random) if remaining and n_random > 0 else []
        return pareto_picks + random_picks

    else:
        raise ValueError(f"Unknown real_eval_strategy: {strategy}")


def finetune_surrogate(
    model: ArchTransformerRanker,
    buffer: RealDataBuffer,
    norm_stats: NormStats,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 32,
    save_path: Optional[str] = None,
    base_batch: Optional[ArchBatchRaw] = None,
    old_to_new_ratio: float = 0.0,
) -> Tuple[ArchTransformerRanker, NormStats]:
    """Fine-tune the surrogate on accumulated real training data.

    Args:
        buffer: per-event accumulated (arch, val_loss) labels.
        base_batch: optional original training corpus (loaded once at
            startup). When supplied with ``old_to_new_ratio > 0`` the
            sampler draws old:new minibatches at the configured ratio
            (experience replay) so the model doesn't forget the broader
            search space while bending toward the recent labels.
        old_to_new_ratio: e.g. 5.0 means 5 old rows per 1 new row in
            expectation per minibatch. 0.0 disables the blend (legacy
            new-only behaviour).

    Returns:
        model: updated model (modified in-place)
        updated_norm_stats: new normalisation statistics (unioned with
            the supplied ``norm_stats``)
    """
    new_batch = buffer.to_raw_batch()
    n_new = new_batch.x_raw.shape[0]

    # ── Build training set: optional old + new blend ────────────────────
    use_blend = base_batch is not None and old_to_new_ratio > 0 and n_new > 0
    if use_blend:
        n_old = base_batch.x_raw.shape[0]
        # Concatenate; weights below pick old:new in the requested ratio.
        full_batch = ArchBatchRaw(
            x_raw=torch.cat([base_batch.x_raw, new_batch.x_raw], dim=0),
            padding_mask=torch.cat([base_batch.padding_mask, new_batch.padding_mask], dim=0),
            val_loss=torch.cat([base_batch.val_loss, new_batch.val_loss], dim=0),
        )
        # Norm stats: compute over the *full* training set so the union
        # covers both old and new data ranges.
        updated_norm_stats = compute_norm_stats(full_batch)
        n_total = n_old + n_new
        print(f"Finetuning surrogate (blend mode): {n_old} old + {n_new} new "
              f"sampled at old:new = {old_to_new_ratio:.1f}:1, "
              f"{epochs} epochs (lr={lr})")
    else:
        full_batch = new_batch
        n_total = n_new
        updated_norm_stats = compute_norm_stats(new_batch)
        print(f"Finetuning surrogate on {n_new} real data points for "
              f"{epochs} epochs (lr={lr})")

    # Merge with the supplied (baseline) norm_stats: expand ranges to cover both
    merged_stats: Dict[str, FieldStats] = {}
    for field in ALL_FIELDS:
        orig = norm_stats.stats.get(field, FieldStats(0.0, 1.0))
        new = updated_norm_stats.stats.get(field, FieldStats(0.0, 1.0))
        merged_stats[field] = FieldStats(
            vmin=min(orig.vmin, new.vmin),
            vmax=max(orig.vmax, new.vmax),
        )
    updated_norm_stats = NormStats(stats=merged_stats)

    normalized_batch = normalize_batch(full_batch, updated_norm_stats)
    dataset = ArchDataset(normalized_batch)

    if use_blend:
        # WeightedRandomSampler with class probability ratio old:new = R:1.
        # Per-sample weights so each *class* (old or new) gets a fixed
        # share of mass regardless of dataset sizes.
        from torch.utils.data import WeightedRandomSampler
        weights = torch.empty(n_total, dtype=torch.float64)
        weights[:n_old] = old_to_new_ratio / n_old      # old class total mass = R
        weights[n_old:] = 1.0 / n_new                   # new class total mass = 1
        # Per epoch, see ~6 mini-batches per new row on average so each new
        # label gets revisited a few times even in early events.
        per_epoch = max(batch_size * 16, n_new * 6)
        sampler = WeightedRandomSampler(weights, num_samples=per_epoch, replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        loss_accum = 0.0
        n_batches = 0
        for x, padding_mask, target in loader:
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            target = target.to(device)

            score = model(x, padding_mask=padding_mask)
            loss = model.absolute_loss(score, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            n_batches += 1

        avg_loss = loss_accum / max(1, n_batches)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  Finetune epoch {epoch}/{epochs}: L1 = {avg_loss:.4f}")

    model.eval()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats_to_dict(updated_norm_stats),
                "max_layers": model.max_layers,
            },
            save_path,
        )
        print(f"  Saved finetuned checkpoint to {save_path}")

    return model, updated_norm_stats
