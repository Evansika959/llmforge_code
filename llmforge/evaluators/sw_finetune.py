"""Surrogate + active-learning SW evaluator.

Wraps SwSurrogate so the per-generation predictions stay fast, plus exposes
a `run_active_learning(...)` hook the dispatcher fires every N gens. The
event:
  1. picks a batch (half Pareto-best μ, half highest-σ),
  2. real-trains the batch on the cluster (synchronous — reuses
     SwRealTrain's submit/wait/fetch flow),
  3. appends the labels to a persistent buffer and fine-tunes the surrogate
     on the accumulated buffer,
  4. swaps in the fine-tuned (model, norm) without ever overwriting the
     baseline ckpt.

Versioned ckpts land at `{surrogate_save_dir}/gen{N}.pt` + sidecar JSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from nsga2 import fast_non_dominated_sort
from search_space import Individual

from .sw_real_train import SwRealTrain, archs_to_training_yaml  # noqa: F401
from .sw_surrogate import SwSurrogate

log = logging.getLogger("sw_finetune")


def _select_active_learning_batch(population, mu_pred: List[float],
                                  sigma_pred: List[float],
                                  batch_size: int) -> List[int]:
    """Half from current Pareto (μ-best), half from highest σ. Returns indices."""
    if batch_size <= 0:
        return []
    n = len(population.individuals)
    if n == 0:
        return []
    objs = [e.objs for e in population.evaluations]
    cons = [e.cons for e in population.evaluations]
    fronts = fast_non_dominated_sort(objs, cons)
    pareto = list(fronts[0]) if fronts else []
    pareto_sorted = sorted(pareto, key=lambda i: mu_pred[i])
    n_par = min(batch_size // 2, len(pareto_sorted))
    par_picks = pareto_sorted[:n_par]
    remaining = [i for i in range(n) if i not in set(par_picks)]
    remaining.sort(key=lambda i: -sigma_pred[i])
    n_div = min(batch_size - n_par, len(remaining))
    div_picks = remaining[:n_div]
    return list(par_picks) + list(div_picks)


class SwFinetune(SwSurrogate):
    """Surrogate driving NSGA + scheduled real-training + safe refit.

    Active-learning event timing (`should_fire(gen)`) is owned by the
    dispatcher; this class only knows how to *run* an event.
    """

    def __init__(self, *, ckpt_path: str, device,
                 mc_dropout_n: int,
                 real_train: SwRealTrain,
                 surrogate_save_dir: str,
                 finetune_every: int,
                 finetune_batch: int,
                 finetune_epochs: int = 10,
                 finetune_lr: float = 1e-4,
                 finetune_batch_size: int = 32,
                 base_dataset_csv: Optional[str] = None,
                 old_to_new_ratio: float = 0.0):
        super().__init__(ckpt_path=ckpt_path, device=device,
                         mc_dropout_n=mc_dropout_n)
        from surrogate.finetune import RealDataBuffer
        self.base_ckpt_path = os.path.abspath(ckpt_path)
        self.base_dataset_csv = base_dataset_csv
        self.old_to_new_ratio = float(old_to_new_ratio)
        # Load the base dataset once at startup (used as the "old data" pool
        # for experience-replay blending in run_active_learning).
        self._base_batch = None
        if self.base_dataset_csv and self.old_to_new_ratio > 0:
            from surrogate.finetune import load_raw_arch_dataset
            csv_abs = (self.base_dataset_csv
                       if os.path.isabs(self.base_dataset_csv)
                       else os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))), self.base_dataset_csv))
            self._base_batch = load_raw_arch_dataset(csv_abs, max_layers=40)
            n_old = int(self._base_batch.x_raw.shape[0])
            log.info(f"[finetune] base dataset loaded: {n_old} rows from {csv_abs} "
                     f"(blend old:new = {self.old_to_new_ratio:.1f}:1)")
        # Cache the base ckpt's sidecar `model` block so fine-tuned ckpts can
        # round-trip the same hyperparams without introspecting nn.Module
        # internals (which would couple us to a specific model class layout).
        import json as _json
        sidecar = os.path.splitext(self.base_ckpt_path)[0] + ".json"
        with open(sidecar, "r") as f:
            self._base_sidecar = _json.load(f)
        if "model" not in self._base_sidecar:
            raise ValueError(
                f"[sw_finetune] base sidecar {sidecar} is missing the 'model' "
                f"block; load_surrogate would fail on the fine-tuned ckpts. "
                f"Verify the surrogate ckpt's sidecar schema."
            )
        self.save_dir = os.path.abspath(surrogate_save_dir)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if os.path.dirname(self.base_ckpt_path) == self.save_dir:
            raise ValueError(
                f"surrogate_save_dir ({self.save_dir}) is the same dir as "
                f"the base ckpt ({self.base_ckpt_path}); fine-tuned gen{{N}}.pt "
                f"would risk colliding. Pick a different save_dir.")
        self.real_train = real_train
        self.finetune_every = int(finetune_every)
        self.finetune_batch = int(finetune_batch)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.finetune_batch_size = int(finetune_batch_size)
        self.label_buffer = RealDataBuffer()

    def should_fire(self, gen: int) -> bool:
        return self.finetune_every > 0 and gen > 0 and gen % self.finetune_every == 0

    def run_active_learning(self, population, mu_pred: List[float],
                            sigma_pred: List[float]) -> bool:
        """Run one event: pick → real-train → refit → swap surrogate.
        Returns True if the surrogate was successfully refit and swapped."""
        picks = _select_active_learning_batch(
            population, mu_pred, sigma_pred, self.finetune_batch)
        if not picks:
            log.info("[finetune] no candidates picked; skipping")
            return False
        log.info(f"[finetune] event @ gen {population.gen}: picked "
                 f"{len(picks)} archs (indices {picks})")
        archs = [population.individuals[i] for i in picks]

        # Tag the real-trainer with the current gen so its payload paths
        # match.
        self.real_train.set_gen(int(population.gen))
        labels, _ = self.real_train.evaluate(archs)

        # Drop NaN/inf labels — failed remote runs would poison the buffer.
        keep_archs, keep_labels = [], []
        for a, y in zip(archs, labels):
            if y is not None and y == y and y != float("inf"):
                keep_archs.append(a)
                keep_labels.append(float(y))
        if not keep_archs:
            log.warning("[finetune] no valid labels in this event; surrogate NOT refit.")
            return False
        self.label_buffer.add(keep_archs, keep_labels)

        # Always reload the frozen baseline before fine-tuning, so per-event
        # drift never compounds against the in-memory model.
        from surrogate.inference import load_surrogate
        from surrogate.finetune import finetune_surrogate
        from surrogate.data import norm_stats_to_dict

        model, norm, _max_layers = load_surrogate(self.base_ckpt_path, self.device)
        model, updated_norm = finetune_surrogate(
            model=model, buffer=self.label_buffer, norm_stats=norm,
            device=self.device, epochs=self.finetune_epochs,
            lr=self.finetune_lr, batch_size=self.finetune_batch_size,
            save_path=None,
            base_batch=self._base_batch,
            old_to_new_ratio=self.old_to_new_ratio,
        )

        gen = int(population.gen)
        out_pt = os.path.join(self.save_dir, f"gen{gen}.pt")
        out_json = os.path.join(self.save_dir, f"gen{gen}.json")
        if os.path.abspath(out_pt) == self.base_ckpt_path:
            raise RuntimeError(
                f"[finetune safety] derived save path {out_pt} would overwrite "
                f"base ckpt {self.base_ckpt_path}; refusing.")

        torch.save({
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats_to_dict(updated_norm),
            "max_layers": model.max_layers,
            "base_ckpt": self.base_ckpt_path,
            "gen": gen,
            "n_buffer": self.label_buffer.size,
        }, out_pt)

        import json as _json
        # Round-trip the baseline's `model` block — that's what load_surrogate
        # reads back. Keeping it identical guarantees the fine-tuned ckpt
        # loads even if the surrogate's class internals change later.
        sidecar = {
            "model": dict(self._base_sidecar.get("model", {})),
            "training": {"finetune_epochs": self.finetune_epochs,
                         "lr": self.finetune_lr,
                         "batch_size": self.finetune_batch_size,
                         "n_buffer": self.label_buffer.size},
            "base_ckpt": self.base_ckpt_path,
            "gen": gen,
        }
        sidecar["model"]["max_layers"] = int(getattr(model, "max_layers",
                                                       sidecar["model"].get("max_layers", 40)))
        with open(out_json, "w") as f:
            _json.dump(sidecar, f, indent=2)

        # Hot-swap the in-memory surrogate so subsequent generations use it.
        self.rebind(model, updated_norm)
        log.info(f"[finetune] saved {out_pt} (buffer size {self.label_buffer.size}); "
                 f"surrogate refreshed.")
        return True
