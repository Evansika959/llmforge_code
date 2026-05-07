"""Surrogate val_loss evaluator with MC-dropout uncertainty.

NSGA receives `μ - β·σ` (UCB acquisition) — μ_pred is preserved separately
in aux for downstream analysis. The MC-dropout path bypasses
surrogate_eval (which calls model.eval() internally and resets dropout)
and runs forwards directly with dropout left enabled.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from search_space import Individual
from surrogate.inference import (
    _build_tensor_from_individuals_df,
    _individuals_to_df,
    load_surrogate,
    normalize_x,
    surrogate_eval,
)


def _enable_dropout_at_inference(model: torch.nn.Module) -> None:
    """Force every Dropout layer ON, leave the rest in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d,
                          torch.nn.AlphaDropout)):
            m.train()


class SwSurrogate:
    """Surrogate-only SW evaluator.

    `mc_dropout_n=1` returns deterministic predictions (sigma=0).
    `acquisition_beta` is applied at the dispatcher level via μ-β·σ; this
    evaluator returns the (μ, σ) pair without modification.
    """

    def __init__(self, ckpt_path: str, device, mc_dropout_n: int = 10):
        self.device = device
        self.mc_dropout_n = int(mc_dropout_n)
        self.model, self.norm, self.max_layers = load_surrogate(ckpt_path, device)

    def rebind(self, model, norm) -> None:
        """Swap the in-memory surrogate (used by active learning to install
        a fine-tuned ckpt without rebuilding the evaluator)."""
        self.model = model
        self.norm = norm

    def evaluate(self, inds: List[Individual]) -> Tuple[List[float], List[float]]:
        if not inds:
            return [], []

        if self.mc_dropout_n <= 1:
            mu = surrogate_eval(inds, self.model, self.norm, self.device, self.max_layers)
            return list(mu), [0.0] * len(mu)

        df = _individuals_to_df(inds, max_layers=self.max_layers)
        x_raw, padding_mask = _build_tensor_from_individuals_df(df, self.max_layers)
        x_norm = normalize_x(x_raw, self.norm).to(self.device)
        padding_mask = padding_mask.to(self.device)

        _enable_dropout_at_inference(self.model)
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_n):
                score = self.model(x_norm, padding_mask=padding_mask)
                samples.append(score.detach().cpu().numpy().astype(np.float64))
        arr = np.stack(samples, axis=0)
        mu = arr.mean(axis=0).tolist()
        sig = arr.std(axis=0, ddof=0).tolist()
        self.model.eval()
        return mu, sig
