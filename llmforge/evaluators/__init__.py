"""Evaluator interface for the unified NSGA co-search.

Two roles:
  SwEvaluator: produces (mu, sigma) val_loss predictions per individual
  HwEvaluator: produces a list of HW-aux dicts per individual

The unified runner always composes one chosen HW evaluator with `hw_none`,
so analytical aux fields (params_M, kv_cache_bytes, flops_per_token) are
present in every run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from search_space import Individual


class SwEvaluator(Protocol):
    def evaluate(self, inds: List[Individual]) -> Tuple[List[float], List[float]]:
        """Return (mu, sigma) val_loss predictions aligned to `inds`.

        Deterministic backends should return sigma = [0.0]*len(inds).
        """
        ...


class HwEvaluator(Protocol):
    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return aligned aux-dicts with substrate-specific HW metrics.

        For HW backends with no measurement (substrate=none), returns the
        analytical aux fields (params_M, kv_cache_bytes, flops_per_token).
        For substrate=rdxe, each dict carries an extra `chip_pareto` key with
        the inner-loop Pareto front; the overall-best chip's metrics are
        also promoted to top-level keys for outer-NSGA selection.
        """
        ...
