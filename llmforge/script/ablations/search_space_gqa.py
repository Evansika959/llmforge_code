"""Strict-GQA variant of HeteroSearchSpace.

Enforces classic GQA shape constraints on every layer in addition to the
parent class's IHA-style repair:
    1. n_kv_group divides n_head             (already enforced upstream)
    2. n_v_head_dim == n_qk_head_dim         (Q/K--V coupling)
    3. n_head * n_qk_head_dim == n_embd      (MHA divisibility)

Together with (1), these reproduce the constraint set inherited by GQA from
multi-head attention. The IHA search space is otherwise unchanged: per-layer
fields are still searched, but each repaired layer is snapped to the nearest
combination satisfying the three constraints. Any candidate produced by
sample / crossover / mutation passes through repair() before evaluation, so
no GQA-violating architecture is ever scored.

Used by Forge-DSE's "NSGA + GQA" ablation (see
script/ablations/nsga_gqa_paramsloss.bash). When --strict_gqa is NOT passed
to run_cosearch.py, the regular HeteroSearchSpace is used and the IHA full
search space applies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from search_space import HeteroSearchSpace, Individual


def _enumerate_valid_gqa_combinations(
    n_embd: int,
    h_low: int, h_high: int,
    qk_low: int, qk_high: int, qk_step: int,
) -> List[Tuple[int, int]]:
    """Return all (n_head, n_qk_head_dim) pairs satisfying:
       n_head * n_qk_head_dim == n_embd, with both fields in their search-space
       ranges and n_qk_head_dim step-aligned starting at qk_low.
    """
    valid: List[Tuple[int, int]] = []
    for h in range(h_low, h_high + 1):
        if n_embd % h != 0:
            continue
        d_qk = n_embd // h
        if d_qk < qk_low or d_qk > qk_high:
            continue
        if (d_qk - qk_low) % qk_step != 0:
            continue
        valid.append((h, d_qk))
    return valid


class StrictGQASearchSpace(HeteroSearchSpace):
    """HeteroSearchSpace variant that snaps every layer to a valid GQA shape
    after the parent's standard repair.

    Inherits all sampling / crossover / mutation logic. Only `repair()` is
    extended: after super().repair() runs, each layer is snapped to the
    closest valid (n_head, n_qk_head_dim) pair (Manhattan distance in the
    normalized search-range) and n_v_head_dim is forced equal to the snapped
    n_qk_head_dim. n_kv_group is then re-snapped to a divisor of the new
    n_head.

    If no valid (n_head, n_qk_head_dim) combination exists for the supplied
    n_embd / search-range setting, repair raises a ValueError to surface
    the misconfiguration loudly.
    """

    @classmethod
    def from_dicts(
        cls,
        globals_spec: Dict[str, Any],
        layer_spec: Dict[str, Any],
        L_max: int = 24,
        L_min: int = 1,
        no_repair: bool = False,
        freeze_layer_mask: bool = False,
    ) -> "StrictGQASearchSpace":
        inst: "StrictGQASearchSpace" = super().from_dicts(  # type: ignore[assignment]
            globals_spec, layer_spec,
            L_max=L_max, L_min=L_min,
            no_repair=no_repair,
            freeze_layer_mask=freeze_layer_mask,
        )
        inst._gqa_validated = False  # validate lazily on first repair
        return inst

    # ---------------- repair ----------------

    def _gqa_constraints_summary(self) -> List[Tuple[int, int]]:
        """Cache the valid (n_head, d_qk) combinations once n_embd is known."""
        if getattr(self, "_gqa_cache", None) is not None:
            return self._gqa_cache
        # Pull bounds from layer_spec (defaults match search_space_200M.yaml)
        h_spec = self.layer_spec.get("n_head", {})
        qk_spec = self.layer_spec.get("n_qk_head_dim", {})
        h_low = int(h_spec.get("low", 1))
        h_high = int(h_spec.get("high", 16))
        qk_low = int(qk_spec.get("low", 64))
        qk_high = int(qk_spec.get("high", 512))
        qk_step = int(qk_spec.get("step", 32))
        # n_embd: prefer value from globals spec (fixed range typical)
        n_embd_spec = self.globals.get("n_embd", {})
        n_embd = int(n_embd_spec.get("low", 768))  # fixed-range yamls have low==high
        valid = _enumerate_valid_gqa_combinations(
            n_embd, h_low, h_high, qk_low, qk_high, qk_step
        )
        if not valid:
            raise ValueError(
                f"StrictGQASearchSpace: no valid (n_head, n_qk_head_dim) "
                f"combinations satisfy n_head * d_qk == {n_embd} within the "
                f"layer-spec ranges (n_head in [{h_low}, {h_high}], "
                f"n_qk_head_dim in [{qk_low}, {qk_high}] step {qk_step}). "
                f"Loosen the layer-spec ranges or change n_embd to enable "
                f"strict-GQA mode."
            )
        self._gqa_cache: List[Tuple[int, int]] = valid
        self._gqa_h_range = (h_low, h_high)
        self._gqa_qk_range = (qk_low, qk_high)
        return valid

    def repair(self, x: Dict[str, Any]) -> Individual:
        ind = super().repair(x)
        if self.no_repair:
            return ind
        valid = self._gqa_constraints_summary()
        h_low, h_high = self._gqa_h_range
        qk_low, qk_high = self._gqa_qk_range
        h_span = max(h_high - h_low, 1)
        qk_span = max(qk_high - qk_low, 1)

        for li in ind["layers"]:
            cur_h = int(li.get("n_head", valid[0][0]))
            cur_qk = int(li.get("n_qk_head_dim", valid[0][1]))

            # Closest valid combination in normalized Manhattan distance.
            def _cost(comb: Tuple[int, int]) -> float:
                h, qk = comb
                return abs(h - cur_h) / h_span + abs(qk - cur_qk) / qk_span

            best_h, best_qk = min(valid, key=_cost)
            li["n_head"] = best_h
            li["n_qk_head_dim"] = best_qk
            li["n_v_head_dim"] = best_qk  # Q/K--V coupling: d_v = d_qk

            # n_kv_group must still divide the snapped n_head; re-clamp.
            cur_kv = int(li.get("n_kv_group", 1))
            cur_kv = max(1, min(cur_kv, best_h))
            if best_h % cur_kv != 0:
                divisors = [g for g in range(1, best_h + 1) if best_h % g == 0]
                cur_kv = min(divisors, key=lambda g: abs(g - cur_kv)) if divisors else 1
            li["n_kv_group"] = cur_kv
        return ind
