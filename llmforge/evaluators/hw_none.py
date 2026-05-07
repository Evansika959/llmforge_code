"""Always-on analytical aux: params_M, kv_cache_bytes, flops_per_token.

Costs nothing — uses the existing Individual.estimate_* helpers from
search_space.py. Composed alongside whatever HW backend the user picks
so these fields are present in every run.
"""

from __future__ import annotations

from typing import Any, Dict, List

from search_space import Individual


class HwNone:
    """Analytical aux. `seq_len` sizes the KV cache and FLOPS per token."""

    def __init__(self, seq_len: int = 256):
        self.seq_len = int(seq_len)

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in ind_dicts:
            ind = d if isinstance(d, Individual) else Individual(
                globals=d.get("globals", {}), layers=d.get("layers", [])
            )
            params_b = ind.estimate_params()
            flops_total = ind.estimate_flops(seq_len=self.seq_len)
            kv_bytes = ind.estimate_kv_cache_size(seq_len=self.seq_len)
            out.append({
                "params_M": params_b / 1e6,
                "params": params_b / 1e6,            # alias for legacy constraint configs
                "flops_per_token": flops_total / max(1, self.seq_len),
                "kv_cache_bytes": kv_bytes,
                "kv_cache_MB": kv_bytes / 1e6,
            })
        return out


def merge_hw_dicts(primary: List[Dict[str, Any]],
                   secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine two aligned aux-dict lists. `primary` keys overwrite `secondary`
    on collision (i.e. measured/substrate metrics win over analytical defaults).
    """
    if len(primary) != len(secondary):
        raise ValueError(
            f"merge_hw_dicts: length mismatch ({len(primary)} vs {len(secondary)})"
        )
    # secondary first, then primary overwrites — matches the docstring.
    return [{**sec, **pri} for pri, sec in zip(primary, secondary)]
