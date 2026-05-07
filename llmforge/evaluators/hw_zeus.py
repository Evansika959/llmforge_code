"""ZEUS HW evaluator — measures the local GPU (A100 in our setup).

Thin wrapper around zeus_eval.run_zeus_eval, with a per-arch hash cache so
identical individuals (e.g. survivors carrying over from previous gens)
don't get re-measured.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def _ind_hash(ind_dict: Dict[str, Any]) -> str:
    g = ind_dict.get("globals", ind_dict)
    mask = g.get("layer_mask") or []
    layers = ind_dict.get("layers", [])
    active = [L for L, m in zip(layers, mask) if m] if mask else list(layers)
    canon = {
        "n_embd": g.get("n_embd"),
        "block_size": g.get("block_size"),
        "use_concat_heads": g.get("use_concat_heads"),
        "layers": [tuple(sorted(L.items())) for L in active],
    }
    return hashlib.sha1(json.dumps(canon, default=str, sort_keys=True).encode()).hexdigest()


class HwZeus:
    """Measure ttft / tpot / energy on the local GPU via ZEUS.

    Args:
        prefill_len, decode_len: workload tokens for the measurement.
        n_repeats, warmup, dtype: ZEUS knobs.
        verbose: print per-individual lines.
        use_kv_cache: route decode through the measurement-only KV-cache
            shim (zeus_kv_cache) so per-token compute matches production
            cached decode (single-query SDPA against cached K, V) instead
            of Evo_GPT's recomputing generate(). Default True. Per-arch
            fallback to non-cached path is automatic if attach-time
            feature guards trip; that fact is surfaced in the per-arch
            result dict via `zeus_kv_cache_used` / `zeus_kv_cache_skip`.
    """

    def __init__(self, prefill_len: int = 128, decode_len: int = 32,
                 n_repeats: int = 1, warmup: int = 1, dtype: str = "bf16",
                 verbose: bool = False, use_kv_cache: bool = True):
        self.prefill_len = int(prefill_len)
        self.decode_len = int(decode_len)
        self.n_repeats = int(n_repeats)
        self.warmup = int(warmup)
        self.dtype = dtype
        self.verbose = verbose
        self.use_kv_cache = bool(use_kv_cache)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self._monitor = None  # lazy: avoid touching CUDA at import time

    def _ensure_monitor(self):
        if self._monitor is None:
            from zeus.monitor import ZeusMonitor
            self._monitor = ZeusMonitor(
                gpu_indices=[0], cpu_indices=[],
                sync_execution_with="torch", approx_instant_energy=True,
            )
        return self._monitor

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from zeus_eval import measure_one
        out: List[Optional[Dict[str, Any]]] = []
        miss_idx, miss_inds = [], []
        for i, ind in enumerate(ind_dicts):
            k = _ind_hash(ind)
            if k in self._cache:
                self.hits += 1
                out.append(self._cache[k])
            else:
                self.misses += 1
                out.append(None)
                miss_idx.append(i)
                miss_inds.append(ind)

        if miss_inds:
            monitor = self._ensure_monitor()
            for j, ind in zip(miss_idx, miss_inds):
                r = measure_one(
                    ind, prefill_len=self.prefill_len, decode_len=self.decode_len,
                    n_repeats=self.n_repeats, warmup=self.warmup,
                    dtype=self.dtype, monitor=monitor,
                    use_kv_cache=self.use_kv_cache,
                )
                self._cache[_ind_hash(ind)] = r
                out[j] = r
                if self.verbose:
                    if r.get("envelope_feasible"):
                        kv_tag = "kv" if r.get("zeus_kv_cache_used") else "no-kv"
                        print(f"  [zeus {j+1}/{len(ind_dicts)}] OK [{kv_tag}]  "
                              f"ttft={r.get('ttft_ms', float('nan')):.2f}ms  "
                              f"tpot={r.get('tpot_ms', float('nan')):.2f}ms  "
                              f"power={r.get('power_W', float('nan')):.1f}W")
                    else:
                        reason = r.get("zeus_error", "unknown")[:80]
                        print(f"  [zeus {j+1}/{len(ind_dicts)}] FAIL  ({reason})")
        return out  # type: ignore[return-value]

    # -- Diagnostics ---------------------------------------------------

    def kv_cache_summary(self, results: List[Dict[str, Any]]) -> str:
        """One-line summary of how many archs in `results` used the KV
        cache (vs fell back to recomputing generate()). Used by the
        run_cosearch dispatcher's per-gen log line."""
        n = len(results)
        if n == 0:
            return "kv-cache n/a"
        used = sum(1 for r in results if r.get("zeus_kv_cache_used"))
        if not self.use_kv_cache:
            return "kv-cache off (--no_kv_cache)"
        skipped_reasons = {
            r.get("zeus_kv_cache_skip") for r in results
            if r.get("zeus_kv_cache_skip")
        }
        tag = f"kv-cache used on {used}/{n}"
        if skipped_reasons:
            sample = next(iter(skipped_reasons))[:60]
            tag += f"  (fallback: {sample}...)" if len(skipped_reasons) > 1 else f"  (fallback: {sample})"
        return tag
