"""Timeloop HW evaluator — dispatches to a published or custom substrate.

Thin wrapper around hw_exp.evaluate_population, with the same prefill/decode
two-pass aggregation that run_exp_hw.run_hw_eval uses. Operates on a list
of Individual dicts directly so the dispatcher can call us without
constructing a Population.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from search_space import Individual

# Substrates exposed at the CLI -> ARCH_CONFIGS key in hw_exp.py
SUBSTRATE_MAP = {
    "eyeriss": "eyeriss",
    "simba": "simba",
    "gemmini": "gemmini",
    "flat_edge": "flat_edge",
    # dxe (strict) has rigid mapper constraints; many search-space GEMM
    # shapes (e.g. n_head·n_qk_head_dim values that don't tile into the
    # PE array) can't be mapped → mapper exhausts candidates → exception.
    # dxe_relaxed has the same chip but loosened mapper constraints; this
    # is what the legacy rDXE inner search uses (run_exp_hw.py:387).
    "dxe": "dxe",
    "dxe_relaxed": "dxe_relaxed",
}


def _override_block_size(individuals, block_size: int):
    orig = []
    for ind in individuals:
        g = ind["globals"]
        orig.append(int(g.get("block_size", 0)))
        g["block_size"] = int(block_size)
    return orig


def _restore_block_size(individuals, originals):
    for ind, orig in zip(individuals, originals):
        ind["globals"]["block_size"] = int(orig)


class HwTimeloop:
    """Run Timeloop on a fixed substrate. Two-pass (prefill+decode) when
    both lengths are positive; single-pass otherwise."""

    def __init__(self, substrate: str, prefill_len: int = 128, decode_len: int = 32):
        if substrate not in SUBSTRATE_MAP:
            raise ValueError(
                f"Unknown timeloop substrate '{substrate}'. "
                f"Available: {sorted(SUBSTRATE_MAP)}")
        self.substrate = substrate
        self.arch = SUBSTRATE_MAP[substrate]
        self.prefill_len = int(prefill_len)
        self.decode_len = int(decode_len)

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from hw_exp import evaluate_population
        # Coerce to Individual so estimate_* is available downstream if needed
        individuals = [
            d if isinstance(d, Individual)
              else Individual(globals=d.get("globals", {}), layers=d.get("layers", []))
            for d in ind_dicts
        ]

        start = time.time()
        total_tokens = self.prefill_len + self.decode_len

        if self.prefill_len > 0 and self.decode_len > 0:
            orig = _override_block_size(individuals, self.prefill_len)
            print(f"  [timeloop:{self.arch}] prefill pass ({self.prefill_len} tok)")
            pf = evaluate_population(individuals,
                                     base_work_dir=f"./hw_eval/runs/{self.arch}/prefill",
                                     arch=self.arch, mode="prefill")

            _restore_block_size(individuals, orig)
            orig = _override_block_size(individuals, self.decode_len)
            print(f"  [timeloop:{self.arch}] decode pass ({self.decode_len} tok)")
            dc = evaluate_population(individuals,
                                     base_work_dir=f"./hw_eval/runs/{self.arch}/decode",
                                     arch=self.arch, mode="decode")
            _restore_block_size(individuals, orig)

            out = []
            for p, d in zip(pf, dc):
                pf_e = (p.get("energy_uJ", 0) if p else 0)
                pf_c = (p.get("cycles", 0) if p else 0)
                dc_e = (d.get("energy_uJ", 0) if d else 0)
                dc_c = (d.get("cycles", 0) if d else 0)
                rec: Dict[str, Any] = {
                    "energy_uJ": pf_e + dc_e * self.decode_len,
                    "cycles": pf_c + dc_c * self.decode_len,
                }
                for k in ("total_ops", "total_memory_accesses",
                         "fusion_saved_energy_uJ", "fusion_saved_cycles"):
                    rec[k] = (p.get(k, 0) if p else 0) + (d.get(k, 0) if d else 0) * self.decode_len
                if total_tokens > 0:
                    rec["energy_per_token_uJ"] = rec["energy_uJ"] / total_tokens
                    rec["cycles_per_token"] = rec["cycles"] / total_tokens
                    rec["token_delay"] = rec["cycles_per_token"] / 1e9
                rec["edp"] = rec["energy_uJ"] * rec["cycles"] / 10e6
                if p:
                    rec["prefill_energy_uJ"] = pf_e
                    rec["prefill_cycles"] = pf_c
                    rec["ttft"] = pf_c / 1e9
                if d:
                    rec["decode_energy_uJ"] = dc_e
                    rec["decode_cycles"] = dc_c
                    rec["tpot"] = dc_c / 1e9
                # Surface D-axis padding annotations from hw_exp's
                # `_pad_D_for_arch` (only set on substrates with a strict
                # spatial mesh, currently the four DXE variants). Lets
                # downstream consumers tell which ops needed padding and
                # by how much. Prefill and decode both contribute.
                pf_pads = (p.get("padded_ops") or []) if p else []
                dc_pads = (d.get("padded_ops") or []) if d else []
                if pf_pads or dc_pads:
                    rec["padded_ops"] = {
                        "prefill": pf_pads,
                        "decode":  dc_pads,
                    }
                    rec["padded_op_count"] = len(pf_pads) + len(dc_pads)
                out.append(rec)
        else:
            out = evaluate_population(individuals,
                                      base_work_dir=f"./hw_eval/runs/{self.arch}",
                                      arch=self.arch, mode="prefill")

        print(f"  [timeloop:{self.arch}] {len(ind_dicts)} inds in {time.time()-start:.1f}s")
        return out
