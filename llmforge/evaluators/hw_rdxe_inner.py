"""rDXE inner-chip co-search HW evaluator.

For each candidate arch, sweeps a (mac_per_vac, max_chips, wmem_per_core_KB)
grid via the existing run_exp_hw.run_rdxe_eval workflow. The full inner
Pareto front is preserved per-individual under `chip_pareto`; the overall
best-by-`select_by` chip's metrics are promoted to top-level keys for
outer-NSGA selection.

The legacy run_rdxe_eval takes a Population (only to pluck individuals),
so we feed it a duck-typed shim with `.gen=0` and `.individuals=...`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class _PopShim:
    """Just enough to satisfy run_rdxe_eval's `population.individuals`."""
    def __init__(self, individuals):
        self.gen = 0
        self.individuals = individuals
        self.offspring: List[Any] = []


class HwRdxeInner:
    """rDXE inner chip co-search.

    Args:
        prefill_len, decode_len, n_users, ctx: workload knobs.
        select_by: which Pareto metric the outer NSGA inherits per arch.
            Default 'per_tok_uJ' matches run_rdxe_eval's legacy default.
        envelope_filter: if True, restricts selection to chips that fit the
            (area, power) edge envelope.
        area_max_mm2 / area_min_mm2 / power_max_W / power_min_W: envelope.
        verbose: forwarded to legacy logger.
    """

    def __init__(self, *, prefill_len: int = 128, decode_len: int = 32,
                 n_users: int = 1, ctx: Optional[int] = None,
                 select_by: str = "per_tok_uJ",
                 envelope_filter: bool = True,
                 area_max_mm2: float = 800.0, area_min_mm2: float = 0.0,
                 power_max_W: float = 100.0, power_min_W: float = 0.0,
                 verbose: bool = False):
        self.prefill_len = int(prefill_len)
        self.decode_len = int(decode_len)
        self.n_users = int(n_users)
        self.ctx = ctx
        self.select_by = select_by
        self.envelope_filter = bool(envelope_filter)
        self.area_max_mm2 = float(area_max_mm2)
        self.area_min_mm2 = float(area_min_mm2)
        self.power_max_W = float(power_max_W)
        self.power_min_W = float(power_min_W)
        self.verbose = bool(verbose)

    def evaluate(self, ind_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from run_exp_hw import run_rdxe_eval
        # ind_dicts must be Individual-like (the legacy code reads
        # ind["globals"], ind["layers"]) — Individual subclasses dict so
        # the shim is enough.
        shim = _PopShim(ind_dicts)
        raw = run_rdxe_eval(
            shim,                                    # type: ignore[arg-type]
            prefill_len=self.prefill_len,
            decode_len=self.decode_len,
            n_users=self.n_users,
            ctx=self.ctx,
            select_by=self.select_by,
            envelope_filter=self.envelope_filter,
            area_max_mm2=self.area_max_mm2,
            area_min_mm2=self.area_min_mm2,
            power_max_W=self.power_max_W,
            power_min_W=self.power_min_W,
            verbose=self.verbose,
        )
        # run_rdxe_eval already promotes the selected chip's metrics to
        # top-level. Rename `pareto_points` -> `chip_pareto` to match the
        # spec's vocabulary; keep everything else as-is.
        out = []
        for r in raw:
            r2 = dict(r)
            if "pareto_points" in r2:
                r2["chip_pareto"] = r2.pop("pareto_points")
            out.append(r2)
        return out
