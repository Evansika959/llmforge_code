"""NSGA-II search with hardware (Timeloop) evaluation.

Supports two modes for software quality estimation:
  1. Real training on remote GPU hosts  (default)
  2. Surrogate model prediction          (--surrogate)

Hardware metrics (energy, latency, etc.) are always computed via Timeloop
for the selected architecture and merged into each individual's evaluation.
This allows NSGA-II objectives/constraints to reference both SW metrics
(val_loss, params) and HW metrics (energy_per_token_uJ, token_delay, etc.).

Examples:
  # Real training + Eyeriss HW eval
  python run_exp_hw.py --arch eyeriss --objectives val_loss energy_per_token_uJ

  # Surrogate + DXE HW eval (no remote training needed)
  python run_exp_hw.py --arch dxe --surrogate --objectives val_loss token_delay

  # Surrogate + Simba, with custom constraints
  python run_exp_hw.py --arch simba --surrogate \\
      --objectives val_loss energy_per_token_uJ \\
      --constraint val_loss=3.8 --constraint energy_per_token_uJ=50
"""

from nsga2 import Population, cons_value
from typing import List, Dict, Any, Tuple, Optional
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer
from hw_exp import ARCH_CONFIGS, evaluate_population

ALL_ARCH_CHOICES = list(ARCH_CONFIGS.keys()) + ["rDXE"]
import logging
import time
import os
import argparse
import random
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
for _name in ("paramiko", "paramiko.transport", "fabric", "invoke",
              "timeloopfe", "Specification", "accelergy"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class _TimeloopSpamFilter(logging.Filter):
    """Suppress noisy Timeloop/Dataspace INFO records without having to
    enumerate every lazily-created logger name. Attached at HANDLER level
    (not logger level) so it applies regardless of which library-specific
    logger emitted the record."""
    _SUBSTR = ('Dataspace', 'Processor', 'Mapspace', 'Probspace',
               'Constraints', 'ReferenceLoader', 'timeloopfe',
               'Specification', 'accelergy')

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO:
            return True
        name = record.name or ''
        return not any(s in name for s in self._SUBSTR)


# Attach to every handler on the root logger — this catches records whose
# originating loggers don't propagate (e.g. libraries that install their own
# handlers). Also apply to the root logger itself for any late-bound handlers.
_spam_filter = _TimeloopSpamFilter()
logging.getLogger().addFilter(_spam_filter)
for _h in logging.getLogger().handlers:
    _h.addFilter(_spam_filter)


# Nuke any pre-existing Timeloop loggers at level INFO → WARNING. Catches
# loggers created at module-import time (e.g., by earlier timeloopfe imports).
def _silence_timeloop_loggers():
    for name in list(logging.root.manager.loggerDict.keys()):
        if any(s in name for s in _TimeloopSpamFilter._SUBSTR):
            logging.getLogger(name).setLevel(logging.WARNING)


_silence_timeloop_loggers()


# ---------------------------------------------------------------------------
# Helpers (shared with run_exp.py)
# ---------------------------------------------------------------------------

def load_hosts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hosts file not found: {path}")
    _, ext = os.path.splitext(path)
    if ext.lower() not in (".yaml", ".yml"):
        raise ValueError("Hosts file must be a YAML file")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Hosts YAML must be a top-level list of IPs")
    hosts = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    if not hosts:
        raise ValueError(f"No hosts parsed from file: {path}")
    return hosts


def load_search_space_from_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Search space file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Search space YAML must define 'global_spec' and 'layer_spec'.")
    global_spec = data.get("global_spec")
    layer_spec = data.get("layer_spec")
    if not isinstance(global_spec, dict) or not isinstance(layer_spec, dict):
        raise ValueError("Search space YAML missing 'global_spec' or 'layer_spec'.")
    return global_spec, layer_spec


def load_initial_individuals(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initial population file not found: {path}")
    _, ext = os.path.splitext(path)
    with open(path, "r", encoding="utf-8") as f:
        if ext.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif ext.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError("Initial population file must be .json, .yaml, or .yml")
    if isinstance(data, dict) and "individuals" in data and isinstance(data["individuals"], list):
        data = data["individuals"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Initial population must be a list of individual dicts")
    individuals = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Individual at index {idx} is not a dict")
        individuals.append(entry)
    if not individuals:
        raise ValueError("No individuals found")
    return individuals


def parse_constraint_arg(entry: str) -> Tuple[str, float]:
    """Accepts 'key=N' / 'key<=N' (upper bound) and 'key>=N' (lower bound).
    Lower bounds are mangled to '<key>_min' so nsga2.cons_value() can
    distinguish direction without changing cons_settings' dict shape."""
    for op in (">=", "<=", "="):
        if op in entry:
            key, value = entry.split(op, 1)
            key = key.strip()
            if not key:
                raise argparse.ArgumentTypeError("Constraint key cannot be empty")
            if op == ">=":
                key = f"{key}_min"
            try:
                return key, float(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Constraint value for '{key}' must be numeric"
                )
    raise argparse.ArgumentTypeError(
        "Constraints must be 'key=N', 'key<=N', or 'key>=N'"
    )


# ---------------------------------------------------------------------------
# HW evaluation wrapper
# ---------------------------------------------------------------------------

def _override_block_size(individuals: list, block_size: int) -> Tuple[list, List[int]]:
    """Temporarily override block_size on individuals. Returns (modified list, original values)."""
    originals = []
    for ind in individuals:
        originals.append(ind["globals"]["block_size"])
        ind["globals"]["block_size"] = block_size
    return individuals, originals


def _restore_block_size(individuals: list, originals: List[int]) -> None:
    for ind, orig in zip(individuals, originals):
        ind["globals"]["block_size"] = orig


def run_hw_eval(population: Population, arch: str,
                prefill_len: int = 0, decode_len: int = 0) -> List[dict]:
    """Run Timeloop HW evaluation on current individuals or offspring.

    If prefill_len and decode_len are both set, runs two passes:
      1) prefill mode with block_size=prefill_len  (batch of tokens)
      2) decode mode  with block_size=decode_len   (token-by-token, KV cache=decode_len)
    Energy/cycles are summed; per-token metrics are averaged over (prefill_len + decode_len).

    If neither is set, runs a single prefill pass using the individual's original block_size.
    """
    if population.gen == 0:
        individuals = population.individuals
    else:
        individuals = population.offspring

    start = time.time()
    total_tokens = prefill_len + decode_len

    if prefill_len > 0 and decode_len > 0:
        # Prefill pass
        individuals, orig_bs = _override_block_size(individuals, prefill_len)
        print(f"  Prefill pass ({prefill_len} tokens)...")
        prefill_data = evaluate_population(individuals, base_work_dir=f"./hw_eval/runs/{arch}/prefill",
                                           arch=arch, mode="prefill")

        # Decode pass
        _restore_block_size(individuals, orig_bs)
        individuals, orig_bs = _override_block_size(individuals, decode_len)
        print(f"  Decode pass ({decode_len} tokens)...")
        decode_data = evaluate_population(individuals, base_work_dir=f"./hw_eval/runs/{arch}/decode",
                                          arch=arch, mode="decode")
        _restore_block_size(individuals, orig_bs)

        # Combine: decode metrics are per-token (proj_seq=1), scale by decode_len for totals
        hw_data = []
        for pf, dc in zip(prefill_data, decode_data):
            combined = {}
            pf_energy = pf.get("energy_uJ", 0) if pf else 0
            pf_cycles = pf.get("cycles", 0) if pf else 0
            dc_energy_per_tok = dc.get("energy_uJ", 0) if dc else 0
            dc_cycles_per_tok = dc.get("cycles", 0) if dc else 0

            combined["energy_uJ"] = pf_energy + dc_energy_per_tok * decode_len
            combined["cycles"] = pf_cycles + dc_cycles_per_tok * decode_len
            for k in ["total_ops", "total_memory_accesses", "fusion_saved_energy_uJ", "fusion_saved_cycles"]:
                pv = pf.get(k, 0) if pf else 0
                dv = dc.get(k, 0) if dc else 0
                combined[k] = pv + dv * decode_len

            if total_tokens > 0:
                combined["energy_per_token_uJ"] = combined["energy_uJ"] / total_tokens
                combined["cycles_per_token"] = combined["cycles"] / total_tokens
                combined["token_delay"] = combined["cycles_per_token"] / 1e9
            combined["edp"] = combined["energy_uJ"] * combined["cycles"] / 10e6

            # prefill/decode breakdowns (decode values are per-token)
            if pf:
                combined["prefill_energy_uJ"] = pf_energy
                combined["prefill_cycles"] = pf_cycles
                combined["ttft"] = pf_cycles / 1e9
            if dc:
                combined["decode_energy_uJ"] = dc_energy_per_tok
                combined["decode_cycles"] = dc_cycles_per_tok
                combined["tpot"] = dc_cycles_per_tok / 1e9

            hw_data.append(combined)
    else:
        hw_data = evaluate_population(individuals, base_work_dir=f"./hw_eval/runs/{arch}",
                                      arch=arch, mode="prefill")

    elapsed = time.time() - start
    print(f"HW evaluation ({arch}) completed in {elapsed:.1f}s")
    return hw_data


# ---------------------------------------------------------------------------
# rDXE ring simulator evaluation
# ---------------------------------------------------------------------------

# ---- HW design-space grid for per-individual Pareto sweep ----
# Each (mac_per_vac, max_chips, wmem_per_core_KB) triple is one HW candidate.
# Keep this small — the grid size multiplies the per-individual eval cost.
# HW design-space grid swept per NSGA individual. Expanded from the original
# 12-config grid because deep/heterogeneous models can fail to pack when
# max_chips forces >2 layers per chip onto a small-WMEM config. Grid is now
# 3 × 3 × 5 = 45 configs; prefetch cost amortizes over the whole population.
_RDXE_SWEEP_MAC_PER_VAC = (16, 32, 64)
_RDXE_SWEEP_MAX_CHIPS   = (8, 16, 32)
_RDXE_SWEEP_WMEM_KB     = (24, 48, 96, 192, 384)


def _layer_dict_to_spec(layer: dict, default_n_embd: int) -> dict:
    """Convert an NSGA Individual-layer dict into the rDXE layer-spec format."""
    d  = default_n_embd
    nh = int(layer.get("n_head", 8))
    return {
        "n_head":            nh,
        "n_kv_group":        int(layer.get("n_kv_group", nh)),
        "n_qk_head_dim":     int(layer.get("n_qk_head_dim", d // max(1, nh))),
        "n_v_head_dim":      int(layer.get("n_v_head_dim",
                                           layer.get("n_qk_head_dim", d // max(1, nh)))),
        "mlp_size":          int(layer.get("mlp_size", 4 * d)),
        "n_cproj":           1,
        "attention_variant": layer.get("attention_variant", "infinite"),
    }


def _pareto_front(points: List[dict], keys: List[str]) -> List[dict]:
    """Return the non-dominated subset (minimize every key)."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p: continue
            if all(q[k] <= p[k] for k in keys) and any(q[k] < p[k] for k in keys):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


# ---- Canonical workload defaults (shared with rDXE demo) ----
# Pinned to match the edge-LLM chat workload profile:
#   ctx=2048, prefill=512, decode=256, users=1.
# CLI flags in main() still allow overrides but these are the defaults.
RDXE_DEFAULT_CTX      = 2048
RDXE_DEFAULT_PREFILL  = 512
RDXE_DEFAULT_DECODE   = 256
RDXE_DEFAULT_N_USERS  = 1

# Edge HW envelope — filters the Pareto candidate set BEFORE selection.
# If an individual has no candidate that fits the envelope it's marked
# infeasible (inf metrics). These are soft defaults; override via CLI.
RDXE_DEFAULT_AREA_MAX_MM2 = 2500.0
RDXE_DEFAULT_AREA_MIN_MM2 = 100.0
RDXE_DEFAULT_POWER_MAX_W  = 2.0
RDXE_DEFAULT_POWER_MIN_W  = 0.005


def run_rdxe_eval(population: Population,
                  n_chips: int = 8,                # retained for CLI compat
                  layer_assignment: str = "round_robin",  # (legacy, unused)
                  prefill_len: int = RDXE_DEFAULT_PREFILL,
                  decode_len: int = RDXE_DEFAULT_DECODE,
                  n_users: int = RDXE_DEFAULT_N_USERS,
                  ctx: Optional[int] = RDXE_DEFAULT_CTX,
                  pareto_objs: Tuple[str, ...] = ("per_tok_uJ",
                                                   "tpot_ms", "ttft_ms"),
                  select_by: str = "per_tok_uJ",
                  area_max_mm2: float = RDXE_DEFAULT_AREA_MAX_MM2,
                  area_min_mm2: float = RDXE_DEFAULT_AREA_MIN_MM2,
                  power_max_W:  float = RDXE_DEFAULT_POWER_MAX_W,
                  power_min_W:  float = RDXE_DEFAULT_POWER_MIN_W,
                  envelope_filter: bool = True,
                  verbose: bool = False,
                  n_workers: int = 8) -> List[dict]:
    """Timeloop-backed rDXE ring evaluator for an NSGA population.

    Replaces the legacy single-point evaluator. For each individual we:
      1. Profile the active layers → per-layer WMEM / KV$ / MAC demand.
      2. Sweep a small HW grid (n_mac_per_vac, max_chips, wmem_per_core).
      3. For each feasible config, pack layers onto chips and run the
         Timeloop-backed ring simulation (reuses the full workflow:
         DXE KV//corrections, per-op mapping for decode +
         Timeloop-mapped prefill projections).
      4. Compute the Pareto front over (per_tok_uJ, tpot_ms, ttft_ms).
      5. Return the `select_by`-argmin point as the primary hw_data dict.
         The full Pareto front is attached under key `pareto_points` for
         logging / post-hoc analysis.

    The scalar fields preserved from the legacy API (ttft, tpot,
    energy_per_token_uJ, …) remain available so NSGA objs/cons keep working.
    """
    import sys
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _parent    = os.path.dirname(_repo_root)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    import rDXE_sim.experiments.timeloop_workflow as wf
    from rDXE_sim.core.timeloop_evaluator import (
        TimeloopEvaluator, enumerate_gemm_shapes_decode,
    )

    if ctx is None:
        # Fall back to canonical default if caller did not supply one.
        ctx = RDXE_DEFAULT_CTX

    # ---- Pick individuals that need evaluation ----
    individuals = (population.individuals if population.gen == 0
                   else population.offspring)

    # ---- Pre-build the HW sweep list ----
    sweep = [(mpv, mc, wkb)
             for mpv in _RDXE_SWEEP_MAC_PER_VAC
             for mc in _RDXE_SWEEP_MAX_CHIPS
             for wkb in _RDXE_SWEEP_WMEM_KB]

    # ---- One evaluator per n_mac_per_vac variant (Timeloop re-maps each) ----
    evaluators = {
        mpv: TimeloopEvaluator(arch='dxe_relaxed', verbose=False,
                               n_mac_per_vac=mpv)
        for mpv in _RDXE_SWEEP_MAC_PER_VAC
    }

    # ---- Gather every decode shape across population for parallel prefetch ----
    # Prefill is analytical (simulate_ring uses closed-form cycles/energy),
    # so we don't enumerate prefill shapes into the Timeloop prefetch.
    decode_shapes_all = set()
    representative_layers = []
    for ind in individuals:
        g = ind["globals"]
        layers_raw = ind["layers"]
        mask = g.get("layer_mask", [True] * len(layers_raw))
        active = [l for l, m in zip(layers_raw, mask) if m]
        if not active:
            representative_layers.append(None)
            continue
        rep = _layer_dict_to_spec(active[0], g["n_embd"])
        representative_layers.append((rep, len(active), g["n_embd"]))
        for (_, ic, oc, sl) in enumerate_gemm_shapes_decode(rep, g["n_embd"], ctx):
            decode_shapes_all.add((ic, oc, max(1, sl * n_users)))

    all_shapes = decode_shapes_all
    if verbose:
        print(f"  [rDXE] unique decode shapes to map: {len(all_shapes)}")
    for mpv, ev in evaluators.items():
        if verbose:
            print(f"  [rDXE] prefetch variant mac_per_vac={mpv} ...")
        ev.prefetch(list(all_shapes), n_workers=n_workers)

    # ---- Per-individual Pareto sweep ----
    start = time.time()
    hw_data = []
    n = len(individuals)
    for i, (ind, rep_info) in enumerate(zip(individuals, representative_layers)):
        if rep_info is None:
            # Empty / all-masked individual → inf metrics keep NSGA honest
            hw_data.append({"ttft": float("inf"), "tpot": float("inf"),
                            "energy_per_token_uJ": float("inf"),
                            "pareto_points": []})
            continue
        rep_layer, n_active, n_embd = rep_info

        # Build a profile-ready homogeneous config (same layer × n_active).
        ind_cfg = {
            "n_layer":    n_active,
            "n_embd":     n_embd,
            "n_head":     rep_layer["n_head"],
            "n_kv_group": rep_layer["n_kv_group"],
            "hd":         rep_layer["n_qk_head_dim"],
            "mlp":        rep_layer["mlp_size"],
        }
        profile, info = wf.profile_model(ind_cfg, ctx=ctx, n_users=n_users)

        candidates = []
        for (mpv, max_chips, wmem_kb) in sweep:
            saved = wf.WMEM_PER_CORE_OPTIONS
            wf.WMEM_PER_CORE_OPTIONS = [wmem_kb * 1024]
            try:
                packing = wf.pack_balanced(
                    profile,
                    max_chips=min(max_chips, n_active),
                    max_wmem_total_B=None,
                    n_mac_per_vac=mpv,
                )
            finally:
                wf.WMEM_PER_CORE_OPTIONS = saved
            if packing is None:
                continue
            r = wf.simulate_ring(info, packing, ctx, evaluators[mpv],
                                 prefill_length=prefill_len,
                                 decode_length=decode_len,
                                 n_users=n_users)
            r["sweep_mac_per_vac"]    = mpv
            r["sweep_max_chips"]      = max_chips
            r["sweep_wmem_per_core_KB"] = wmem_kb
            # Power (W) during decode = per-token energy × steady-state rate.
            #   per_tok_uJ (µJ) × n_users / tpot_ms (ms) → mW → /1000 → W
            r["power_W"] = (r["per_tok_uJ"] * n_users
                            / max(1e-9, r["tpot_ms"])) * 1e-3
            candidates.append(r)

        if not candidates:
            hw_data.append({"ttft": float("inf"), "tpot": float("inf"),
                            "energy_per_token_uJ": float("inf"),
                            "pareto_points": []})
            continue

        # ---- Optional envelope filter: restricts selection to HW configs ----
        # that fit the edge budget (area + power). When disabled, the selector
        # sees the full Pareto candidate set (raw argmin on `select_by`).
        #
        # Rationale:
        #   ON  — production runs where you have a known HW budget. The
        #         Pareto-set → scalar mapping picks the best operating point
        #         that fits. Falls back to least-violating if nothing fits
        #         (NSGA --constraint then marks the individual infeasible).
        #   OFF — HW-agnostic exploration / sensitivity studies. The search
        #         ranks models by their globally-best HW config, and any
        #         area/power constraints applied downstream still flag
        #         envelope violations on the selected point.
        if envelope_filter:
            feasible = [r for r in candidates
                        if area_min_mm2 <= r["total_area_mm2"] <= area_max_mm2
                        and power_min_W  <= r["power_W"]        <= power_max_W]

            if feasible:
                candidates_for_selection = feasible
                envelope_feasible = True
            else:
                # Least-violating: min overshoot on area + power (relative)
                def _overshoot(r):
                    a_over = max(0, r["total_area_mm2"] - area_max_mm2) / area_max_mm2
                    a_under = max(0, area_min_mm2 - r["total_area_mm2"]) / max(1, area_min_mm2)
                    p_over = max(0, r["power_W"] - power_max_W) / power_max_W
                    p_under = max(0, power_min_W - r["power_W"]) / max(1e-9, power_min_W)
                    return a_over + a_under + p_over + p_under
                candidates_for_selection = [min(candidates, key=_overshoot)]
                envelope_feasible = False
        else:
            # No envelope filter — raw argmin on the full candidate set.
            candidates_for_selection = candidates
            # Still report whether the final selection fits the envelope
            # (computed after the argmin below).
            envelope_feasible = None

        # Pareto front is still computed on the FULL candidate set for
        # downstream analysis (`pareto_points`), not just the feasible
        # subset. This preserves visibility into area/power tradeoffs.
        front = _pareto_front(candidates, list(pareto_objs))
        sel   = min(candidates_for_selection, key=lambda r: r[select_by])

        # When envelope_filter was off, retrospectively flag whether the
        # selected point actually fits the envelope (useful for reporting).
        if envelope_feasible is None:
            envelope_feasible = (area_min_mm2 <= sel["total_area_mm2"] <= area_max_mm2
                                 and power_min_W <= sel["power_W"] <= power_max_W)

        hw_data.append({
            # Scalar fields consumed by NSGA objs/cons (legacy names kept).
            "ttft":                  sel["ttft_ms"] / 1e3,     # seconds
            "tpot":                  sel["tpot_ms"] / 1e3,     # seconds
            "ttft_ms":               sel["ttft_ms"],
            "tpot_ms":               sel["tpot_ms"],
            "energy_per_token_uJ":   sel["per_tok_uJ"],
            # prefill+decode session-amortized E/tok kept in the CSV/aux
            # but NOT used as a primary NSGA objective (prefill is analytical).
            "session_e_per_tok_uJ":  sel.get("session_e_per_tok_uJ",
                                              sel["per_tok_uJ"]),
            "total_area_mm2":        sel["total_area_mm2"],
            "power_W":               sel["power_W"],
            "mac_util_pct":          sel["mac_util_pct"],
            "n_chips":               sel["n_chips"],
            "chip_macs":             sel["chip_macs"],
            "selected_mac_per_vac":  sel["sweep_mac_per_vac"],
            "selected_max_chips":    sel["sweep_max_chips"],
            "selected_wmem_KB":      sel["sweep_wmem_per_core_KB"],
            # Envelope-feasibility flag — True if the selected point fits
            # the edge HW budget; False means we returned the least-violating
            # point and NSGA --constraint should mark this infeasible.
            "envelope_feasible":     envelope_feasible,
            # Legacy compatibility (aliases)
            "tokens_per_second":     sel["throughput_tps"],
            "inter_chip_comm_energy_uJ": sel["e_hop_uJ"],
            "pipeline_depth":        sel["n_chips"],
            # Pareto front — keyed by (TTFT, TPOT, E/tok) for this arch.
            # Also carries area + power for post-hoc HW-budget filtering.
            "pareto_points": [
                {k: p[k] for k in ("sweep_mac_per_vac", "sweep_max_chips",
                                    "sweep_wmem_per_core_KB",
                                    "ttft_ms", "tpot_ms",
                                    "per_tok_uJ", "session_e_per_tok_uJ",
                                    "total_area_mm2", "power_W",
                                    "mac_util_pct",
                                    "n_chips", "chip_macs")}
                for p in front
            ],
        })
        if verbose:
            print(f"\r  rDXE eval [{i+1}/{n}]  "
                  f"Pareto={len(front):>2d}/{len(candidates)} configs  ",
                  end="", flush=True)
    if verbose:
        print()

    elapsed = time.time() - start
    # Terse single-line summary even when quiet: how many individuals
    # evaluated, how many sit inside the envelope, and wall time.
    n_feas = sum(1 for h in hw_data
                 if h.get("envelope_feasible") is True)
    print(f"[rDXE] eval: {len(hw_data)} ind | {n_feas} envelope-feasible | "
          f"{len(sweep)} HW configs/ind | {elapsed:.1f}s")
    return hw_data


# ---------------------------------------------------------------------------
# Available HW metrics (for help text)
# ---------------------------------------------------------------------------

HW_METRICS_HELP = """
Available HW metrics for objectives/constraints:
  ttft                    Time to first token (seconds, prefill latency, requires --prefill_len)
  tpot                    Time per output token (seconds, decode latency, requires --decode_len)
  energy_per_token_uJ     Energy per token (microjoules, prefill+decode combined)
  energy_uJ               Total energy (microjoules)
  cycles                  Total computation cycles
  cycles_per_token        Cycles per token
  token_delay             Token latency (seconds, assumes 1GHz clock)
  edp                     Energy-Delay Product
  utilization_pct         Hardware utilization percentage
  total_ops               Total floating-point operations
  fusion_saved_energy_uJ  Energy savings from on-chip fusion

Available SW metrics for objectives/constraints:
  val_loss                Validation loss (from training or surrogate)
  params                  Parameter count (millions)
"""


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-II search with hardware evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Hardware architectures (Timeloop-based):
  gemmini        Gemmini systolic array (32nm, legacy cached)
  flat_edge      FLAT-Edge fused attention dataflow (ASPLOS 2023)
  simba_edge     Simba-based edge architecture
  dxe_relaxed    rDXE decoder engine, Timeloop only (relaxed constraints)

Hardware architecture (ring simulator):
  rDXE           rDXE multi-chip ring simulator (KV cache,  pipeline)
                 Use with --n_chips and --layer_assignment
{HW_METRICS_HELP}
Examples:
  # Real training + Eyeriss
  python run_exp_hw.py --arch eyeriss --objectives val_loss energy_per_token_uJ

  # Surrogate + DXE Timeloop
  python run_exp_hw.py --arch dxe --surrogate --objectives val_loss token_delay

  # Surrogate + rDXE ring (8-chip pipeline)
  python run_exp_hw.py --arch rDXE --surrogate --n_chips 8 \\
      --objectives val_loss ttft tpot energy_per_token_uJ \\
      --constraint val_loss=3.8

  # Real training + rDXE ring co-search
  python run_exp_hw.py --arch rDXE --n_chips 8 --layer_assignment balanced \\
      --objectives val_loss ttft tpot energy_per_token_uJ
""",
    )

    # --- Architecture ---
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=ALL_ARCH_CHOICES,
        help="Hardware architecture for evaluation.",
    )
    parser.add_argument("--n_chips", type=int, default=8,
                        help="Number of chips in rDXE ring (only used with --arch rDXE).")
    parser.add_argument("--layer_assignment", type=str, default="round_robin",
                        choices=["round_robin", "balanced", "single_layer"],
                        help="Layer-to-chip assignment strategy (only used with --arch rDXE).")

    # --- Surrogate switch ---
    parser.add_argument(
        "--surrogate",
        action="store_true",
        default=False,
        help="Use surrogate model to predict val_loss instead of real training on remote hosts.",
    )
    parser.add_argument(
        "--surrogate_ckpt",
        type=str,
        default="surrogate/ckpts/model_flex40_optimized.pt",
        help="Path to surrogate model checkpoint (.pt with .json sidecar).",
    )

    # --- Remote training args (ignored when --surrogate is set) ---
    parser.add_argument("--hosts-file", type=str, default="script/examples/hosts_example.yaml",
                        help="Path to YAML hosts file (ignored with --surrogate)")
    parser.add_argument("--user", type=str, default=os.environ.get("USER", "anon"), help="SSH username")
    parser.add_argument("--key", type=str, default="$HOME/.ssh/id_rsa", help="SSH private key path")
    parser.add_argument("--conda_env", type=str, default="llmforge", help="Remote conda environment")
    parser.add_argument("--max_iters", type=int, default=10000, help="Max training iterations per evaluation")
    parser.add_argument("--dataset", type=str, default="minipile", help="Training dataset")
    parser.add_argument("--timeout", type=int, default=10000, help="Remote job timeout (seconds)")

    # --- Search configuration ---
    parser.add_argument("--pop_size", type=int, default=16, help="Population size")
    parser.add_argument("--max_layers", type=int, default=10, help="Max layers (L_max)")
    parser.add_argument("--min_layers", type=int, default=1, help="Min layers (L_min)")
    parser.add_argument("--offspring", type=int, default=8, help="Offspring per generation")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations")
    parser.add_argument("--crossover_rate", type=float, default=0.9, help="Crossover rate")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--exp_name", type=str, default="hw_search", help="Experiment name")
    parser.add_argument(
        "--search_space_config",
        type=str,
        default="search_space_def/default_search_space.yaml",
        help="Search space YAML definition",
    )
    parser.add_argument(
        "--init_individuals",
        type=str,
        default=None,
        help="Path to predefined individuals (JSON/YAML)",
    )

    # --- HW evaluation sequence ---
    # Canonical edge-LLM chat-workload defaults — pinned here and mirrored
    # by the rDXE demo (rDXE_sim/scripts/reproduce_shape_study.sh).
    parser.add_argument("--prefill_len", type=int, default=RDXE_DEFAULT_PREFILL,
                        help=f"Prefill sequence length for HW eval "
                             f"(default {RDXE_DEFAULT_PREFILL} — typical prompt length).")
    parser.add_argument("--decode_len", type=int, default=RDXE_DEFAULT_DECODE,
                        help=f"Decode sequence length for HW eval "
                             f"(default {RDXE_DEFAULT_DECODE} — typical chat-response length). "
                             f"Decode uses token-by-token mode with KV cache.")
    parser.add_argument("--ctx_len", type=int, default=RDXE_DEFAULT_CTX,
                        help=f"KV-cache context window used for rDXE decode "
                             f"(default {RDXE_DEFAULT_CTX} — typical edge-LLM ctx).")
    parser.add_argument("--n_users", type=int, default=RDXE_DEFAULT_N_USERS,
                        help=f"Concurrent users in one forward pass (rDXE only; "
                             f"decode batch size). Default {RDXE_DEFAULT_N_USERS} — "
                             f"single-user edge.")

    # --- Edge HW envelope (rDXE only; filters the Pareto set BEFORE selection) ---
    parser.add_argument("--rdxe_area_max_mm2", type=float,
                        default=RDXE_DEFAULT_AREA_MAX_MM2,
                        help=f"Max silicon area allowed for a single HW config "
                             f"(default {RDXE_DEFAULT_AREA_MAX_MM2} mm²).")
    parser.add_argument("--rdxe_area_min_mm2", type=float,
                        default=RDXE_DEFAULT_AREA_MIN_MM2,
                        help=f"Min silicon area (default {RDXE_DEFAULT_AREA_MIN_MM2} mm²).")
    parser.add_argument("--rdxe_power_max_W", type=float,
                        default=RDXE_DEFAULT_POWER_MAX_W,
                        help=f"Max decode power (default {RDXE_DEFAULT_POWER_MAX_W} W).")
    parser.add_argument("--rdxe_power_min_W", type=float,
                        default=RDXE_DEFAULT_POWER_MIN_W,
                        help=f"Min decode power (default {RDXE_DEFAULT_POWER_MIN_W} W).")
    parser.add_argument("--rdxe_envelope_filter", dest="rdxe_envelope_filter",
                        action="store_true", default=True,
                        help="Filter rDXE HW candidates by the area/power "
                             "envelope BEFORE scalar selection (default on). "
                             "Ensures the HW config NSGA sees fits the edge budget.")
    parser.add_argument("--no_rdxe_envelope_filter",
                        dest="rdxe_envelope_filter", action="store_false",
                        help="Disable the envelope pre-filter; select raw "
                             "argmin on the full Pareto candidate set. Useful "
                             "for HW-agnostic sensitivity studies.")
    parser.add_argument("--rdxe_verbose", action="store_true", default=False,
                        help="Print per-shape and per-individual rDXE progress. "
                             "Off by default — NSGA co-search logs stay compact.")

    # --- Objectives & constraints ---
    parser.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        default=["val_loss", "energy_per_token_uJ"],
        help="Objectives to minimize (SW and/or HW metrics).",
    )
    parser.add_argument(
        "--max_params",
        type=float,
        default=800_000_000,
        help="Default parameter count constraint.",
    )
    parser.add_argument(
        "--max_val_loss",
        type=float,
        default=3.6,
        help="Default validation loss constraint.",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        type=parse_constraint_arg,
        metavar="KEY=VALUE",
        help="Custom constraint thresholds (e.g., --constraint val_loss=3.8 --constraint energy_per_token_uJ=50).",
    )

    args = parser.parse_args()
    random.seed(45)

    # --- Resolve paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = args.search_space_config
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)

    surrogate_ckpt = args.surrogate_ckpt
    if not os.path.isabs(surrogate_ckpt):
        surrogate_ckpt = os.path.join(script_dir, surrogate_ckpt)

    # --- Load search space ---
    global_spec, layer_spec = load_search_space_from_yaml(config_path)
    search_space = HeteroSearchSpace.from_dicts(global_spec, layer_spec,
                                                 L_max=args.max_layers,
                                                 L_min=args.min_layers)
    print("Using search space:")
    print(search_space.print_search_space())

    arch = args.arch
    use_rdxe = (arch == "rDXE")
    use_surrogate = args.surrogate
    print(f"\nHardware architecture: {arch}")
    if use_rdxe:
        print(f"  rDXE ring: {args.n_chips} chips, {args.layer_assignment} assignment")
    print(f"SW evaluation: {'surrogate' if use_surrogate else 'real training'}")
    if args.prefill_len > 0 and args.decode_len > 0:
        print(f"HW eval mode: prefill {args.prefill_len} tokens + decode {args.decode_len} tokens")
    else:
        print(f"HW eval mode: prefill only (using block_size from search space)")

    # --- Load surrogate model if needed ---
    surrogate_model = None
    surrogate_norm = None
    surrogate_max_layers = None
    surrogate_device = None
    if use_surrogate:
        import torch
        from surrogate.inference import load_surrogate, surrogate_eval
        surrogate_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        surrogate_model, surrogate_norm, surrogate_max_layers = load_surrogate(
            checkpoint_path=surrogate_ckpt,
            device=surrogate_device,
        )
        print(f"Loaded surrogate model from {surrogate_ckpt} (max_layers={surrogate_max_layers})")
    else:
        hosts = load_hosts_from_file(args.hosts_file)
        logging.info(f"Loaded {len(hosts)} hosts from {args.hosts_file}")

    # --- Objectives & constraints ---
    objs = args.objectives
    if not args.constraint:
        cons = {
            "params": args.max_params,
            "val_loss": args.max_val_loss,
        }
    else:
        cons = {}
        for key, value in args.constraint:
            cons[key] = value

    exp_name = args.exp_name
    init_population_size = args.pop_size

    # --- Initialize or resume population ---
    if args.resume_ckpt is not None:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_ckpt}")
        logging.info(f"Resuming from checkpoint: {args.resume_ckpt}")
        population = Population.load_checkpoint(args.resume_ckpt,
                                                 from_pkl=args.resume_ckpt.endswith('.pkl'))
        population.search_space = search_space
        population.objs_settings = objs
        population.cons_settings = cons
        population.print_summary()
    else:
        if args.init_individuals:
            init_path = args.init_individuals
            if not os.path.isabs(init_path):
                init_path = os.path.join(script_dir, init_path)
            logging.info(f"Loading initial individuals from: {init_path}")
            individuals = load_initial_individuals(init_path)
            init_population_size = len(individuals)
        else:
            individuals = [search_space.sample() for _ in range(init_population_size)]

        population = Population(individuals, search_space=search_space,
                                objs_settings=objs, cons_settings=cons)
        population.delete_duplicates()

        # --- Initial evaluation ---
        if use_surrogate:
            pred_loss = surrogate_eval(
                individuals=population.individuals,
                model=surrogate_model,
                norm=surrogate_norm,
                device=surrogate_device,
                max_layers=surrogate_max_layers,
            )
            if use_rdxe:
                hw_data = run_rdxe_eval(population, args.n_chips, args.layer_assignment, args.prefill_len, args.decode_len, n_users=args.n_users, ctx=args.ctx_len, area_max_mm2=args.rdxe_area_max_mm2, area_min_mm2=args.rdxe_area_min_mm2, power_max_W=args.rdxe_power_max_W, power_min_W=args.rdxe_power_min_W, envelope_filter=args.rdxe_envelope_filter, verbose=args.rdxe_verbose)
            else:
                hw_data = run_hw_eval(population, arch, args.prefill_len, args.decode_len)
            population.apply_surrogate_and_hw(pred_loss, hw_data)
        else:
            if use_rdxe:
                hw_data = run_rdxe_eval(population, args.n_chips, args.layer_assignment, args.prefill_len, args.decode_len, n_users=args.n_users, ctx=args.ctx_len, area_max_mm2=args.rdxe_area_max_mm2, area_min_mm2=args.rdxe_area_min_mm2, power_max_W=args.rdxe_power_max_W, power_min_W=args.rdxe_power_min_W, envelope_filter=args.rdxe_envelope_filter, verbose=args.rdxe_verbose)
                population.sw_eval(
                    hosts=hosts, user=args.user, key_filename=args.key,
                    run_dir_name=exp_name, conda_env=args.conda_env,
                    max_iters=args.max_iters, dataset=args.dataset,
                    sw_only=True, timeout=args.timeout,
                    arch_list=[], prefill_len=0, decode_len=0,
                )
                # merge rDXE hw_data into evaluations
                evals = population.evaluations if population.gen == 0 else population.offspring_evaluations
                for ev, hw in zip(evals, hw_data):
                    ev.aux.update(hw)
                    # recompute objs/cons with hw metrics now present
                    ev.objs = [float(ev.aux.get(o, float("inf"))) for o in population.objs_settings]
                    ev.cons = [cons_value(c, population.cons_settings[c], ev.aux) for c in population.cons_settings]
            else:
                population.sw_eval(
                    hosts=hosts, user=args.user, key_filename=args.key,
                    run_dir_name=exp_name, conda_env=args.conda_env,
                    max_iters=args.max_iters, dataset=args.dataset,
                    sw_only=False, timeout=args.timeout,
                    arch_list=[arch],
                    prefill_len=args.prefill_len, decode_len=args.decode_len,
                )
        population.print_summary()

    # --- NSGA-II parameters ---
    population.n_population = init_population_size
    population.n_offspring = args.offspring
    population.crossover_rate = args.crossover_rate
    population.mutation_rate = args.mutation_rate

    # --- Save initial checkpoint ---
    run_time = time.strftime("%m%d_%H%M", time.localtime())
    if args.resume_ckpt is None:
        os.makedirs(f"ckpts/{exp_name}", exist_ok=True)
        os.makedirs(f"ckpts/{exp_name}/pkl", exist_ok=True)
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{population.gen}.json")
        population.save_checkpoint_pkl(f"ckpts/{exp_name}/pkl/{run_time}_pop_gen{population.gen}.pkl")

    # --- Git pull on remote hosts (real training only) ---
    if not use_surrogate:
        trainer = RemoteTrainer(hosts=hosts, user=args.user, key_filename=args.key)
        trainer.perform_git_pull(remote_work_dir=os.environ.get("EVO_GPT_DIR", os.path.expanduser("~/evo_gpt")))

    # --- NSGA-II generation loop ---
    n_gen = args.generations
    for i in range(n_gen):
        population.generate_offspring()
        gen = population.gen
        print(f"\n\n================ Generation {gen} ================\n")

        if use_surrogate:
            pred_loss = surrogate_eval(
                individuals=population.offspring,
                model=surrogate_model,
                norm=surrogate_norm,
                device=surrogate_device,
                max_layers=surrogate_max_layers,
            )
            if use_rdxe:
                hw_data = run_rdxe_eval(population, args.n_chips, args.layer_assignment, args.prefill_len, args.decode_len, n_users=args.n_users, ctx=args.ctx_len, area_max_mm2=args.rdxe_area_max_mm2, area_min_mm2=args.rdxe_area_min_mm2, power_max_W=args.rdxe_power_max_W, power_min_W=args.rdxe_power_min_W, envelope_filter=args.rdxe_envelope_filter, verbose=args.rdxe_verbose)
            else:
                hw_data = run_hw_eval(population, arch, args.prefill_len, args.decode_len)
            population.apply_surrogate_and_hw(pred_loss, hw_data)
        else:
            if use_rdxe:
                hw_data = run_rdxe_eval(population, args.n_chips, args.layer_assignment, args.prefill_len, args.decode_len, n_users=args.n_users, ctx=args.ctx_len, area_max_mm2=args.rdxe_area_max_mm2, area_min_mm2=args.rdxe_area_min_mm2, power_max_W=args.rdxe_power_max_W, power_min_W=args.rdxe_power_min_W, envelope_filter=args.rdxe_envelope_filter, verbose=args.rdxe_verbose)
                population.sw_eval(
                    hosts=hosts, user=args.user, key_filename=args.key,
                    run_dir_name=exp_name, conda_env=args.conda_env,
                    max_iters=args.max_iters, dataset=args.dataset,
                    sw_only=True, timeout=args.timeout,
                    arch_list=[], prefill_len=0, decode_len=0,
                )
                evals = population.offspring_evaluations
                for ev, hw in zip(evals, hw_data):
                    ev.aux.update(hw)
                    ev.objs = [float(ev.aux.get(o, float("inf"))) for o in population.objs_settings]
                    ev.cons = [cons_value(c, population.cons_settings[c], ev.aux) for c in population.cons_settings]
            else:
                population.sw_eval(
                    hosts=hosts, user=args.user, key_filename=args.key,
                    run_dir_name=exp_name, conda_env=args.conda_env,
                    max_iters=args.max_iters, dataset=args.dataset,
                    sw_only=False, timeout=args.timeout,
                    arch_list=[arch],
                    prefill_len=args.prefill_len, decode_len=args.decode_len,
                )

        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_offspring_gen{gen}.json")
        population.update_elimination()
        population.print_summary()
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{gen}.json")
        population.save_checkpoint_pkl(f"ckpts/{exp_name}/pkl/{run_time}_pop_gen{gen}.pkl")


if __name__ == "__main__":
    main()
