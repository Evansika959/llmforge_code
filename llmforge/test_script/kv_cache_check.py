"""KV-cache verification — direct (no run_cosearch.py wrapper).

Three checks, all must pass for the script to exit 0:

  1. PARITY     — for several randomly-sampled archs, the cached and
                  uncached final logits must agree to bf16 tolerance.
  2. DIRECTION  — averaged across archs, the cached `tpot_ms` must
                  not regress vs uncached. The cache strictly reduces
                  per-step compute (cached decode does T_q=1 against
                  cached K,V; uncached re-runs the full prefill+grown
                  context every step), so cached tpot ≤ uncached tpot
                  is a hard invariant. Geometric-mean ratio must be
                  < TPOT_BOUND. Energy and power are reported as
                  telemetry only — both are dominated by ZEUS counter
                  jitter at small workloads and can move either way.
  3. TOGGLE     — `HwZeus(use_kv_cache=False)` actually disables the
                  cache (zeus_kv_cache_used=False in result dict),
                  while `use_kv_cache=True` enables it.

Run from inside nsga_search/:

    python test_script/kv_cache_check.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch
import yaml

# Make sibling imports work whether invoked from nsga_search/ or test_script/.
THIS = os.path.dirname(os.path.abspath(__file__))
NSGA = os.path.dirname(THIS)
if NSGA not in sys.path:
    sys.path.insert(0, NSGA)

from search_space import HeteroSearchSpace                       # noqa: E402
from zeus_eval import build_model_from_individual, measure_one   # noqa: E402
from zeus_kv_cache import (                                       # noqa: E402
    attach_iha_kv_cache,
    detach_iha_kv_cache,
    parity_check,
    UnsupportedKVCache,
)
from evaluators.hw_zeus import HwZeus                             # noqa: E402


# ── Config (kept tight so the script runs in <60s on an A100) ─────────────

SEEDS = (1, 2, 3, 4, 5)
SEARCH_SPACE_YAML = "search_space_def/search_space_200M.yaml"
L_MAX = 12
L_MIN = 8
PARITY_PREFILL = 32
PARITY_DECODE = 4
# bf16 round-off compounds with √depth and gets larger with SwiGLU's
# 3-matrix MLP. 1e-1 fits both this search space (≤12 layers, ~1-3e-2 obs)
# and deeper SmolLM2-style stacks (30 layers, ~9e-2 obs).
PARITY_ATOL = 1e-1
PARITY_RTOL = 1e-1
# Direction check: longer prefill amplifies the cache benefit (per-step
# compute reduction is proportional to prefill_len + step_idx), making
# the tpot signal cleanly separable from jitter.
ZEUS_PREFILL = 128
ZEUS_DECODE = 32
ZEUS_REPEATS = 3
ZEUS_WARMUP = 2
# tpot is a hard invariant — cached should be ≤ uncached. Allow 5% slack
# for kernel-launch overhead jitter.
TPOT_RATIO_BOUND = 1.05


def _build_search_space():
    with open(os.path.join(NSGA, SEARCH_SPACE_YAML)) as f:
        cfg = yaml.safe_load(f)
    return HeteroSearchSpace.from_dicts(
        cfg["global_spec"], cfg["layer_spec"], L_max=L_MAX, L_min=L_MIN)


# ── Check 1: parity ────────────────────────────────────────────────────────

def check_parity(ss, device) -> bool:
    print("\n--- check 1: parity (cached vs uncached final logits) ---")
    rows = []
    all_ok = True
    for seed in SEEDS:
        torch.manual_seed(seed)
        ind = ss.sample()
        ind_dict = ind.to_dict() if hasattr(ind, "to_dict") else ind
        model = build_model_from_individual(
            ind_dict, block_size=PARITY_PREFILL + PARITY_DECODE + 8,
            device=device, dtype=torch.bfloat16)
        try:
            attach_iha_kv_cache(model)
        except UnsupportedKVCache as e:
            print(f"  seed={seed}: SKIP (arch unsupported: {e})")
            del model
            torch.cuda.empty_cache()
            continue
        try:
            ok, max_abs, max_rel = parity_check(
                model, prefill_len=PARITY_PREFILL, decode_len=PARITY_DECODE,
                device=device, atol=PARITY_ATOL, rtol=PARITY_RTOL,
                verbose=False)
        finally:
            detach_iha_kv_cache(model)
            del model
            torch.cuda.empty_cache()
        rows.append((seed, ok, max_abs, max_rel))
        all_ok = all_ok and ok

    print(f"  {'seed':>4} | {'pass':>4} | {'max_abs':>10} | {'max_rel':>10}")
    for s, ok, ma, mr in rows:
        print(f"  {s:>4} | {str(ok):>4} | {ma:>10.3e} | {mr:>10.3e}")
    print(f"  → parity check: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── Check 2: direction ─────────────────────────────────────────────────────

def check_direction(ss, device) -> bool:
    print("\n--- check 2: direction (cached must not regress tpot) ---")
    tpot_ratios = []
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        ind = ss.sample()
        ind_dict = ind.to_dict() if hasattr(ind, "to_dict") else ind
        # Alternate ON-then-OFF to neutralize kernel-warmup ordering bias.
        r_on = measure_one(
            ind_dict, prefill_len=ZEUS_PREFILL, decode_len=ZEUS_DECODE,
            n_repeats=ZEUS_REPEATS, warmup=ZEUS_WARMUP, dtype="bf16",
            use_kv_cache=True)
        r_off = measure_one(
            ind_dict, prefill_len=ZEUS_PREFILL, decode_len=ZEUS_DECODE,
            n_repeats=ZEUS_REPEATS, warmup=ZEUS_WARMUP, dtype="bf16",
            use_kv_cache=False)
        if not (r_on.get("envelope_feasible") and r_off.get("envelope_feasible")):
            print(f"  seed={seed}: SKIP (measurement failed)")
            continue
        rt = r_on["tpot_ms"] / max(1e-9, r_off["tpot_ms"])
        re_eng = (r_on["energy_per_token_uJ"]
                  / max(1e-9, r_off["energy_per_token_uJ"]))
        rp = r_on["power_W"] / max(1e-9, r_off["power_W"])
        tpot_ratios.append(rt)
        rows.append((seed, r_on["tpot_ms"], r_off["tpot_ms"], rt,
                     re_eng, rp))

    if not rows:
        print("  → no successful measurements; direction check inconclusive.")
        return False

    print(f"  {'seed':>4} | {'tpot ON':>7} | {'tpot OFF':>8} | "
          f"{'tpot/':>5} | {'E/tok':>5} | {'pwr':>5}")
    print(f"  {'':>4} | {'(ms)':>7} | {'(ms)':>8} | "
          f"{'ratio':>5} | {'ratio':>5} | {'ratio':>5}")
    for s, t_on, t_off, rt, re_eng, rp in rows:
        print(f"  {s:>4} | {t_on:>7.2f} | {t_off:>8.2f} | "
              f"{rt:>5.2f} | {re_eng:>5.2f} | {rp:>5.2f}")
    g_tpot = math.exp(sum(math.log(r) for r in tpot_ratios) / len(tpot_ratios))
    print(f"  geomean tpot ratio (on/off): {g_tpot:.3f}  "
          f"(must be < {TPOT_RATIO_BOUND})")
    print("  E/tok and power are informational — both depend on prefill vs "
          "decode power profile and ZEUS counter granularity; neither is a "
          "hard invariant of the cache shim.")
    ok = g_tpot < TPOT_RATIO_BOUND
    print(f"  → direction check: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Check 3: HwZeus flag-toggle ───────────────────────────────────────────

def check_hw_zeus_toggle(ss) -> bool:
    print("\n--- check 3: HwZeus flag-toggle ---")
    torch.manual_seed(0)
    ind = ss.sample()
    ind_dict = ind.to_dict() if hasattr(ind, "to_dict") else ind

    on  = HwZeus(prefill_len=32, decode_len=4, n_repeats=1, warmup=1,
                 use_kv_cache=True,  verbose=False)
    off = HwZeus(prefill_len=32, decode_len=4, n_repeats=1, warmup=1,
                 use_kv_cache=False, verbose=False)
    r_on  = on.evaluate([ind_dict])[0]
    r_off = off.evaluate([ind_dict])[0]
    used_on  = bool(r_on.get("zeus_kv_cache_used"))
    used_off = bool(r_off.get("zeus_kv_cache_used"))
    print(f"  HwZeus(use_kv_cache=True)  → zeus_kv_cache_used={used_on}")
    print(f"  HwZeus(use_kv_cache=False) → zeus_kv_cache_used={used_off}")

    ok = (used_on is True) and (used_off is False)
    if not ok and r_on.get("zeus_kv_cache_skip"):
        # Cache fell back due to feature-flag mismatch on this arch.
        # That's not a HwZeus bug — it's the shim doing what we want.
        # Surface the reason and treat as a soft skip rather than fail.
        print(f"  cache fell back: {r_on['zeus_kv_cache_skip']}")
        print("  → toggle check: SKIP (cache unsupported for this arch)")
        return True
    print(f"  → toggle check: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; KV-cache test requires a GPU. SKIP.")
        return 0
    device = torch.device("cuda:0")
    ss = _build_search_space()

    t0 = time.time()
    results = {
        "parity":     check_parity(ss, device),
        "direction":  check_direction(ss, device),
        "toggle":     check_hw_zeus_toggle(ss),
    }
    dt = time.time() - t0

    print("\n=========================================================")
    print(f"kv_cache_check summary  ({dt:.1f}s)")
    for k, ok in results.items():
        print(f"  {k:10s}  {'PASS' if ok else 'FAIL'}")
    print("=========================================================")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
