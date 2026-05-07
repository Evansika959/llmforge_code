"""Multi-platform HW evaluation runner.

Given a reference arch YAML, runs one or more HW evaluators (ZEUS, Timeloop
substrates, rDXE inner ring), collects ttft / tpot / energy_per_token across
all chosen platforms, and writes a single JSON "checkpoint" alongside a
human-readable comparison table.

Reuses the existing evaluator classes verbatim:
  - evaluators.hw_zeus.HwZeus           (local A100 measurement)
  - evaluators.hw_timeloop.HwTimeloop   (eyeriss / simba / gemmini /
                                         flat_edge / dxe / dxe_relaxed)
  - evaluators.hw_rdxe_inner.HwRdxeInner (rDXE ring sweep)
  - evaluators.hw_none.HwNone           (always-on analytical aux)

Output JSON schema (per-arch):
  {
    "name": "...",
    "globals": {...},
    "n_active_layers": int,
    "params_M_est": float,
    "config": {prefill_len, decode_len, seq_len, dtype, no_kv_cache},
    "platforms_run": [...],
    "headline": {                     # cross-platform comparison row keys
        "<platform>": {ttft_ms, tpot_ms, energy_per_token_uJ, power_W?},
        ...
    },
    "analytical": {params_M, flops_per_token, kv_cache_bytes, kv_cache_MB},
    "hw": {                           # full per-platform aux dict
        "<platform>": {...full evaluator output...},
        ...
    },
    "errors": {                       # platforms that raised
        "<platform>": "<traceback>",
        ...
    }
  }

Available platforms:
  zeus
  timeloop_eyeriss timeloop_simba timeloop_gemmini timeloop_flat_edge
  timeloop_dxe_relaxed
  rdxe

  (timeloop_dxe — strict mapper — is intentionally not exposed: many
  search-space GEMM shapes can't be tiled into the rigid PE array, so the
  mapper exhausts its candidate set and raises. Use timeloop_dxe_relaxed
  for the same chip with loosened mapper constraints — that's what the
  rDXE inner search uses, and what we expose here.)

Usage (single arch, ZEUS only):
  python bench_hw_eval.py --ref_yaml reference_archs/smollm2_135m.yaml \\
      --exp_name smollm2_135m_hw --out_json out.json --hw zeus

Usage (everything, takes a while):
  python bench_hw_eval.py --ref_yaml reference_archs/smollm2_135m.yaml \\
      --exp_name smollm2_135m_hw_all --out_json out.json --hw all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger("bench_hw_eval")


# ── Platform registry ─────────────────────────────────────────────────────

# Order matters for `--hw all`: cheap first so failures land early.
# timeloop_dxe (strict mapper) is intentionally excluded — its rigid
# mapper constraints reject most search-space GEMM shapes, so it's not a
# useful default. timeloop_dxe_relaxed covers the same chip.
ALL_PLATFORMS = (
    "zeus",
    "timeloop_eyeriss",
    "timeloop_simba",
    "timeloop_gemmini",
    "timeloop_flat_edge",
    "timeloop_dxe_relaxed",
    "rdxe",
)


def parse_platforms(arg: str) -> List[str]:
    """Comma-separated list, plus the special tokens `all` and `timeloop_all`."""
    if arg.lower() == "all":
        return list(ALL_PLATFORMS)
    if arg.lower() == "timeloop_all":
        return [p for p in ALL_PLATFORMS if p.startswith("timeloop_")]
    out = []
    for tok in arg.split(","):
        t = tok.strip()
        if not t:
            continue
        if t not in ALL_PLATFORMS:
            raise SystemExit(
                f"[args] unknown platform: {t!r}. "
                f"valid: {', '.join(ALL_PLATFORMS)} | all | timeloop_all"
            )
        out.append(t)
    if not out:
        raise SystemExit("[args] --hw must list at least one platform")
    return out


# ── Reference loader (reused from bench_smollm2_baseline) ─────────────────

def reference_to_ind_dict(ref_yaml_path: str) -> Dict[str, Any]:
    """Load reference YAML → {globals, layers} dict, all layers active.

    Honors an optional top-level `mlp_variant` field in the YAML (e.g.
    Pythia tags itself `mlp_variant: mlp` because the published model
    uses 2-matrix GeLU; SmolLM2 may omit the field and inherit
    `build_model_from_individual`'s SwiGLU default). When present, the
    field is stashed on `globals.mlp_variant` so the build picks it up
    without needing a CLI flag."""
    from init.seed_arch import load_reference_yaml
    import yaml as _yaml
    ref = load_reference_yaml(ref_yaml_path)
    layers = [dict(li) for li in ref["layers"]]
    globals_ = {
        "n_embd": ref["n_embd"],
        "block_size": ref["block_size"],
        "use_concat_heads": ref["use_concat_heads"],
        "layer_mask": [True] * len(layers),
    }
    with open(ref_yaml_path) as f:
        raw = _yaml.safe_load(f)
    if isinstance(raw, dict) and "mlp_variant" in raw:
        globals_["mlp_variant"] = str(raw["mlp_variant"])
    return {"globals": globals_, "layers": layers, "name": ref.get("name")}


# ── Per-platform dispatch ─────────────────────────────────────────────────

def evaluate_platform(name: str, ind_dict: Dict[str, Any], args) -> Dict[str, Any]:
    """Run a single platform's evaluator on `ind_dict`. Returns its raw
    result dict (the [0] element of the evaluator's per-arch list)."""
    if name == "zeus":
        from evaluators.hw_zeus import HwZeus
        ev = HwZeus(
            prefill_len=args.prefill_len, decode_len=args.decode_len,
            n_repeats=args.zeus_n_repeats, warmup=args.zeus_warmup,
            dtype=args.zeus_dtype, verbose=args.verbose,
            use_kv_cache=not args.no_kv_cache,
        )
        return ev.evaluate([ind_dict])[0]

    if name.startswith("timeloop_"):
        substrate = name[len("timeloop_"):]
        from evaluators.hw_timeloop import HwTimeloop, SUBSTRATE_MAP
        if substrate not in SUBSTRATE_MAP:
            raise ValueError(f"Unknown timeloop substrate: {substrate}; "
                             f"valid: {', '.join(SUBSTRATE_MAP)}")
        ev = HwTimeloop(substrate=substrate,
                         prefill_len=args.prefill_len,
                         decode_len=args.decode_len)
        return ev.evaluate([ind_dict])[0]

    if name == "rdxe":
        from evaluators.hw_rdxe_inner import HwRdxeInner
        ev = HwRdxeInner(
            prefill_len=args.prefill_len,
            decode_len=args.decode_len,
            n_users=args.rdxe_n_users,
            ctx=args.rdxe_ctx,
            select_by=args.rdxe_select_by,
            envelope_filter=not args.rdxe_no_envelope_filter,
            area_max_mm2=args.rdxe_area_max,
            area_min_mm2=args.rdxe_area_min,
            power_max_W=args.rdxe_power_max,
            power_min_W=args.rdxe_power_min,
            verbose=args.verbose,
        )
        return ev.evaluate([ind_dict])[0]

    raise ValueError(f"Unknown platform: {name!r}")


# ── Headline normalization ────────────────────────────────────────────────

def _to_float(v: Any) -> Optional[float]:
    """Coerce to finite float or return None for inf / NaN / non-numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f == float("inf") or f == float("-inf"):
        return None
    return f


def normalize_headline(name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return {ttft_ms, tpot_ms, energy_per_token_uJ, power_W} when present.

    Each evaluator emits its own field names; this funnels them into a
    single comparable schema:
      - ZEUS: emits ttft_ms / tpot_ms / energy_per_token_uJ / power_W directly.
      - Timeloop: emits ttft, tpot in seconds (cycles/1e9). Convert to ms.
        Emits energy_per_token_uJ directly. No power_W (analytical).
      - rDXE: emits both ttft_ms and ttft (seconds), ditto tpot. Plus power_W.
    """
    out: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out

    # ttft (ms)
    v = raw.get("ttft_ms")
    if v is None:
        v = raw.get("ttft")
        if isinstance(v, (int, float)) and v not in (float("inf"), float("-inf")) and v == v:
            v = v * 1e3
    fv = _to_float(v) if v is not None else None
    if fv is not None:
        out["ttft_ms"] = fv

    # tpot (ms)
    v = raw.get("tpot_ms")
    if v is None:
        v = raw.get("tpot")
        if isinstance(v, (int, float)) and v not in (float("inf"), float("-inf")) and v == v:
            v = v * 1e3
    fv = _to_float(v) if v is not None else None
    if fv is not None:
        out["tpot_ms"] = fv

    # energy per token (uJ)
    fv = _to_float(raw.get("energy_per_token_uJ"))
    if fv is not None:
        out["energy_per_token_uJ"] = fv

    # power W (only zeus + rdxe; timeloop omits)
    fv = _to_float(raw.get("power_W"))
    if fv is not None:
        out["power_W"] = fv

    return out


# ── Param-count estimate (reused from bench_smollm2_baseline) ─────────────

def estimate_params_M(ind_dict: Dict[str, Any]) -> float:
    """SwiGLU-aware estimate; matches the build defaults."""
    g = ind_dict["globals"]
    n_embd = int(g["n_embd"])
    layers = ind_dict["layers"]
    mask = g.get("layer_mask", [True] * len(layers))
    total = 0
    for L, m in zip(layers, mask):
        if not m:
            continue
        n_head = int(L.get("n_head", 8))
        n_kv = int(L.get("n_kv_group", n_head))
        qk_d = int(L.get("n_qk_head_dim", n_embd // max(1, n_head)))
        v_d  = int(L.get("n_v_head_dim", qk_d))
        mlp  = int(L.get("mlp_size", 4 * n_embd))
        total += n_embd * (n_head * qk_d)
        total += n_embd * (n_kv * qk_d)
        total += n_embd * (n_kv * v_d)
        total += (n_head * v_d) * n_embd if g.get("use_concat_heads", True) else v_d * n_embd
        total += 3 * n_embd * mlp        # SwiGLU
    return total / 1e6


# ── Pretty-print + JSON serialization ─────────────────────────────────────

def _fmt(v: Any, ndp: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v or v == float("inf") or v == float("-inf"):
            return "—"
        if abs(v) >= 1e6:
            return f"{v/1e6:.{ndp}f}M"
        if abs(v) >= 1e3:
            return f"{v:,.{ndp}f}"
        return f"{v:.{ndp}f}"
    return str(v)


def print_summary(ind_dict: Dict[str, Any], headline: Dict[str, Dict[str, Any]],
                  errors: Dict[str, str], args) -> None:
    print()
    print("=" * 88)
    print(f"MULTI-PLATFORM HW EVAL — {ind_dict.get('name', 'arch')}")
    print("=" * 88)
    g = ind_dict["globals"]
    n_active = sum(1 for m in g.get("layer_mask", []) if m)
    print(f"  arch        : n_embd={g['n_embd']}, block_size={g['block_size']}, "
          f"n_layer={n_active}, est params={estimate_params_M(ind_dict):.2f} M")
    print(f"  workload    : prefill={args.prefill_len}, decode={args.decode_len}, "
          f"seq_len={args.seq_len}, dtype={args.zeus_dtype}, kv_cache="
          f"{'OFF' if args.no_kv_cache else 'ON'}")
    print()
    cols = ("ttft_ms", "tpot_ms", "energy_per_token_uJ", "power_W")
    print(f"  {'platform':24s} | {'ttft_ms':>10s} | {'tpot_ms':>10s} | "
          f"{'E/tok_uJ':>12s} | {'power_W':>9s}")
    print("  " + "-" * 86)
    for plat in args.hw:
        if plat in errors:
            print(f"  {plat:24s} | {'ERROR':>10s} | "
                  f"{errors[plat][:50]:>10s}")
            continue
        h = headline.get(plat, {})
        print(f"  {plat:24s} | "
              f"{_fmt(h.get('ttft_ms')):>10s} | "
              f"{_fmt(h.get('tpot_ms')):>10s} | "
              f"{_fmt(h.get('energy_per_token_uJ'), 3):>12s} | "
              f"{_fmt(h.get('power_W'), 1):>9s}")
    print()


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch / inf / NaN into JSON-serializable
    Python primitives. Matches what NSGA's save_checkpoint does."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    # numpy scalars / arrays
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        if isinstance(obj, (np.floating,)):
            return _json_safe(float(obj))
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except ImportError:
        pass
    try:
        if isinstance(obj, torch.Tensor):
            return _json_safe(obj.detach().cpu().tolist())
    except Exception:
        pass
    return str(obj)                                  # last resort


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Reference + identity
    p.add_argument("--ref_yaml", type=str, required=True)
    p.add_argument("--exp_name", type=str, required=True,
                   help="Used to namespace per-platform work dirs (e.g. "
                        "Timeloop's hw_eval/runs/<arch>/{prefill,decode}/) "
                        "and to label the output JSON.")
    p.add_argument("--out_json", type=str, required=True,
                   help="Output JSON path. Parent dir created if missing.")
    p.add_argument("--verbose", action="store_true")

    # Platform selection
    p.add_argument("--hw", type=parse_platforms, default=parse_platforms("zeus"),
                   metavar="LIST",
                   help=f"Comma-separated platforms or 'all' / 'timeloop_all'. "
                        f"Available: {', '.join(ALL_PLATFORMS)}. Default: zeus.")

    # Workload (shared across platforms)
    p.add_argument("--prefill_len", type=int, default=256)
    p.add_argument("--decode_len", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=512,
                   help="Used by HwNone (analytical kv_cache_bytes / "
                        "flops_per_token). Doesn't affect ZEUS/Timeloop "
                        "directly — those use prefill_len + decode_len.")

    # ZEUS knobs
    p.add_argument("--zeus_n_repeats", type=int, default=3)
    p.add_argument("--zeus_warmup", type=int, default=2)
    p.add_argument("--zeus_dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no_kv_cache", action="store_true",
                   help="Disable the measurement-only KV-cache shim.")

    # rDXE knobs
    p.add_argument("--rdxe_n_users", type=int, default=1)
    p.add_argument("--rdxe_ctx", type=int, default=None)
    p.add_argument("--rdxe_select_by", type=str, default="per_tok_uJ",
                   choices=["per_tok_uJ", "tpot_ms", "ttft_ms"])
    p.add_argument("--rdxe_no_envelope_filter", action="store_true")
    p.add_argument("--rdxe_area_max", type=float, default=800.0)
    p.add_argument("--rdxe_area_min", type=float, default=0.0)
    p.add_argument("--rdxe_power_max", type=float, default=100.0)
    p.add_argument("--rdxe_power_min", type=float, default=0.0)

    args = p.parse_args()
    print(f"Platforms to run: {args.hw}")

    ref_path = (args.ref_yaml if os.path.isabs(args.ref_yaml)
                else os.path.join(SCRIPT_DIR, args.ref_yaml))
    ind_dict = reference_to_ind_dict(ref_path)
    print(f"Loaded reference: {ind_dict.get('name', '?')}  "
          f"({sum(1 for m in ind_dict['globals']['layer_mask'] if m)} layers, "
          f"~{estimate_params_M(ind_dict):.1f} M est params)")

    # Always run the analytical baseline.
    from evaluators.hw_none import HwNone
    analytical = HwNone(seq_len=args.seq_len).evaluate([ind_dict])[0]

    # Per-platform sequential dispatch.
    hw: Dict[str, Dict[str, Any]] = {}
    headline: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}

    for plat in args.hw:
        if plat == "zeus" and not torch.cuda.is_available():
            errors[plat] = "CUDA not available; ZEUS requires a GPU."
            log.warning(errors[plat])
            continue
        log.info(f"───── platform: {plat} ─────")
        t0 = time.time()
        try:
            raw = evaluate_platform(plat, ind_dict, args)
            timings[plat] = time.time() - t0
            hw[plat] = raw
            headline[plat] = normalize_headline(plat, raw)
            log.info(f"  {plat} done in {timings[plat]:.1f}s")
        except Exception:
            timings[plat] = time.time() - t0
            tb = traceback.format_exc(limit=4)
            errors[plat] = tb
            log.warning(f"  {plat} FAILED in {timings[plat]:.1f}s:\n{tb}")
            # Keep going — one platform's failure shouldn't kill the rest.

    # Assemble output.
    out = {
        "name": ind_dict.get("name"),
        "exp_name": args.exp_name,
        "ref_yaml": ref_path,
        "globals": ind_dict["globals"],
        "n_active_layers": sum(1 for m in ind_dict["globals"]["layer_mask"] if m),
        "params_M_est": estimate_params_M(ind_dict),
        "config": {
            "prefill_len": args.prefill_len,
            "decode_len": args.decode_len,
            "seq_len": args.seq_len,
            "zeus_dtype": args.zeus_dtype,
            "no_kv_cache": args.no_kv_cache,
        },
        "platforms_run": list(args.hw),
        "platform_timings_s": timings,
        "headline": headline,
        "analytical": analytical,
        "hw": hw,
        "errors": errors,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(_json_safe(out), f, indent=2)

    print_summary(ind_dict, headline, errors, args)
    print(f"  Saved: {args.out_json}")
    if errors:
        print(f"  ⚠ {len(errors)}/{len(args.hw)} platform(s) failed: "
              f"{', '.join(errors.keys())}")


if __name__ == "__main__":
    main()
