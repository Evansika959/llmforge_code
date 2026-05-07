"""Local A100-based HW evaluator using ZEUS.

Given an NSGA Individual dict (`{"globals": ..., "layers": [...]}`), instantiate
the corresponding Evo_GPT model with random weights, run prefill + decode on
the local A100 wrapped in a ZeusMonitor window, and return ttft / tpot /
energy metrics matching the aux-dict schema consumed by NSGA.

No weight training and no remote hosts — the arch alone determines the
compute shapes, which is what ZEUS measures.

Expected metric shapes on an A100 (bf16, ~300M-param SmolLM2-scale arch):
  ttft_ms              ~20 – 80 ms   (single prefill forward)
  tpot_ms              ~15 – 60 ms   (naive AR decode — no KV cache in Evo_GPT)
  energy_per_token_uJ  ~1000 – 4000 μJ
  power_W              ~100 – 300 W
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

# Make Evo_GPT importable
_EVO_GPT = "${EVO_GPT_DIR:-$HOME/evo_gpt}"
if _EVO_GPT not in sys.path:
    sys.path.insert(0, _EVO_GPT)

try:
    from zeus.monitor import ZeusMonitor
except ImportError as e:
    raise ImportError("Install zeus: `pip install zeus`") from e

from gpt_conf import GPTConfig  # noqa: E402
from model import GPT           # noqa: E402

from zeus_kv_cache import (
    attach_iha_kv_cache,
    detach_iha_kv_cache,
    set_iha_mode,
    clear_iha_kv_state,
    run_cached_decode,
    UnsupportedKVCache,
)

log = logging.getLogger(__name__)


def _active_layers(ind: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return only the layers where layer_mask is True."""
    g = ind.get("globals", ind)
    mask = g.get("layer_mask", None)
    layers = ind.get("layers", [])
    if not mask:
        return list(layers)
    return [L for L, m in zip(layers, mask) if m]


def build_model_from_individual(ind: Dict[str, Any], block_size: int,
                                 device: torch.device, dtype: torch.dtype,
                                 mlp_variant: str = "swiglu") -> GPT:
    """Instantiate an Evo_GPT model from an Individual dict, random weights.

    `mlp_variant` defaults to "swiglu" to match `remote_trainer.py`'s
    `_FIXED_MODEL_OVERRIDES`, which pin `mlp_variant=swiglu` on every remote
    training launch. Keeping the measurement build aligned with the training
    build means ZEUS HW metrics correspond to the same model that produces
    `val_loss` — this matters because SwiGLU's gated 3-matrix MLP has ~50%
    more MLP parameters and FLOPs than the 2-matrix GeLU `OriginalMLP`. Pass
    `mlp_variant="mlp"` explicitly to recover the 2-matrix build.

    A reference arch YAML may override the default per-arch by setting
    `mlp_variant` at the top level (e.g. Pythia-160M uses GeLU 2-matrix
    MLP and tags itself `mlp_variant: mlp`). When `globals["mlp_variant"]`
    is present on `ind`, it wins over the function argument.
    """
    g = ind.get("globals", ind)
    # Per-arch YAML override wins over the function-default mlp_variant.
    mlp_variant = g.get("mlp_variant", mlp_variant)
    active = _active_layers(ind)
    if not active:
        raise ValueError("Individual has no active layers")

    # Helper with safe default
    def _col(key: str, default):
        return [L.get(key, default) for L in active]

    n_head_ll = _col("n_head", 8)
    n_kv_ll   = _col("n_kv_group", 8)
    mlp_ll    = _col("mlp_size", 4 * int(g.get("n_embd", 768)))
    qk_ll     = _col("n_qk_head_dim", None)
    v_ll      = _col("n_v_head_dim", None)
    cproj_ll  = _col("n_cproj", 1)
    attn_ll   = _col("attention_variant", "infinite")
    # Evo_GPT's attention_dictionary recognizes "infinite" (IHA) and
    # "identity" directly — pass through unchanged.

    cfg_kwargs = dict(
        n_layer=len(active),
        n_embd=int(g.get("n_embd", 768)),
        block_size=int(block_size),
        vocab_size=50304,
        n_head=int(n_head_ll[0]),
        n_kv_group=int(n_kv_ll[0]),
        n_head_layerlist=[int(x) for x in n_head_ll],
        n_kv_group_layerlist=[int(x) for x in n_kv_ll],
        mlp_size_layerlist=[int(x) for x in mlp_ll],
        n_cproj_layerlist=[int(x) for x in cproj_ll],
        attention_variant_layerlist=attn_ll,
        mlp_variant=mlp_variant,
    )
    if any(x is not None for x in qk_ll):
        cfg_kwargs["n_qk_head_dim_layerlist"] = [int(x) for x in qk_ll if x is not None]
    if any(x is not None for x in v_ll):
        cfg_kwargs["n_v_head_dim_layerlist"] = [int(x) for x in v_ll if x is not None]

    cfg = GPTConfig(**cfg_kwargs)
    model = GPT(cfg)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def measure_one(ind: Dict[str, Any],
                prefill_len: int = 128,
                decode_len: int = 32,
                n_repeats: int = 1,
                warmup: int = 1,
                dtype: str = "bf16",
                monitor: Optional[ZeusMonitor] = None,
                use_kv_cache: bool = True) -> Dict[str, Any]:
    """Measure ttft/tpot/energy for one Individual on the local A100.

    `use_kv_cache=True` (default) routes decode through `zeus_kv_cache`,
    which gives KV-cached single-query SDPA per step — production-style.
    The supported feature subset (no rotary, plain softmax, flash on,
    no qk/v-norm, no flash-lobo) covers the search_space_200M default
    arches; if an arch enables an unsupported flag, we transparently
    fall back to non-cached `model.generate()` and surface this in the
    result dict via `zeus_kv_cache_used: False` plus `zeus_kv_cache_skip`.

    `use_kv_cache=False` reproduces the original behavior: Evo_GPT's
    `generate()` recomputes the full prefill+generated context every
    step. This is useful as an A/B baseline but inflates tpot / energy
    per token and biases power toward prefill-style compute-bound
    operation.
    """
    device = torch.device("cuda:0")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)

    if monitor is None:
        # approx_instant_energy smooths over NVML's ~10 ms counter update
        # period; short prefill windows otherwise report 0 J.
        monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[],
                              sync_execution_with="torch",
                              approx_instant_energy=True)

    try:
        model = build_model_from_individual(ind, block_size=prefill_len + decode_len + 8,
                                             device=device, dtype=torch_dtype)
    except Exception as e:
        log.warning(f"Model build failed: {e}")
        return _inf_result(str(e))

    # Try to enable KV cache. If the arch trips an unsupported feature flag,
    # we leave kv_cache_active=False and fall back to model.generate() below.
    kv_cache_active = False
    kv_cache_skip_reason: Optional[str] = None
    if use_kv_cache:
        try:
            attach_iha_kv_cache(model)
            kv_cache_active = True
        except UnsupportedKVCache as e:
            kv_cache_skip_reason = str(e)
            log.info(f"KV cache unsupported for this arch — falling back: {e}")

    try:
        bs = 1
        tokens = torch.randint(0, 50304, (bs, prefill_len), device=device, dtype=torch.long)

        # Warmup. With cache enabled, exercise both phases so the first
        # measured iteration has a stable kernel cache and ZEUS reading.
        for _ in range(max(0, warmup)):
            if kv_cache_active:
                clear_iha_kv_state(model)
                set_iha_mode(model, "capture")
                _ = model(tokens)
                set_iha_mode(model, "decode")
                _ = run_cached_decode(model, prefill_len=prefill_len,
                                      decode_len=decode_len, batch_size=bs,
                                      device=device)
                set_iha_mode(model, "off")
            else:
                _ = model(tokens)

        ttft_list, tpot_list, e_pre_list, e_dec_list, p_dec_list = [], [], [], [], []
        for _ in range(max(1, n_repeats)):
            if kv_cache_active:
                clear_iha_kv_state(model)
                set_iha_mode(model, "capture")
                torch.cuda.synchronize()
                monitor.begin_window("prefill")
                _ = model(tokens)
                torch.cuda.synchronize()
                m_pre = monitor.end_window("prefill")

                set_iha_mode(model, "decode")
                torch.cuda.synchronize()
                monitor.begin_window("decode")
                _ = run_cached_decode(model, prefill_len=prefill_len,
                                      decode_len=decode_len, batch_size=bs,
                                      device=device)
                torch.cuda.synchronize()
                m_dec = monitor.end_window("decode")
                set_iha_mode(model, "off")
            else:
                torch.cuda.synchronize()
                monitor.begin_window("prefill")
                _ = model(tokens)
                torch.cuda.synchronize()
                m_pre = monitor.end_window("prefill")

                monitor.begin_window("decode")
                _ = model.generate(tokens, max_new_tokens=decode_len,
                                   temperature=1.0, top_k=None)
                torch.cuda.synchronize()
                m_dec = monitor.end_window("decode")

            ttft_s = float(m_pre.time)
            decode_total_s = float(m_dec.time)
            tpot_s = decode_total_s / max(1, decode_len)
            e_pre_J = float(getattr(m_pre, "total_energy", 0.0))
            e_dec_J = float(getattr(m_dec, "total_energy", 0.0))
            p_dec_W = e_dec_J / max(1e-9, decode_total_s)

            ttft_list.append(ttft_s)
            tpot_list.append(tpot_s)
            e_pre_list.append(e_pre_J)
            e_dec_list.append(e_dec_J)
            p_dec_list.append(p_dec_W)

        # Take median to be robust to jitter
        ttft_s = _median(ttft_list)
        tpot_s = _median(tpot_list)
        e_pre_J = _median(e_pre_list)
        e_dec_J = _median(e_dec_list)
        p_dec_W = _median(p_dec_list)

        result = {
            "ttft": ttft_s,
            "tpot": tpot_s,
            "ttft_ms": ttft_s * 1e3,
            "tpot_ms": tpot_s * 1e3,
            "energy_per_token_uJ": (e_dec_J / max(1, decode_len)) * 1e6,
            "session_e_per_tok_uJ": ((e_pre_J + e_dec_J) / max(1, prefill_len + decode_len)) * 1e6,
            "power_W": p_dec_W,
            "envelope_feasible": True,
            "zeus_prefill_energy_J": e_pre_J,
            "zeus_decode_energy_J": e_dec_J,
            "zeus_decode_len": decode_len,
            "zeus_prefill_len": prefill_len,
            "zeus_dtype": dtype,
            "zeus_kv_cache_used": kv_cache_active,
            "zeus_kv_cache_skip": kv_cache_skip_reason,
        }
        return result
    except torch.cuda.OutOfMemoryError as e:
        log.warning(f"OOM on A100: {e}")
        return _inf_result(f"OOM: {e}")
    except Exception as e:
        log.warning(f"Measurement failed: {e}")
        return _inf_result(str(e))
    finally:
        if kv_cache_active:
            try:
                detach_iha_kv_cache(model)
            except Exception:
                pass
        del model
        torch.cuda.empty_cache()


def _inf_result(reason: str) -> Dict[str, Any]:
    inf = float("inf")
    return {
        "ttft": inf, "tpot": inf, "ttft_ms": inf, "tpot_ms": inf,
        "energy_per_token_uJ": inf, "session_e_per_tok_uJ": inf,
        "power_W": inf,
        "envelope_feasible": False, "zeus_error": reason,
    }


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def run_zeus_eval(individuals: List[Dict[str, Any]],
                  prefill_len: int = 128,
                  decode_len: int = 32,
                  n_repeats: int = 1,
                  warmup: int = 1,
                  dtype: str = "bf16",
                  verbose: bool = False,
                  use_kv_cache: bool = True) -> List[Dict[str, Any]]:
    """Measure all individuals serially on the local A100. Returns aligned list."""
    monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[],
                          sync_execution_with="torch",
                          approx_instant_energy=True)
    out = []
    t0 = time.time()
    for i, ind in enumerate(individuals):
        t_start = time.time()
        r = measure_one(ind, prefill_len=prefill_len, decode_len=decode_len,
                        n_repeats=n_repeats, warmup=warmup, dtype=dtype,
                        monitor=monitor, use_kv_cache=use_kv_cache)
        elapsed = time.time() - t_start
        if verbose or True:
            tag = "OK" if r.get("envelope_feasible") else "FAIL"
            print(f"  [zeus {i+1}/{len(individuals)}] {tag}  "
                  f"ttft={r.get('ttft_ms', float('nan')):.2f}ms  "
                  f"tpot={r.get('tpot_ms', float('nan')):.2f}ms  "
                  f"E/tok={r.get('energy_per_token_uJ', float('nan')):.1f}uJ  "
                  f"[{elapsed:.1f}s]")
        out.append(r)
    total = time.time() - t0
    print(f"  ZEUS eval done: {len(individuals)} individuals in {total:.1f}s "
          f"(mean {total/max(1,len(individuals)):.1f}s/ind)")
    return out


if __name__ == "__main__":
    # Smoke test: load one individual from an existing ckpt and measure it.
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="ckpts/smollm2_rdxe_cosearch/0422_0705_ckpt_gen30.json")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--prefill_len", type=int, default=128)
    parser.add_argument("--decode_len", type=int, default=32)
    parser.add_argument("--n_repeats", type=int, default=1)
    args = parser.parse_args()
    d = json.load(open(args.ckpt))
    ind = d["individuals"][args.idx]
    r = measure_one(ind, prefill_len=args.prefill_len, decode_len=args.decode_len,
                    n_repeats=args.n_repeats)
    print(json.dumps({k: (float(v) if isinstance(v, (int, float)) else str(v))
                      for k, v in r.items() if k != "zeus_error" or v is not None},
                     indent=2, default=str))
