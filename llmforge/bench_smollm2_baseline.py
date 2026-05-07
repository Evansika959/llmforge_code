"""Baseline measurement for a single reference arch (e.g. SmolLM2-135M).

Two phases — either or both can be selected via flags:

  1. ZEUS HW eval     — `--do_zeus` (default ON)
     Builds the model from the reference YAML with random weights and
     measures ttft / tpot / power / energy via `zeus_eval.measure_one`.
     Routes through the KV-cache shim by default (matches production
     decode); add `--no_kv_cache` to A/B against the recompute path.

  2. Real training    — `--do_train` (default ON)
     Submits the same arch to the remote cluster via `SwRealTrain` and
     blocks until val_loss is returned. Uses the canonical 20k-iter
     recipe by default; override with `--max_iters`.

The output is one summary table covering whichever phases ran, written
to stdout and to `logs/<exp_name>_<TS>.log`.

Example:
    python bench_smollm2_baseline.py \
        --ref_yaml reference_archs/smollm2_135m.yaml \
        --exp_name smollm2_135m_baseline \
        --prefill_len 256 --decode_len 256 --seq_len 512 \
        --realtrain_hosts_file script/examples/hosts_example.yaml \
        --max_iters 20000 --timeout 16000 --poll_interval 600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s: %(message)s")
for _name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("bench_smollm2")


# ── Reference → Individual-shaped dict ─────────────────────────────────────

def reference_to_ind_dict(ref_yaml_path: str) -> Dict[str, Any]:
    """Load the reference YAML and pack into the {globals, layers} layout
    consumed by both `zeus_eval.measure_one` and `archs_to_training_yaml`.
    No layer-mask padding — every layer is active.

    Honors an optional top-level `mlp_variant` YAML field (e.g. Pythia
    sets `mlp_variant: mlp` for 2-matrix GeLU); pulled onto
    `globals.mlp_variant` so `build_model_from_individual` builds the
    matching MLP without a CLI flag."""
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


def estimate_params_M(ind_dict: Dict[str, Any]) -> float:
    """Cheap parameter count for the active layers (vocab/embedding excluded
    consistent with HwNone). Used for the summary line only.

    Assumes SwiGLU MLP (3 matrices: gate + up + down, each `n_embd × mlp_size`)
    to match `zeus_eval.build_model_from_individual`'s `mlp_variant="swiglu"`
    default and `remote_trainer._FIXED_MODEL_OVERRIDES["mlp_variant"]="swiglu"`.
    A 2-matrix GeLU MLP would underestimate by ~33% per layer; do not change
    this without also flipping the build defaults.
    """
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
        # Q, K, V, c_proj
        total += n_embd * (n_head * qk_d)
        total += n_embd * (n_kv * qk_d)
        total += n_embd * (n_kv * v_d)
        total += (n_head * v_d) * n_embd if g.get("use_concat_heads", True) else v_d * n_embd
        # SwiGLU MLP — gate + up + down, each n_embd × mlp_size.
        total += 3 * n_embd * mlp
    return total / 1e6


# ── Phase 1: ZEUS HW eval ──────────────────────────────────────────────────

def run_zeus_phase(ind_dict: Dict[str, Any], args) -> Dict[str, Any]:
    log.info("──────── Phase 1: ZEUS HW eval ────────")
    from zeus_eval import measure_one
    from zeus.monitor import ZeusMonitor
    monitor = ZeusMonitor(gpu_indices=[0], cpu_indices=[],
                          sync_execution_with="torch",
                          approx_instant_energy=True)
    log.info(f"Workload: prefill={args.prefill_len}, decode={args.decode_len}, "
             f"dtype={args.zeus_dtype}, n_repeats={args.zeus_n_repeats}, "
             f"warmup={args.zeus_warmup}, kv_cache={'OFF' if args.no_kv_cache else 'ON'}")
    t0 = time.time()
    r = measure_one(
        ind_dict,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        n_repeats=args.zeus_n_repeats,
        warmup=args.zeus_warmup,
        dtype=args.zeus_dtype,
        monitor=monitor,
        use_kv_cache=not args.no_kv_cache,
    )
    log.info(f"ZEUS measurement took {time.time()-t0:.1f}s")
    if not r.get("envelope_feasible"):
        log.warning(f"ZEUS measurement failed: {r.get('zeus_error', 'unknown')}")
    return r


# ── Phase 2: Remote real training (val_loss oracle) ───────────────────────

def run_train_phase(ind_dict: Dict[str, Any], args) -> Dict[str, Any]:
    log.info("──────── Phase 2: Remote real training (val_loss) ────────")
    from evaluators.sw_real_train import SwRealTrain
    rt = SwRealTrain(
        hosts_file=args.realtrain_hosts_file,
        user=args.realtrain_user,
        ssh_key=os.path.expanduser(args.realtrain_ssh_key),
        conda_env=args.realtrain_conda_env,
        remote_llmforge_train_dir=args.realtrain_remote_llmforge_train_dir,
        max_iters=args.max_iters,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        exp_name=args.exp_name,
        dataset=args.dataset,
    )
    rt.set_gen(0)                                  # single-shot baseline
    log.info(f"Submitting 1 arch with max_iters={args.max_iters}, "
             f"timeout={args.timeout}s, dataset={args.dataset}")
    t0 = time.time()
    labels, _sigmas = rt.evaluate([ind_dict])
    val_loss = labels[0] if labels else float("nan")
    log.info(f"Real-training took {time.time()-t0:.1f}s; val_loss={val_loss}")
    return {"val_loss": val_loss, "max_iters": args.max_iters,
            "dataset": args.dataset, "wall_s": time.time() - t0}


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(ind_dict: Dict[str, Any], zeus: Optional[Dict[str, Any]],
                  train: Optional[Dict[str, Any]], args) -> None:
    print()
    print("=" * 78)
    print(f"BASELINE SUMMARY — {ind_dict.get('name', 'arch')}")
    print("=" * 78)

    g = ind_dict["globals"]
    n_active = sum(1 for m in g.get("layer_mask", []) if m)
    print(f"  n_embd            : {g['n_embd']}")
    print(f"  block_size        : {g['block_size']}")
    print(f"  active n_layer    : {n_active}")
    print(f"  use_concat_heads  : {g['use_concat_heads']}")
    print(f"  est. params       : {estimate_params_M(ind_dict):.2f} M (attn+MLP, no embedding)")
    print()

    if zeus is not None:
        print(f"  [ZEUS HW]   prefill={args.prefill_len}, decode={args.decode_len}, "
              f"dtype={args.zeus_dtype}, kv_cache={'OFF' if args.no_kv_cache else 'ON'}")
        if zeus.get("envelope_feasible"):
            print(f"    ttft_ms                : {zeus['ttft_ms']:.3f}")
            print(f"    tpot_ms                : {zeus['tpot_ms']:.3f}")
            print(f"    energy_per_token_uJ    : {zeus['energy_per_token_uJ']:.0f}")
            print(f"    session_e_per_tok_uJ   : {zeus['session_e_per_tok_uJ']:.0f}")
            print(f"    power_W                : {zeus['power_W']:.1f}")
            print(f"    decode_energy_J        : {zeus['zeus_decode_energy_J']:.3f}")
            print(f"    prefill_energy_J       : {zeus['zeus_prefill_energy_J']:.3f}")
            print(f"    zeus_kv_cache_used     : {zeus.get('zeus_kv_cache_used')}")
        else:
            print(f"    FAILED: {zeus.get('zeus_error', 'unknown')}")
        print()

    if train is not None:
        print(f"  [Real training]  max_iters={train['max_iters']}, "
              f"dataset={train['dataset']}, wall={train['wall_s']:.0f}s")
        v = train["val_loss"]
        if v == v and v != float("inf"):
            print(f"    val_loss               : {v:.4f}")
        else:
            print(f"    val_loss               : FAILED ({v})")
        print()

    # Persist to JSON sidecar for downstream consumers (e.g. Pareto plots).
    if args.summary_json:
        out = {
            "name": ind_dict.get("name"),
            "globals": g,
            "n_active_layers": n_active,
            "params_M_est": estimate_params_M(ind_dict),
            "zeus": zeus,
            "train": train,
            "config": {
                "prefill_len": args.prefill_len,
                "decode_len": args.decode_len,
                "seq_len": args.seq_len,
                "zeus_dtype": args.zeus_dtype,
                "no_kv_cache": args.no_kv_cache,
            },
        }
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Summary JSON saved → {args.summary_json}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Reference + identity
    p.add_argument("--ref_yaml", type=str, required=True,
                   help="Path to reference arch YAML (e.g. "
                        "reference_archs/smollm2_135m.yaml).")
    p.add_argument("--exp_name", type=str, required=True,
                   help="Used to namespace remote-training payloads + result CSV.")
    p.add_argument("--summary_json", type=str, default=None,
                   help="If set, write a JSON sidecar with all metrics here.")

    # Phase toggles
    p.add_argument("--do_zeus",  dest="do_zeus",  action="store_true", default=True)
    p.add_argument("--no_zeus",  dest="do_zeus",  action="store_false")
    p.add_argument("--do_train", dest="do_train", action="store_true", default=True)
    p.add_argument("--no_train", dest="do_train", action="store_false")
    p.add_argument("--inject_val_loss", type=float, default=None,
                   help="Bypass remote training and fill the summary's `train` "
                        "block with this val_loss (e.g. when a previous run "
                        "already produced the number). Implies --no_train.")

    # ZEUS knobs
    p.add_argument("--prefill_len", type=int, default=256)
    p.add_argument("--decode_len", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=512,
                   help="Reported in summary; not directly used by ZEUS but "
                        "matches the dispatcher's --seq_len for consistency.")
    p.add_argument("--zeus_n_repeats", type=int, default=3)
    p.add_argument("--zeus_warmup", type=int, default=2)
    p.add_argument("--zeus_dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no_kv_cache", action="store_true",
                   help="A/B against the recompute path.")

    # Remote-training knobs
    p.add_argument("--realtrain_hosts_file", type=str, default=None)
    p.add_argument("--realtrain_user", type=str, default=os.environ.get("USER", "anon"))
    p.add_argument("--realtrain_ssh_key", type=str, default="~/.ssh/id_rsa")
    p.add_argument("--realtrain_conda_env", type=str, default="llmforge")
    p.add_argument("--realtrain_remote_llmforge_train_dir", type=str,
                   default="${LLMFORGE_TRAIN_DIR:-$HOME/llmforge_train}")
    p.add_argument("--max_iters", type=int, default=20000,
                   help="Matches the canonical 20k-iter recipe used by "
                        "the cosearch finetune scripts.")
    p.add_argument("--timeout", type=int, default=16000)
    p.add_argument("--poll_interval", type=int, default=600)
    p.add_argument("--dataset", type=str, default="minipile")

    args = p.parse_args()

    if args.inject_val_loss is not None:
        # Caller is supplying the val_loss out-of-band; skip the cluster.
        args.do_train = False

    if args.do_train and not args.realtrain_hosts_file:
        raise SystemExit("[args] --realtrain_hosts_file is required when "
                         "--do_train is on (set --no_train to skip).")

    ref_path = (args.ref_yaml if os.path.isabs(args.ref_yaml)
                else os.path.join(SCRIPT_DIR, args.ref_yaml))
    ind_dict = reference_to_ind_dict(ref_path)
    print(f"Loaded reference arch: {ind_dict.get('name', '?')} from {ref_path}")
    print(f"  active layers: {sum(1 for m in ind_dict['globals']['layer_mask'] if m)}")
    print(f"  est params   : {estimate_params_M(ind_dict):.2f} M")

    zeus_result = None
    train_result = None

    if args.do_zeus:
        if not torch.cuda.is_available():
            log.warning("CUDA not available — skipping ZEUS phase.")
        else:
            zeus_result = run_zeus_phase(ind_dict, args)

    if args.do_train:
        train_result = run_train_phase(ind_dict, args)
    elif args.inject_val_loss is not None:
        train_result = {
            "val_loss": float(args.inject_val_loss),
            "max_iters": args.max_iters,
            "dataset": args.dataset,
            "wall_s": 0.0,
            "source": "injected (--inject_val_loss); not run in this process",
        }
        log.info(f"Injected val_loss={args.inject_val_loss} (no remote training)")

    print_summary(ind_dict, zeus_result, train_result, args)


if __name__ == "__main__":
    main()
