"""All-in-one NSGA-II co-search driver.

Replaces the scattered run_cosearch_*/run_exp_* drivers with a single
dispatcher. Choose SW + HW evaluation backends with --sw_mode / --hw_mode;
`hw_none` (analytical params_M / kv_cache_bytes / flops_per_token) is
always composed in so those aux fields are present in every run.

See htmls/requirement.md for the full spec.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from nsga2 import EvaluationResult, Population, cons_value
from search_space import HeteroSearchSpace, Individual

# Evaluators
from evaluators.hw_none import HwNone, merge_hw_dicts
from evaluators.sw_surrogate import SwSurrogate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s: %(message)s")
for _name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("cosearch")


# ── CLI helpers ───────────────────────────────────────────────────────────

def _parse_constraint(entry: str) -> Tuple[str, float]:
    for op in (">=", "<=", "="):
        if op in entry:
            key, value = entry.split(op, 1)
            key = key.strip()
            if not key:
                raise argparse.ArgumentTypeError("Constraint key cannot be empty")
            if op == ">=":
                key = f"{key}_min"
            return key, float(value)
    raise argparse.ArgumentTypeError(
        "Constraints must be 'key=N', 'key<=N', or 'key>=N'")


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def _load_search_space_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["global_spec"], data["layer_spec"]


def _load_init_individuals_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    if isinstance(data, dict) and "individuals" in data:
        data = data["individuals"]
    elif isinstance(data, dict):
        data = [data]
    return data


# ── Hypervolume (early-stop on plateau) ───────────────────────────────────

def _hv_mc(points: np.ndarray, ref: np.ndarray, n_mc: int = 5000,
           rng: Optional[np.random.Generator] = None) -> float:
    if points.size == 0 or np.any(ref <= 0):
        return 0.0
    rng = rng or np.random.default_rng(0)
    samples = rng.uniform(0.0, ref, size=(n_mc, ref.size))
    dominated = np.zeros(n_mc, dtype=bool)
    for p in points:
        dominated |= np.all(samples >= p, axis=1)
    return float(np.prod(ref)) * float(dominated.mean())


def _population_hv(population, ref: np.ndarray) -> float:
    objs = np.array([e.objs for e in population.evaluations], dtype=np.float64)
    if objs.size == 0:
        return 0.0
    keep = np.ones(len(objs), bool)
    for i in range(len(objs)):
        if not keep[i]:
            continue
        for j in range(len(objs)):
            if i == j or not keep[j]:
                continue
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                keep[i] = False
                break
    finite = np.all(np.isfinite(objs[keep]), axis=1)
    return _hv_mc(objs[keep][finite], ref)


# ── Evaluator factories ───────────────────────────────────────────────────

def build_sw_evaluator(args, device, exp_name: str):
    """Returns (sw_eval, finetune_hooks_or_None).

    finetune_hooks is a small object with `should_fire(gen)` and
    `run_active_learning(...)` when sw_mode=surrogate_finetune; None
    otherwise. The dispatcher fires it between generations.
    """
    if args.sw_mode == "surrogate":
        ckpt = _resolve_path(args.surrogate_ckpt)
        return SwSurrogate(ckpt_path=ckpt, device=device,
                            mc_dropout_n=args.mc_dropout_n), None

    if args.sw_mode == "real_train":
        from evaluators.sw_real_train import SwRealTrain
        return SwRealTrain(
            hosts_file=args.realtrain_hosts_file,
            user=args.realtrain_user,
            ssh_key=os.path.expanduser(args.realtrain_ssh_key),
            conda_env=args.realtrain_conda_env,
            remote_evo_gpt_dir=args.realtrain_remote_evo_gpt_dir,
            max_iters=args.realtrain_max_iters,
            timeout=args.realtrain_timeout,
            poll_interval=args.realtrain_poll_interval,
            exp_name=exp_name,
        ), None

    if args.sw_mode == "surrogate_finetune":
        from evaluators.sw_finetune import SwFinetune
        from evaluators.sw_real_train import SwRealTrain
        ckpt = _resolve_path(args.surrogate_ckpt)
        rt = SwRealTrain(
            hosts_file=args.realtrain_hosts_file,
            user=args.realtrain_user,
            ssh_key=os.path.expanduser(args.realtrain_ssh_key),
            conda_env=args.realtrain_conda_env,
            remote_evo_gpt_dir=args.realtrain_remote_evo_gpt_dir,
            max_iters=args.realtrain_max_iters,
            timeout=args.realtrain_timeout,
            poll_interval=args.realtrain_poll_interval,
            exp_name=exp_name,
        )
        save_dir = (args.surrogate_save_dir
                    or os.path.join("ckpts", exp_name, "surrogate"))
        save_dir = _resolve_path(save_dir)
        ev = SwFinetune(
            ckpt_path=ckpt, device=device,
            mc_dropout_n=args.mc_dropout_n,
            real_train=rt,
            surrogate_save_dir=save_dir,
            finetune_every=args.finetune_every,
            finetune_batch=args.finetune_batch,
            base_dataset_csv=(args.finetune_base_csv or None),
            old_to_new_ratio=args.finetune_old_to_new_ratio,
        )
        return ev, ev   # ev itself is the finetune hook holder

    raise ValueError(f"Unknown sw_mode: {args.sw_mode}")


def build_hw_evaluator(args):
    """Returns (primary_hw_eval, hw_none_eval). Composer always merges them."""
    hw_none = HwNone(seq_len=args.seq_len)

    if args.hw_mode == "none":
        return None, hw_none

    if args.hw_mode == "zeus":
        from evaluators.hw_zeus import HwZeus
        return HwZeus(prefill_len=args.prefill_len, decode_len=args.decode_len,
                       n_repeats=args.zeus_n_repeats, warmup=args.zeus_warmup,
                       dtype=args.zeus_dtype, verbose=args.verbose,
                       use_kv_cache=not args.no_kv_cache), hw_none

    if args.hw_mode == "timeloop":
        if args.timeloop_substrate is None:
            raise ValueError("--timeloop_substrate is required when hw_mode=timeloop")
        if args.timeloop_substrate == "rdxe":
            from evaluators.hw_rdxe_inner import HwRdxeInner
            return HwRdxeInner(prefill_len=args.prefill_len,
                                decode_len=args.decode_len,
                                verbose=args.verbose), hw_none
        from evaluators.hw_timeloop import HwTimeloop
        return HwTimeloop(substrate=args.timeloop_substrate,
                           prefill_len=args.prefill_len,
                           decode_len=args.decode_len), hw_none

    raise ValueError(f"Unknown hw_mode: {args.hw_mode}")


# ── Dispatcher (per-generation evaluation) ────────────────────────────────

def evaluate_generation(inds: List[Individual], *,
                         sw_eval, hw_primary, hw_none, args):
    """Returns aligned (mu, sigma, hw_dicts) for a list of individuals.

    SW: one (mu, sigma) pair per individual.
    HW: hw_none aux is always present; primary HW backend (if any) merged on top.
    """
    if not inds:
        return [], [], []

    t0 = time.time()
    mu, sigma = sw_eval.evaluate(inds)
    sw_dt = time.time() - t0

    ind_dicts = [d.to_dict() if hasattr(d, "to_dict") else d for d in inds]
    if hw_primary is None:
        hw = hw_none.evaluate(ind_dicts)
        hw_label = "none"
    else:
        t1 = time.time()
        primary = hw_primary.evaluate(ind_dicts)
        hw = merge_hw_dicts(primary, hw_none.evaluate(ind_dicts))
        hw_label = type(hw_primary).__name__
        extra = ""
        if hasattr(hw_primary, "kv_cache_summary"):
            extra = f"  [{hw_primary.kv_cache_summary(primary)}]"
        log.info(f"  HW [{hw_label}]: {len(hw)} eval'd in {time.time()-t1:.1f}s{extra}")

    sig_range = (f"σ {min(sigma):.4f}..{max(sigma):.4f}"
                 if sigma else "σ n/a")
    log.info(f"  SW [{type(sw_eval).__name__}]: {len(mu)} preds in "
             f"{sw_dt:.1f}s; {sig_range}")
    return mu, sigma, hw


def build_evaluation_result(ind: Individual, mu_i: float, sigma_i: float,
                             hw: Dict[str, Any], objs: List[str],
                             cons: Dict[str, float],
                             acquisition_beta: float) -> EvaluationResult:
    val_for_nsga = mu_i - acquisition_beta * sigma_i
    auxs: Dict[str, Any] = {
        "val_loss": val_for_nsga,
        "val_loss_mu": mu_i,
        "val_loss_sigma": sigma_i,
        **hw,
    }
    for key in objs:
        if key not in auxs or auxs[key] is None:
            auxs[key] = float("inf")
    obj_vals = [float(auxs[o]) for o in objs]
    con_vals = [cons_value(c, cons[c], auxs) for c in cons]
    return EvaluationResult(obj_vals, con_vals, auxs)


# ── Validation: incompatible flag combos ──────────────────────────────────

def _validate_args(args, user_supplied: set) -> None:
    if args.sw_mode == "real_train":
        if "mc_dropout_n" in user_supplied:
            log.warning("--mc_dropout_n is ignored when sw_mode=real_train")
        if args.surrogate_ckpt:
            log.warning("--surrogate_ckpt is ignored when sw_mode=real_train")
    if args.sw_mode in ("surrogate", "surrogate_finetune"):
        if args.surrogate_ckpt is None:
            raise SystemExit(
                "[args] --surrogate_ckpt is required when sw_mode involves surrogate")
    if args.sw_mode in ("real_train", "surrogate_finetune"):
        if args.realtrain_hosts_file is None:
            raise SystemExit(
                "[args] --realtrain_hosts_file is required when real training is used")
    if args.hw_mode == "timeloop" and args.timeloop_substrate is None:
        raise SystemExit("[args] --timeloop_substrate required for hw_mode=timeloop")
    if args.hw_mode != "timeloop" and args.timeloop_substrate is not None:
        log.warning("--timeloop_substrate is ignored when hw_mode != timeloop")
    if args.hw_mode != "zeus":
        for k in ("no_kv_cache", "kv_cache_parity_check"):
            if k in user_supplied:
                log.warning(f"--{k} is ignored when hw_mode != zeus")
    init_flags = sum(x is not None for x in
                     (args.resume_ckpt, args.init_individuals, args.seed_arch))
    if init_flags > 1:
        raise SystemExit(
            "[args] --resume_ckpt / --init_individuals / --seed_arch are "
            "mutually exclusive — pick at most one")
    if args.seed_arch is None:
        leaked = [k for k in ("seed_p_mlp", "seed_p_head", "seed_p_kv",
                                "seed_p_qk_dim", "seed_p_v_dim", "seed_p_identity")
                  if k in user_supplied]
        if leaked:
            log.warning(f"--seed_p_* flags ignored without --seed_arch: {leaked}")


# ── KV-cache parity self-test (hw_mode=zeus only) ─────────────────────────

def _run_kv_cache_parity_check(search_space, device, dtype: str,
                                prefill_len: int) -> None:
    """Sample one arch, build it with random weights, and verify cached-vs-
    uncached final logits agree to bf16 tolerance. Aborts the run on
    failure so a silent miscompute can't propagate. No-ops cleanly on
    arches whose feature flags fall outside the cached path's supported
    subset (this is not a regression — the per-arch fallback in HwZeus
    will catch them at measurement time too)."""
    from zeus_eval import build_model_from_individual
    from zeus_kv_cache import (attach_iha_kv_cache, detach_iha_kv_cache,
                                parity_check, UnsupportedKVCache)
    torch_dtype = (torch.bfloat16 if dtype == "bf16"
                   else torch.float16 if dtype == "fp16"
                   else torch.float32)
    ind = search_space.sample()
    ind_dict = ind.to_dict() if hasattr(ind, "to_dict") else ind
    print("[kv-cache parity] building probe arch with random weights...")
    model = build_model_from_individual(
        ind_dict, block_size=prefill_len + 8 + 4,
        device=device, dtype=torch_dtype)
    try:
        attach_iha_kv_cache(model)
    except UnsupportedKVCache as e:
        print(f"[kv-cache parity] arch unsupported by cached path: {e}")
        del model
        torch.cuda.empty_cache()
        return
    try:
        ok, max_abs, max_rel = parity_check(
            model, prefill_len=16, decode_len=4, device=device, verbose=False)
        print(f"[kv-cache parity] max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(
                "[kv-cache parity] FAIL — cached and uncached final logits "
                "disagree beyond tolerance. Aborting before NSGA loop.")
    finally:
        detach_iha_kv_cache(model)
        del model
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Unified NSGA co-search (SW × HW dispatcher)")

    # [infrastructure]
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--save_offspring", action="store_true",
                   help="Also write {run_time}_offspring_gen{N}.json each "
                        "generation, capturing the evaluated offspring pool "
                        "before survival selection wipes it. Default off.")

    # [search-space]
    p.add_argument("--search_space_config", type=str,
                   default="search_space_def/search_space_200M.yaml")
    p.add_argument("--max_layers", type=int, default=40)
    p.add_argument("--min_layers", type=int, default=8)

    # [population init]
    p.add_argument("--init_individuals", type=str, default=None)
    p.add_argument("--seed_arch", type=str, default=None)
    p.add_argument("--seed_p_mlp", type=float, default=0.15)
    p.add_argument("--seed_p_head", type=float, default=0.10)
    p.add_argument("--seed_p_kv", type=float, default=0.10)
    p.add_argument("--seed_p_qk_dim", type=float, default=0.05)
    p.add_argument("--seed_p_v_dim", type=float, default=0.05)
    p.add_argument("--seed_p_identity", type=float, default=0.03)

    # [NSGA]
    p.add_argument("--pop_size", type=int, default=24)
    p.add_argument("--offspring", type=int, default=12)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--crossover_rate", type=float, default=0.6)
    p.add_argument("--mutation_rate", type=float, default=0.3)
    p.add_argument("--objectives", type=str, nargs="+",
                   default=["val_loss", "params"])
    p.add_argument("--constraint", action="append", type=_parse_constraint,
                   metavar="KEY=N|KEY<=N|KEY>=N")
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_eps", type=float, default=1e-3)

    # [SW eval]
    p.add_argument("--sw_mode", choices=["surrogate", "real_train",
                                         "surrogate_finetune"],
                   required=True)
    p.add_argument("--surrogate_ckpt", type=str, default=None)
    p.add_argument("--mc_dropout_n", type=int, default=10)
    p.add_argument("--acquisition_beta", type=float, default=1.0)
    p.add_argument("--surrogate_save_dir", type=str, default=None)
    p.add_argument("--realtrain_hosts_file", type=str,
                   default=None)
    p.add_argument("--realtrain_user", type=str, default=os.environ.get("USER", "anon"))
    p.add_argument("--realtrain_ssh_key", type=str,
                   default="~/.ssh/id_rsa")
    p.add_argument("--realtrain_conda_env", type=str, default="llmforge")
    p.add_argument("--realtrain_remote_evo_gpt_dir", type=str,
                   default="${EVO_GPT_DIR:-$HOME/evo_gpt}")
    p.add_argument("--realtrain_max_iters", type=int, default=20000)
    p.add_argument("--realtrain_timeout", type=int, default=16000)
    p.add_argument("--realtrain_poll_interval", type=int, default=120,
                   help="Seconds between checks for remote training "
                        "completion. Default 120 is fine for "
                        "max_iters=20k+ runs; tests should use ~5–10s.")
    p.add_argument("--finetune_every", type=int, default=5)
    p.add_argument("--finetune_batch", type=int, default=8)
    p.add_argument("--finetune_base_csv", type=str,
                   default="surrogate/dataset/dataset_200M.csv",
                   help="Old-data corpus mixed in at every fine-tune event "
                        "(experience replay). Leave empty string to disable "
                        "blending and train on the new buffer only.")
    p.add_argument("--finetune_old_to_new_ratio", type=float, default=5.0,
                   help="old:new sampling ratio per minibatch when "
                        "--finetune_base_csv is set. 5.0 = 5 old rows per 1 "
                        "new row in expectation. 0.0 disables the blend.")

    # [HW eval]
    p.add_argument("--hw_mode", choices=["none", "zeus", "timeloop"],
                   default="none")
    p.add_argument("--prefill_len", type=int, default=128)
    p.add_argument("--decode_len", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--zeus_n_repeats", type=int, default=1)
    p.add_argument("--zeus_warmup", type=int, default=1)
    p.add_argument("--zeus_dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--no_kv_cache", action="store_true",
                   help="hw_mode=zeus only: disable the measurement-only "
                        "KV-cache shim and fall back to Evo_GPT's "
                        "recomputing generate(). Default is cached, which "
                        "matches production decode and avoids inflating "
                        "tpot/energy with unnecessary recompute.")
    p.add_argument("--kv_cache_parity_check", action="store_true",
                   help="hw_mode=zeus only: at startup sample one arch and "
                        "verify cached-vs-uncached final logits agree to "
                        "bf16 tolerance. Aborts the run on failure.")
    p.add_argument("--timeloop_substrate", type=str, default=None,
                   choices=["eyeriss", "simba", "gemmini",
                            "flat_edge", "dxe", "dxe_relaxed", "rdxe"])

    # Capture which flags the user actually supplied (vs default-fallback) by
    # diffing argv tokens against the parser's option strings. Handles both
    # "--flag value" and "--flag=value" forms; filters to tokens that start
    # with '-' so positional values (e.g. "params" in `--objectives val_loss
    # params`) can never be mistaken for a flag name.
    raw = {tok.split("=", 1)[0] for tok in sys.argv[1:] if tok.startswith("-")}
    user_supplied = {a.dest for a in p._actions
                     if any(opt in raw for opt in a.option_strings)}
    args = p.parse_args()
    _validate_args(args, user_supplied)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load search space ──
    config_path = _resolve_path(args.search_space_config)
    global_spec, layer_spec = _load_search_space_yaml(config_path)
    search_space = HeteroSearchSpace.from_dicts(
        global_spec, layer_spec, L_max=args.max_layers, L_min=args.min_layers)
    print(f"Search space: {config_path}  (L_max={args.max_layers}, L_min={args.min_layers})")

    # ── Constraints ──
    objs = args.objectives
    cons = dict(args.constraint) if args.constraint else {"val_loss": 3.8}
    print(f"Objectives: {objs}")
    print(f"Constraints: {cons}")
    print(f"sw_mode={args.sw_mode}  hw_mode={args.hw_mode}"
          + (f"({args.timeloop_substrate})" if args.hw_mode == "timeloop" else ""))

    # ── Build evaluators ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sw_eval, finetune_hook = build_sw_evaluator(args, device, args.exp_name)
    hw_primary, hw_none = build_hw_evaluator(args)

    # ── ZEUS KV-cache banner + optional parity self-test ──
    if args.hw_mode == "zeus":
        kv_mode = "OFF (--no_kv_cache)" if args.no_kv_cache else "ON (cached decode)"
        print(f"ZEUS KV-cache mode: {kv_mode}")
        if not args.no_kv_cache and args.kv_cache_parity_check:
            _run_kv_cache_parity_check(
                search_space, device, args.zeus_dtype, args.prefill_len)

    # ── Init population ──
    if args.resume_ckpt:
        population = Population.load_checkpoint(
            args.resume_ckpt, from_pkl=args.resume_ckpt.endswith(".pkl"))
        population.search_space = search_space
        print(f"Resumed from {args.resume_ckpt} at gen {population.gen}")
    else:
        if args.seed_arch:
            from init.seed_arch import build_seeded_population
            seeds = build_seeded_population(
                _resolve_path(args.seed_arch), n=args.pop_size,
                search_space=search_space, global_spec=global_spec,
                layer_spec=layer_spec, L_max=args.max_layers, L_min=args.min_layers,
                p_mlp=args.seed_p_mlp, p_head=args.seed_p_head,
                p_kv=args.seed_p_kv, p_qk_dim=args.seed_p_qk_dim,
                p_v_dim=args.seed_p_v_dim, p_identity=args.seed_p_identity,
            )
            individuals = seeds
            print(f"Seeded population: {len(individuals)} from {args.seed_arch}")
        elif args.init_individuals:
            individuals = _load_init_individuals_json(_resolve_path(args.init_individuals))
            print(f"Init from {args.init_individuals}: {len(individuals)} archs")
        else:
            individuals = [search_space.sample() for _ in range(args.pop_size)]
            print(f"Random init: {len(individuals)} archs")
        population = Population(individuals, search_space=search_space,
                                 objs_settings=objs, cons_settings=cons)
        population.delete_duplicates()

    population.objs_settings = objs
    population.cons_settings = cons
    population.n_population = args.pop_size
    population.n_offspring = args.offspring
    population.crossover_rate = args.crossover_rate
    population.mutation_rate = args.mutation_rate

    exp_name = args.exp_name
    run_time = time.strftime("%m%d_%H%M", time.localtime())
    os.makedirs(f"ckpts/{exp_name}", exist_ok=True)

    # ── Initial evaluation ──
    if not args.resume_ckpt:
        print(f"\n{'='*60}\nInitial population: {len(population.individuals)} individuals")
        # SwRealTrain needs to know the current gen to tag payloads
        if hasattr(sw_eval, "set_gen"):
            sw_eval.set_gen(int(population.gen))
        mu, sigma, hw = evaluate_generation(
            population.individuals,
            sw_eval=sw_eval, hw_primary=hw_primary, hw_none=hw_none, args=args)
        population.evaluations = [
            build_evaluation_result(ind, m, s, h, objs, cons, args.acquisition_beta)
            for ind, m, s, h in zip(population.individuals, mu, sigma, hw)]
        population.eval_source = f"{args.sw_mode}+{args.hw_mode}"
        population.print_summary()
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen0.json")

    # ── Reference HV ──
    obj_arr = np.array([e.objs for e in population.evaluations], dtype=np.float64)
    finite = np.all(np.isfinite(obj_arr), axis=1)
    if finite.any():
        hv_ref = obj_arr[finite].max(axis=0) * 1.05
    else:
        hv_ref = np.full(len(objs), 1.0)
    hv_history = [_population_hv(population, hv_ref)]
    print(f"  HV (gen {population.gen}): {hv_history[-1]:.6g}")

    # ── Generation loop ──
    plateau = 0
    for gen_i in range(args.generations):
        population.generate_offspring()
        gen = population.gen
        print(f"\n{'='*60}\nGeneration {gen}")

        if hasattr(sw_eval, "set_gen"):
            sw_eval.set_gen(int(gen))

        mu_o, sigma_o, hw_o = evaluate_generation(
            population.offspring,
            sw_eval=sw_eval, hw_primary=hw_primary, hw_none=hw_none, args=args)
        population.offspring_evaluations = [
            build_evaluation_result(ind, m, s, h, objs, cons, args.acquisition_beta)
            for ind, m, s, h in zip(population.offspring, mu_o, sigma_o, hw_o)]
        population.eval_source = f"{args.sw_mode}+{args.hw_mode}"
        if args.save_offspring:
            population.save_checkpoint(
                f"ckpts/{exp_name}/{run_time}_offspring_gen{gen}.json")
        population.update_elimination()
        population.print_summary()
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen{gen}.json")

        hv = _population_hv(population, hv_ref)
        delta = hv - hv_history[-1]
        hv_history.append(hv)
        print(f"  HV (gen {gen}): {hv:.6g}  (Δ={delta:+.4g})")

        if args.early_stop_patience > 0:
            if delta < args.early_stop_eps:
                plateau += 1
                if plateau >= args.early_stop_patience:
                    print(f"\n[early-stop] HV plateau {plateau} gens "
                          f"(Δ < {args.early_stop_eps}); stopping.")
                    break
            else:
                plateau = 0

        # Active-learning event (sw_mode=surrogate_finetune only)
        if finetune_hook is not None and finetune_hook.should_fire(gen):
            # Re-predict on current pop with the same MC budget
            mu_full, sigma_full = sw_eval.evaluate(population.individuals)
            log.info(f"[finetune] event fired at gen {gen}")
            finetune_hook.run_active_learning(population, mu_full, sigma_full)

    print(f"\n{'='*60}\nDone. Checkpoints in ckpts/{exp_name}/")
    hv_str = ", ".join(f"{v:.4g}" for v in hv_history)
    print(f"HV trajectory ({len(hv_history)} pts): {hv_str}")
    print(f"HV gain (gen0→final): {hv_history[-1] - hv_history[0]:+.4g}")


if __name__ == "__main__":
    main()
