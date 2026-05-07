"""Ablation runner for §4.4 search-strategy comparison.

A focused, ablation-only entry point that mirrors run_cosearch.py for the
two-objective surrogate-only configuration used by the search-strategy
study. Imports the production helpers (constraint parsing, evaluation-result
builder, etc.) from run_cosearch so the output ckpt schema is byte-compatible
with the existing plotting infrastructure.

Three configurations are supported via flags:

    (default)            NSGA + IHA              (main recipe)
    --search_strategy random            Random search + IHA
    --strict_gqa                         NSGA + GQA-restricted (IHA snapped
                                         to GQA-feasible subset)

The surrogate is the only software evaluator (no fine-tuning, no
real-training, no active learning), the hardware backend is "none" (params
and flops are computed analytically), and the search space is the IHA
200M-class space defined by the yaml passed via --search_space_config.

Output: ckpts/<exp_name>/<ts>_ckpt_genN.json — same schema as run_cosearch.py.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow this file to import from the parent nsga_search/ package and from
# its sibling search_space_gqa.py module.
THIS_DIR = Path(__file__).resolve().parent
NSGA_DIR = THIS_DIR.parent.parent  # script/ablations/ -> script/ -> nsga_search/
if str(NSGA_DIR) not in sys.path:
    sys.path.insert(0, str(NSGA_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from nsga2 import Population
from search_space import HeteroSearchSpace
from evaluators.sw_surrogate import SwSurrogate
from evaluators.hw_none import HwNone

# Reuse production helpers so ckpt format and constraint semantics match
# run_cosearch.py exactly.
from run_cosearch import (
    _parse_constraint,
    build_evaluation_result,
)

from search_space_gqa import StrictGQASearchSpace


def _load_search_space_yaml(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["global_spec"], cfg["layer_spec"]


def _evaluate_population(individuals, sw_eval, hw_eval, objs, cons,
                        acquisition_beta):
    """Surrogate prediction + analytical hardware estimate; build EvalResults."""
    mu, sigma = sw_eval.evaluate(individuals)
    hw_results = hw_eval.evaluate(individuals)
    return [
        build_evaluation_result(ind, m, s, h, objs, cons, acquisition_beta)
        for ind, m, s, h in zip(individuals, mu, sigma, hw_results)
    ]


def main():
    p = argparse.ArgumentParser(
        description="Ablation runner for §4.4 search-strategy comparison "
                    "(surrogate-only, 2-objective)."
    )

    # Infrastructure
    p.add_argument("--exp_name", required=True)
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--seed", type=int, default=42)

    # Search space
    p.add_argument("--search_space_config",
                   default="search_space_def/search_space_200M.yaml")
    p.add_argument("--max_layers", type=int, default=40)
    p.add_argument("--min_layers", type=int, default=8)

    # Population / NSGA params (NSGA params ignored when --search_strategy=random)
    p.add_argument("--pop_size", type=int, default=24)
    p.add_argument("--offspring", type=int, default=48)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--crossover_rate", type=float, default=0.6)
    p.add_argument("--mutation_rate", type=float, default=0.3)
    p.add_argument("--objectives", nargs="+",
                   default=["val_loss", "params_M"])
    p.add_argument("--constraint", action="append", type=_parse_constraint,
                   metavar="KEY=N|KEY<=N|KEY>=N")

    # Ablation knobs
    p.add_argument("--search_strategy", choices=["nsga", "random"],
                   default="nsga",
                   help="'nsga' = tournament + crossover + mutation; "
                        "'random' = uniformly resample offspring each gen "
                        "(NSGA-II elitism still applies during survival).")
    p.add_argument("--strict_gqa", action="store_true",
                   help="Snap every layer to the GQA-feasible subset of the "
                        "IHA search space (n_v_head_dim = n_qk_head_dim and "
                        "n_head * n_qk_head_dim = n_embd).")

    # Surrogate
    p.add_argument("--surrogate_ckpt", required=True)
    p.add_argument("--mc_dropout_n", type=int, default=10)
    p.add_argument("--acquisition_beta", type=float, default=1.0,
                   help="UCB-style μ - β·σ acquisition; preserved for ckpt-"
                        "schema compatibility with run_cosearch.")
    p.add_argument("--ckpt_dir", default=None,
                   help="Output ckpt directory. Default: ckpts/<exp_name>/")

    args = p.parse_args()

    # Resolve paths against nsga_search/ (mirrors run_cosearch.py behavior).
    os.chdir(NSGA_DIR)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Search space ──
    global_spec, layer_spec = _load_search_space_yaml(args.search_space_config)
    if args.strict_gqa:
        search_space = StrictGQASearchSpace.from_dicts(
            global_spec, layer_spec,
            L_max=args.max_layers, L_min=args.min_layers,
        )
        valid = search_space._gqa_constraints_summary()
        print(f"[search-space] StrictGQASearchSpace; "
              f"valid (n_head, d_qk) pairs: {valid}")
    else:
        search_space = HeteroSearchSpace.from_dicts(
            global_spec, layer_spec,
            L_max=args.max_layers, L_min=args.min_layers,
        )
        print(f"[search-space] HeteroSearchSpace (full IHA)")

    # ── Constraints ──
    cons = dict(args.constraint) if args.constraint else {"val_loss": 3.8}

    # ── Evaluators ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sw_eval = SwSurrogate(
        ckpt_path=args.surrogate_ckpt, device=device,
        mc_dropout_n=args.mc_dropout_n,
    )
    hw_eval = HwNone()
    print(f"[eval] surrogate={args.surrogate_ckpt}  hw=none")
    print(f"[search] objectives={args.objectives}  constraints={cons}")
    print(f"[search] strategy={args.search_strategy}  "
          f"strict_gqa={args.strict_gqa}")

    # ── Init population ──
    individuals = [search_space.sample() for _ in range(args.pop_size)]
    population = Population(
        individuals,
        search_space=search_space,
        objs_settings=list(args.objectives),
        cons_settings=cons,
    )
    population.delete_duplicates()
    population.n_population = args.pop_size
    population.n_offspring = args.offspring
    population.crossover_rate = args.crossover_rate
    population.mutation_rate = args.mutation_rate

    # ── Output dirs ──
    exp_name = args.exp_name
    ckpt_dir = args.ckpt_dir or f"ckpts/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    run_time = time.strftime("%m%d_%H%M", time.localtime())
    print(f"[output] ckpts -> {ckpt_dir}/{run_time}_ckpt_genN.json")

    # ── Initial evaluation ──
    print(f"\n{'='*60}\nGen 0: evaluating {len(population.individuals)} individuals")
    population.evaluations = _evaluate_population(
        population.individuals, sw_eval, hw_eval,
        list(args.objectives), cons, args.acquisition_beta,
    )
    population.eval_source = (
        f"surrogate+none+{args.search_strategy}"
        + ("+strict_gqa" if args.strict_gqa else "")
    )
    population.save_checkpoint(f"{ckpt_dir}/{run_time}_ckpt_gen0.json")
    print(f"[gen 0] saved checkpoint")

    # ── Generation loop ──
    for gen_i in range(args.generations):
        if args.search_strategy == "random":
            population.generate_offspring_random()
        else:
            population.generate_offspring()
        gen = population.gen
        print(f"\n{'='*60}\nGeneration {gen}")

        population.offspring_evaluations = _evaluate_population(
            population.offspring, sw_eval, hw_eval,
            list(args.objectives), cons, args.acquisition_beta,
        )
        population.update_elimination()
        population.save_checkpoint(f"{ckpt_dir}/{run_time}_ckpt_gen{gen}.json")
        print(f"[gen {gen}] saved checkpoint  "
              f"(population size: {len(population.individuals)})")

    print(f"\n{'='*60}\nDone. {args.generations} generations completed.")
    print(f"Run directory: {ckpt_dir}/")


if __name__ == "__main__":
    main()
