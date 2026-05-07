#!/usr/bin/env bash
# Example 4: Multi-seed sweep of the §4.4 search-strategy ablation.
#
# Runs three NSGA-II configurations:
#   * NSGA-II + IHA      (main recipe)
#   * Random  + IHA      (search-structure floor)
#   * NSGA-II + GQA      (IHA-expansion isolation, snapped to GQA-feasible)
#
# Each configuration is repeated for every seed listed in SEEDS, producing
# `len(SEEDS) x 3` independent NSGA-II runs whose checkpoints feed the
# multi-seed mean+/-std HV plot rendered by 05_plot_search_ablation_multi_seed.bash.
#
# All three configurations use:
#   - search space:    search_space_def/search_space_200M.yaml
#   - sw evaluator:    static Forge-Former (no fine-tuning, no real-train)
#   - hw evaluator:    none (params/flops/kv from analytical estimate)
#   - objectives:      (val_loss, params_M)  -- both minimized
#   - pop=24, off=48, gen=40, xover=0.6, mut=0.3
#
# Cost: each run is surrogate-only with HwNone, so wall-clock per run is a few
# minutes. 5 seeds * 3 configurations ~ 1-2 hours total on a single A100.
#
# Override the seed list with: SEEDS="42 1 7" bash 04_..._seed_sweep.bash
#
# Usage:
#   bash example_scripts/04_search_ablation_seed_sweep.bash

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
EVO_GPT_DIR="$REPO_ROOT/evo_gpt"

export PYTHONPATH="$EVO_GPT_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
cd "$LLMFORGE_DIR"

SEEDS="${SEEDS:-42 1 7 100 2026}"
DATE_TAG="$(date +%Y%m%d)"

# Shared NSGA-II hyperparameters across all configurations.
SHARED_FLAGS=(
    --search_space_config search_space_def/search_space_200M.yaml
    --max_layers 40 --min_layers 8
    --pop_size 24 --offspring 48 --generations 40
    --crossover_rate 0.6 --mutation_rate 0.3
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt
    --mc_dropout_n 10
    --objectives val_loss params_M
    --constraint val_loss=3.8
)

run_one () {
    local cfg_name="$1"; shift
    local cfg_flags=("$@")
    local seed
    for seed in $SEEDS; do
        local exp="${DATE_TAG}_ablation_${cfg_name}_paramsloss_seed${seed}"
        local ck_dir="ckpts/${exp}"
        if [[ -d "$ck_dir" ]] && ls "$ck_dir"/*ckpt_gen*.json >/dev/null 2>&1; then
            echo "[skip] ${cfg_name} seed=${seed} already ran -> ${ck_dir}"
            continue
        fi
        local log="logs/${exp}.log"
        mkdir -p logs "$ck_dir"
        echo "[run]  ${cfg_name} seed=${seed} -> ${ck_dir}"
        python -u script/ablations/run_ablation_search.py \
            --exp_name "$exp" \
            --log_dir logs \
            --seed "$seed" \
            "${SHARED_FLAGS[@]}" \
            "${cfg_flags[@]}" \
            2>&1 | tee -a "$log"
    done
}

run_one nsga_iha    --search_strategy nsga
run_one random_iha  --search_strategy random
run_one nsga_gqa    --search_strategy nsga --strict_gqa

echo ""
echo "[04_search_ablation_seed_sweep] done."
echo "  Output ckpt dirs match: ${LLMFORGE_DIR}/ckpts/${DATE_TAG}_ablation_*_paramsloss_seed*/"
echo "  Render the multi-seed figure with:"
echo "    bash example_scripts/05_plot_search_ablation_multi_seed.bash"
