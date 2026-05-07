#!/bin/bash
# Ablation: NSGA-II + IHA, surrogate-only, 2-objective (val_loss + params_M).
# This is the "main recipe" reference curve for the §4.4 search-strategy
# ablation. Same population/generation budget as the random and GQA variants
# below so the three HV trajectories are directly comparable.
#
# Search recipe:
#   - Search space: full IHA 200M-class (search_space_def/search_space_200M.yaml)
#   - Selection:    NSGA-II tournament + crossover + mutation
#   - Surrogate:    static ForgeFormer (no active-learning fine-tune)
#   - HW backend:   none (params/flops/kv-cache from analytical estimate)
#   - Objectives:   (val_loss, params_M)  — both minimized
#
# Outputs:
#   ckpts/<EXP_NAME>/{ts}_ckpt_gen{N}.json — same schema as run_cosearch.py
#                                             (compatible with the existing
#                                              plot_hv_compare / plot_substrate_summary
#                                              scripts under paper/scripts/)
#   logs/<EXP_NAME>_<ts>.log
set -e
cd "$(dirname "$0")/../.."  # nsga_search/

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_ablation_nsga_iha_paramsloss"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u script/ablations/run_ablation_search.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 24 --offspring 48 --generations 40 \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --search_strategy nsga \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
