#!/bin/bash
# Ablation: NSGA-II + strict-GQA, surrogate-only, 2-objective (val_loss + params_M).
# Same population / generation budget and surrogate as the IHA variants, but
# every layer's attention shape is snapped to the GQA-feasible subset:
#     n_v_head_dim = n_qk_head_dim
#     n_head * n_qk_head_dim = n_embd
#     n_kv_group divides n_head      (already enforced upstream)
# These constraints reproduce the shape constraint set inherited by GQA from
# multi-head attention. Used to isolate the contribution of IHA's expanded
# search space relative to NSGA-II + ForgeFormer alone.
#
# Search recipe:
#   - Search space: search_space_200M.yaml, repaired through StrictGQASearchSpace
#   - Selection:    NSGA-II tournament + crossover + mutation
#   - Surrogate:    static ForgeFormer (no active-learning fine-tune)
#   - HW backend:   none
#   - Objectives:   (val_loss, params_M)  — both minimized
#
# Outputs: ckpts/<EXP_NAME>/{ts}_ckpt_gen{N}.json, logs/<EXP_NAME>_<ts>.log
set -e
cd "$(dirname "$0")/../.."  # nsga_search/

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_ablation_nsga_gqa_paramsloss"
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
    --strict_gqa \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
