#!/bin/bash
# Ablation: Random search + IHA, surrogate-only, 2-objective (val_loss + params_M).
# Each generation samples N fresh candidates uniformly from the IHA search
# space; NSGA-II elitism still applies in the survival step so the population
# carries forward the best architectures discovered so far. This is the
# "search-structure floor" baseline for the §4.4 ablation: it shares search
# space + surrogate + budget with the NSGA + IHA recipe and only differs in
# how offspring are generated.
#
# Search recipe:
#   - Search space: full IHA 200M-class (search_space_def/search_space_200M.yaml)
#   - Selection:    random offspring per generation (no crossover / mutation
#                   pressure); NSGA-II survival selects the top N each gen.
#   - Surrogate:    static ForgeFormer (no active-learning fine-tune)
#   - HW backend:   none
#   - Objectives:   (val_loss, params_M)  — both minimized
#
# Outputs: ckpts/<EXP_NAME>/{ts}_ckpt_gen{N}.json, logs/<EXP_NAME>_<ts>.log
set -e
cd "$(dirname "$0")/../.."  # nsga_search/

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_ablation_random_iha_paramsloss"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u script/ablations/run_ablation_search.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 24 --offspring 48 --generations 40 \
    --search_strategy random \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
