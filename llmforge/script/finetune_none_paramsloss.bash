#!/bin/bash
# Production NSGA-II co-search.
#   sw_mode = surrogate_finetune  (surrogate drives selection; every 5 gens a
#                                  batch of 8 archs is real-trained and the
#                                  surrogate is refit on the augmented labels)
#   hw_mode = none                (analytical params_M / kv_cache_bytes /
#                                  flops_per_token always recorded in aux)
# Search target: (val_loss, params_M). No params-band constraint; only
# val_loss<=3.8 keeps obviously-broken archs from poisoning the Pareto front.
#
# Reproducibility:
#   --seed 42 controls NSGA evolution (init sample, crossover/mutation, MC-
#   dropout RNG). The remote training process re-seeds itself each invocation
#   so per-arch dataset shuffles vary across generations even when the search
#   seed stays fixed.
#
# Outputs land at:
#   ckpts/<EXP_NAME>/                — per-gen Population JSON ckpts
#   ckpts/<EXP_NAME>/surrogate/      — fine-tuned surrogate gen{N}.pt + .json
#                                       (baseline ckpt is NEVER overwritten)
#   ckpts/<EXP_NAME>/al_payloads/    — per-event training YAML payloads
#   train/<EXP_NAME>/gen{N}/         — remote-trainer working dirs (CSV results)
#   logs/<EXP_NAME>_<ts>.log         — full stdout/stderr
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$426_finetune_200m_paramsloss"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs_ \
    --seed 42 \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 24 --offspring 48 --generations 40 \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --sw_mode surrogate_finetune \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 --acquisition_beta 1.0 \
    --finetune_every 5 --finetune_batch 8 \
    --finetune_base_csv surrogate/dataset/dataset_200M.csv \
    --finetune_old_to_new_ratio 5.0 \
    --realtrain_hosts_file script/examples/hosts_example.yaml \
    --realtrain_user ${USER:-anon} \
    --realtrain_ssh_key ~/.ssh/id_rsa \
    --realtrain_conda_env ${CONDA_ENV:-llmforge} \
    --realtrain_remote_evo_gpt_dir ${EVO_GPT_DIR:-$HOME/evo_gpt} \
    --realtrain_max_iters 20000 \
    --realtrain_timeout 16000 \
    --realtrain_poll_interval 600 \
    --hw_mode none --seq_len 256 \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Fine-tuned surrogate ckpts: ckpts/${EXP_NAME}/surrogate/gen{N}.pt"
echo "Baseline surrogate at surrogate/ckpts/forgeformer.pt was NOT modified."
