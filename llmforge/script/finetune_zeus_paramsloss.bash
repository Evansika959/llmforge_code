#!/bin/bash
# Production NSGA-II co-search — ZEUS HW backend variant.
#   sw_mode = surrogate_finetune  (surrogate drives selection; every 5 gens
#                                  a batch of 8 archs is real-trained and
#                                  the surrogate is refit on the augmented
#                                  labels with 5:1 old:new buffer blend)
#   hw_mode = zeus                (every individual is measured on the local
#                                  GPU via ZEUS — ttft, tpot, decode energy,
#                                  power. Recorded in aux for analysis.)
# Search target: (val_loss, params_M).  ZEUS metrics are recorded but DO
# NOT drive NSGA selection — they only widen the aux dict so each Pareto
# candidate has a measured HW trace alongside its analytical params/KV/FLOPs.
# To make HW objectives drive the search, add e.g. `ttft` or
# `energy_per_token_uJ` to --objectives below and a corresponding constraint.
#
# Reproducibility / data flow:
#   --seed 42 controls NSGA evolution. Remote training process re-seeds
#   itself each invocation so per-arch dataset shuffles vary across gens.
#   ZEUS measurements are local A100, cached per arch hash (no re-measure
#   when an arch survives elimination across gens).
#
# Outputs:
#   ckpts/<EXP_NAME>/                — per-gen Population JSON ckpts
#                                       (aux carries ttft_ms, tpot_ms,
#                                        energy_per_token_uJ, power_W in
#                                        addition to the analytical fields)
#   ckpts/<EXP_NAME>/surrogate/      — fine-tuned surrogate gen{N}.pt + .json
#                                       (baseline ckpt is NEVER overwritten)
#   ckpts/<EXP_NAME>/al_payloads/    — per-event training YAML payloads
#   train/<EXP_NAME>/gen{N}/         — remote-trainer working dirs (CSV results)
#   logs/<EXP_NAME>_<ts>.log         — full stdout/stderr
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_finetune_200m_paramsloss_zeus"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
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
    --hw_mode zeus \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --zeus_n_repeats 1 --zeus_warmup 1 --zeus_dtype bf16 \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Fine-tuned surrogate ckpts: ckpts/${EXP_NAME}/surrogate/gen{N}.pt"
echo "Baseline surrogate at surrogate/ckpts/forgeformer.pt was NOT modified."
echo "ZEUS metrics (ttft_ms, tpot_ms, energy_per_token_uJ, power_W) are in"
echo "  each individual's aux dict — useful for HW-aware downstream analysis."
