#!/bin/bash
# Resume the 20260427_finetune_200m_paramsloss_zeus run that crashed at the
# gen-35 refit step (sidecar JSON for the v2 surrogate had been renamed
# mid-flight).  Now resolved via symlinks; this script picks up from gen 35.
#
# What's restored:
#   - All gen ckpts 0..35 are on disk (pop state intact)
#   - Train CSVs for AL events 1..7 are on disk (real labels recoverable)
#   - Surrogate refits 5/10/15/20/25/30 saved (AL #7's refit was lost to the
#     crash; the 7 valid labels from gen-35's CSV are NOT carried forward
#     into the in-memory label_buffer on resume — they live in
#     train/<EXP>/CSV for offline merging if you want)
#
# What this run will do:
#   --generations 5 → produce gens 36..40 + AL #8 at gen 40 (final event)
#   AL #8 will refit on the freshly-collected 7 labels (no buffer continuity).
#   The post-AL-#8 ckpt will save as ckpts/<EXP>/surrogate/gen40.pt.
#
# Surrogate continuity:
#   The script uses surrogate/ckpts/forgeformer.pt which is now
#   a symlink → forgeformer.pt (the canonical v2). Same architecture, same
#   weights as the file the original run used during gens 0..30.
set -e
cd "$(dirname "$0")/.."

EXP_NAME="20260427_finetune_200m_paramsloss_zeus"
RESUME_CKPT="ckpts/${EXP_NAME}/0427_0647_ckpt_gen35.json"
ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/${EXP_NAME}_resume_${ts}.log"
mkdir -p logs

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "ERROR: $RESUME_CKPT not found"; exit 1
fi

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --resume_ckpt "$RESUME_CKPT" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 24 --offspring 48 --generations 5 \
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
    --realtrain_remote_llmforge_train_dir ${LLMFORGE_TRAIN_DIR:-$HOME/llmforge_train} \
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
echo "Resume done. Run dir: ckpts/${EXP_NAME}/"
echo "Resume log:           ${log}"
