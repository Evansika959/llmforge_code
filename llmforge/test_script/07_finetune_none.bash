#!/bin/bash
# Test: sw=surrogate_finetune × hw=none. End-to-end active learning.
# generations=2 + finetune_every=1 forces an event after gen 1.
# Cluster-bound; tuned for fast smoke:
#   max_iters=100   - just past warmup, ~30s/H100 job
#   poll_interval=5 - notice the cluster jobs finishing within seconds
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_07_finetune_none_${TS}"
GENS=${GENS_OVERRIDE:-2}
FT_ITERS=${FT_ITERS:-100}
FT_POLL=${FT_POLL:-5}

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate_finetune --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC --acquisition_beta 1.0 \
    --finetune_every 1 --finetune_batch 2 \
    --realtrain_hosts_file "$HOSTS_FILE" \
    --realtrain_max_iters $FT_ITERS --realtrain_timeout $RT_TIMEOUT \
    --realtrain_poll_interval $FT_POLL \
    --hw_mode none --seq_len $SQL \
    --objectives val_loss params_M \
    --constraint val_loss=99

assert_ckpt "$EXP" "$GENS"
# Confirm a fine-tuned ckpt landed in the dedicated save dir.
if ! ls "ckpts/${EXP}/surrogate/"gen*.pt >/dev/null 2>&1; then
  echo "[FAIL] no fine-tuned surrogate ckpt at ckpts/${EXP}/surrogate/"
  exit 1
fi
pass "07_finetune_none"
