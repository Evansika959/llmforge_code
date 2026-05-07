#!/bin/bash
# Test: --resume_ckpt. Run gen 0+1, then resume from the gen-1 ckpt and
# continue for one more generation. Confirms ckpt round-trip.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_10_resume_${TS}"

# Phase A: produce a gen-1 ckpt
run_test "${EXP}_A" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations 1 \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode none --seq_len $SQL \
    --objectives val_loss params_M \
    --constraint val_loss=4.0
assert_ckpt "${EXP}_A" 1

CKPT_PATH=$(ls "ckpts/${EXP}_A/"*"_ckpt_gen1.json" | head -1)
echo "[resume] from $CKPT_PATH"

# Phase B: resume + 1 more gen
run_test "${EXP}_B" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations 1 \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode none --seq_len $SQL \
    --resume_ckpt "$CKPT_PATH" \
    --objectives val_loss params_M \
    --constraint val_loss=4.0
assert_ckpt "${EXP}_B" 2

pass "10_resume_ckpt"
