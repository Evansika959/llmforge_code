#!/bin/bash
# Test: sw=surrogate × hw=zeus. Local-GPU (A100) measurement.
# Expected runtime: ~30-60s.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_02_surrogate_zeus_${TS}"

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC --acquisition_beta 1.0 \
    --hw_mode zeus --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --zeus_n_repeats 1 --zeus_warmup 1 --zeus_dtype bf16 \
    --objectives val_loss params_M ttft tpot \
    --constraint val_loss=4.0

assert_ckpt "$EXP" "$GENS"
pass "02_surrogate_zeus"
