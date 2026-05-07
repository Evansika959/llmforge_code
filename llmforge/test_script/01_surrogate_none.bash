#!/bin/bash
# Test: sw=surrogate × hw=none. Pure analytical aux (params/kv/flops).
# Expected runtime: ~10s.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_01_surrogate_none_${TS}"

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC --acquisition_beta 1.0 \
    --hw_mode none --seq_len $SQL \
    --objectives val_loss params_M \
    --constraint val_loss=4.0 --constraint "params_M<=400"

assert_ckpt "$EXP" "$GENS"
pass "01_surrogate_none"
