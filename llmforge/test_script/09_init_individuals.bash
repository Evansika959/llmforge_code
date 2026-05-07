#!/bin/bash
# Test: --init_individuals population init (load N pre-built archs verbatim).
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_09_init_individuals_${TS}"

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode none --seq_len $SQL \
    --init_individuals test_script/fixtures/init_individuals.json \
    --objectives val_loss params_M \
    --constraint val_loss=4.0

assert_ckpt "$EXP" "$GENS"
pass "09_init_individuals"
