#!/bin/bash
# Test: sw=surrogate × hw=timeloop, substrate=dxe (custom).
# Same shape as 03 but the dxe substrate. Confirms substrate-dispatch routing.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_04_surr_tl_dxe_${TS}"
GENS=${GENS_OVERRIDE:-0}

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode timeloop --timeloop_substrate dxe \
    --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --objectives val_loss params_M energy_per_token_uJ token_delay \
    --constraint val_loss=4.0

assert_ckpt "$EXP" 0
pass "04_surrogate_timeloop_dxe"
