#!/bin/bash
# Test: sw=surrogate × hw=timeloop, substrate=rdxe (inner chip co-search).
# Most expensive — sweeps a chip-config grid per individual. Tiny pop=2,
# generations=0 to keep wall time bounded.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_05_surr_tl_rdxe_${TS}"
POP=${POP_OVERRIDE:-2}
OFFS=${OFFS_OVERRIDE:-2}
GENS=${GENS_OVERRIDE:-0}

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode timeloop --timeloop_substrate rdxe \
    --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --objectives val_loss params_M energy_per_token_uJ tpot \
    --constraint val_loss=4.0

assert_ckpt "$EXP" 0
pass "05_surrogate_timeloop_rdxe"
