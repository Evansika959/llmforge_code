#!/bin/bash
# Test: sw=surrogate × hw=timeloop, substrate=eyeriss (published).
# Slow — Timeloop runs prefill+decode passes per individual.
# Expected runtime: 3-8 min depending on cache state. Use --generations 0 to
# skip the offspring round and only smoke gen-0 init.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_03_surr_tl_eyeriss_${TS}"
GENS=${GENS_OVERRIDE:-0}

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode timeloop --timeloop_substrate eyeriss \
    --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --objectives val_loss params_M energy_per_token_uJ token_delay \
    --constraint val_loss=4.0

assert_ckpt "$EXP" 0
pass "03_surrogate_timeloop_eyeriss"
