#!/bin/bash
# Test: --seed_arch population init (jittered neighbours of one ref arch).
# Validates the in-space-only check + the per-field jitter.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_08_seed_arch_${TS}"

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n $MC \
    --hw_mode none --seq_len $SQL \
    --seed_arch test_script/fixtures/seed_ref.yaml \
    --seed_p_mlp 0.4 --seed_p_head 0.4 \
    --objectives val_loss params_M \
    --constraint val_loss=4.0

assert_ckpt "$EXP" "$GENS"
pass "08_seed_arch"
