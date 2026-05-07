#!/bin/bash
# Test: sw=real_train × hw=none. Synchronous remote training oracle.
# Cluster-bound. realtrain_max_iters=50 keeps each H100 job ~30s; total
# wall time still 5-10 min from queue/setup overhead.
source "$(dirname "$0")/_common.bash"
t_start

EXP="_test_06_real_train_none_${TS}"
GENS=${GENS_OVERRIDE:-0}

run_test "$EXP" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations $GENS \
    --sw_mode real_train \
    --realtrain_hosts_file "$HOSTS_FILE" \
    --realtrain_max_iters $RT_ITERS --realtrain_timeout $RT_TIMEOUT \
    --realtrain_poll_interval $RT_POLL \
    --hw_mode none --seq_len $SQL \
    --objectives val_loss params_M \
    --constraint val_loss=99 --constraint "params_M<=400"

assert_ckpt "$EXP" 0
pass "06_real_train_none"
