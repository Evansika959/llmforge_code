#!/bin/bash
# Test: KV-cache shim for ZEUS HW eval.
#
# Phase A — direct verification via `kv_cache_check.py`
#   1. parity     — cached vs uncached final logits agree (bf16 tol)
#   2. direction  — cached tpot/energy not slower than uncached on average
#   3. toggle     — HwZeus(use_kv_cache=True/False) flips zeus_kv_cache_used
#
# Phase B — integration verification via `run_cosearch.py`
#   4. parity-check flag fires and prints PASS at startup
#   5. cache ON  reports `[kv-cache used on N/N]` per gen
#   6. cache OFF reports `[kv-cache off (--no_kv_cache)]` per gen
#
# Expected runtime: ~60-90s on an A100.
source "$(dirname "$0")/_common.bash"
t_start

# ── Phase A: direct ────────────────────────────────────────────────────────
echo
echo "[phase A] direct verification (test_script/kv_cache_check.py)"
echo "------------------------------------------------------------"
python -u test_script/kv_cache_check.py

# ── Phase B: integration ───────────────────────────────────────────────────
EXP_ON="_test_11_kv_cache_on_${TS}"
EXP_OFF="_test_11_kv_cache_off_${TS}"

echo
echo "[phase B.1] run_cosearch.py with --kv_cache_parity_check + cache ON"
echo "------------------------------------------------------------"
run_test "$EXP_ON" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations 1 \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n 1 --acquisition_beta 0 \
    --hw_mode zeus --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --zeus_n_repeats 1 --zeus_warmup 1 --zeus_dtype bf16 \
    --kv_cache_parity_check --verbose \
    --objectives val_loss params_M tpot power_W \
    --constraint val_loss=4.0

LOG_ON="logs/${EXP_ON}.log"
if ! grep -q "ZEUS KV-cache mode: ON" "$LOG_ON"; then
    echo "[FAIL] $EXP_ON: missing 'ZEUS KV-cache mode: ON' banner"; exit 1
fi
if ! grep -q "\[kv-cache parity\] .* PASS" "$LOG_ON"; then
    echo "[FAIL] $EXP_ON: parity check did not print PASS"; exit 1
fi
if ! grep -q "kv-cache used on" "$LOG_ON"; then
    echo "[FAIL] $EXP_ON: per-gen 'kv-cache used on N/M' line missing"; exit 1
fi
assert_ckpt "$EXP_ON" "1"

echo
echo "[phase B.2] run_cosearch.py with --no_kv_cache"
echo "------------------------------------------------------------"
run_test "$EXP_OFF" \
    --search_space_config "$SEARCH_SPACE" \
    --max_layers $LMAX --min_layers $LMIN \
    --pop_size $POP --offspring $OFFS --generations 1 \
    --sw_mode surrogate --surrogate_ckpt "$SURROGATE_CKPT" \
    --mc_dropout_n 1 --acquisition_beta 0 \
    --hw_mode zeus --prefill_len $PFL --decode_len $DCL --seq_len $SQL \
    --zeus_n_repeats 1 --zeus_warmup 1 --zeus_dtype bf16 \
    --no_kv_cache --verbose \
    --objectives val_loss params_M tpot power_W \
    --constraint val_loss=4.0

LOG_OFF="logs/${EXP_OFF}.log"
if ! grep -q "ZEUS KV-cache mode: OFF (--no_kv_cache)" "$LOG_OFF"; then
    echo "[FAIL] $EXP_OFF: missing OFF banner"; exit 1
fi
if ! grep -q "kv-cache off (--no_kv_cache)" "$LOG_OFF"; then
    echo "[FAIL] $EXP_OFF: per-gen 'kv-cache off' summary missing"; exit 1
fi
if grep -q "\[kv\]" "$LOG_OFF"; then
    echo "[FAIL] $EXP_OFF: '[kv]' tag appeared while --no_kv_cache was set"; exit 1
fi
assert_ckpt "$EXP_OFF" "1"

pass "11_kv_cache"
