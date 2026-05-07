#!/bin/bash
# Multi-platform HW eval for SmolLM2-135M.
#
# Default platforms (override with --hw if you want a different mix):
#   - zeus               (local A100 measurement, ~30s with KV cache)
#   - timeloop_eyeriss   (Eyeriss systolic array)
#   - timeloop_simba     (NVIDIA Simba edge accelerator)
#   - timeloop_gemmini   (UCB Gemmini)
#   - timeloop_flat_edge (flat 2-level edge accelerator)
#   - timeloop_dxe_relaxed (custom DXE chip, loosened mapper)
#
# Add `rdxe` for the chip-sweep ring evaluator (slow — many Timeloop subruns).
# Note: timeloop_dxe (strict mapper) is not in the registry — it rejects
# most search-space GEMM shapes. Use timeloop_dxe_relaxed for that chip.
#
# Workload: prefill=256, decode=256, seq_len=512, bf16. Same as the
# baseline ZEUS/training script.
#
# Output:
#   logs/${EXP}_${TS}.log
#   reference_archs/baseline_results/smollm2_135m_hw_all.json
set -e
cd "$(dirname "$0")/.."

EXP_NAME="smollm2_135m_hw_all"
ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/${EXP_NAME}_${ts}.log"
out_json="reference_archs/baseline_results/${EXP_NAME}.json"
mkdir -p logs

python -u bench_hw_eval.py \
    --ref_yaml reference_archs/smollm2_135m.yaml \
    --exp_name "$EXP_NAME" \
    --out_json "$out_json" \
    --hw zeus,timeloop_eyeriss,timeloop_simba,timeloop_gemmini,timeloop_flat_edge,timeloop_dxe_relaxed \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --zeus_n_repeats 3 --zeus_warmup 2 --zeus_dtype bf16 \
    "$@" \
    2>&1 | tee -a "$log"

echo
echo "Log:        $log"
echo "JSON ckpt:  $out_json"
