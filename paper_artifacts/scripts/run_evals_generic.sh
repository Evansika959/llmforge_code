#!/bin/bash
# Run ARC-E + BoolQ + HellaSwag for a generic model dir.
# Usage: bash run_evals_generic.sh <result_dir_name> <log_label>
set -u
RESULT_DIR="$1"; LABEL="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniconda3/envs/llmforge/bin:$PATH"

# Locate ckpt subdir (find first subdir containing ckpt.pt)
D=""
for sub in paper_artifacts/results/${RESULT_DIR}/*/; do
  if [ -f "$sub/ckpt.pt" ]; then D="${sub%/}"; break; fi
done
if [ -z "$D" ]; then echo "[$(date)] ERROR no ckpt found under paper_artifacts/results/${RESULT_DIR}/"; exit 1; fi

LOG="paper_artifacts/results/evals_${LABEL}.log"
echo "[$(date)] eval start for $LABEL ($D)" | tee -a "$LOG"
for bench in arc-easy boolq hellaswag; do
  out="$D/eval_${bench//-/_}.json"
  if [ -f "$out" ]; then
    echo "[$(date)] SKIP $LABEL $bench (exists)" | tee -a "$LOG"
    continue
  fi
  echo "[$(date)] EVAL $LABEL $bench" | tee -a "$LOG"
  python benchmarks/evaluate_custom_models.py \
    --out_dir "$D" --benchmark "$bench" --split validation \
    --output_json "$out" >> "$LOG" 2>&1
  acc=$(python3 -c "import json; print(json.load(open('$out'))[0]['accuracy'])" 2>/dev/null)
  echo "[$(date)] DONE $LABEL $bench = $acc" | tee -a "$LOG"
done
echo "[$(date)] all evals done for $LABEL" | tee -a "$LOG"
