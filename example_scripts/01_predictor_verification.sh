#!/usr/bin/env bash
# Example 1: Predictor verification (reproduces Table 1 of the paper).
#
# Stages (each is skipped if its trained artefact already exists on disk):
#   1. Train Forge-Former on the bundled IHA dataset (dataset_200M.csv).
#   2. Train Forge-Former on HW-GPT-Bench gpt_l (skipped automatically if
#      the gpt_l raw data is not on disk).
#   3. Fit and save random-forest and MLP baselines.
#   4. Render the comparison table (Markdown + LaTeX).
#
# Force a full retrain by deleting the corresponding checkpoint file, or
# by passing --refresh to table_a.py at the bottom.

set -euo pipefail

# Resolve paths relative to this script so it can be invoked from any cwd.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
EVO_GPT_DIR="$REPO_ROOT/evo_gpt"

export PYTHONPATH="$EVO_GPT_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
cd "$LLMFORGE_DIR"

# ── 1. Forge-Former on the IHA dataset ───────────────────────────────────
CUSTOM_CKPT=surrogate/ckpts/forgeformer.pt
CUSTOM_DATASET=surrogate/dataset/dataset_200M.csv
CUSTOM_SEED=100

if [[ ! -f "$CUSTOM_CKPT" ]]; then
    echo "[01] training Forge-Former on IHA dataset (seed=$CUSTOM_SEED)"
    echo "[01]   dataset: $CUSTOM_DATASET"
    echo "[01]   output:  $CUSTOM_CKPT"
    python -m surrogate.train \
        --csv_paths "$CUSTOM_DATASET" \
        --save_path "$CUSTOM_CKPT" \
        --max_layers 40 \
        --d_model 64 --nhead 4 --num_layers 4 --dropout 0.2 \
        --epochs 200 --batch_size 32 --lr 1e-4 \
        --test_ratio 0.2 --seed "$CUSTOM_SEED"
else
    echo "[01] Forge-Former IHA checkpoint found: $CUSTOM_CKPT"
    echo "     delete it to retrain from scratch"
fi

# ── 2. Forge-Former on HW-GPT-Bench gpt_l (optional) ─────────────────────
HWGPT_SCALE=l
HWGPT_CKPT=paper_plot/predictor_verify/hw_gpt_bench/ckpts/forgeformer_hwgpt_${HWGPT_SCALE}.pt
HWGPT_SEED=7
HWGPT_TEST_RATIO=0.3
HWGPT_DATA_PATH="${HWGPT_DATA_PATH:-}"

if [[ -f "$HWGPT_CKPT" ]]; then
    echo "[02] HW-GPT-Bench checkpoint found: $HWGPT_CKPT"
elif [[ -z "$HWGPT_DATA_PATH" ]]; then
    echo "[02] HW-GPT-Bench data not configured (set HWGPT_DATA_PATH); skipping."
    echo "     The IHA-dataset row of Table 1 will still render correctly."
else
    echo "[02] training Forge-Former on HW-GPT-Bench gpt_${HWGPT_SCALE}"
    python paper_plot/predictor_verify/hw_gpt_bench/train_forgeformer.py \
        --scale "$HWGPT_SCALE" --seed "$HWGPT_SEED" --test_ratio "$HWGPT_TEST_RATIO" \
        --epochs 200 --batch_size 64 --lr 1e-4 \
        --d_model 256 --nhead 4 --num_layers 4 --dropout 0.2
fi

# ── 3. Baselines (RF + MLP) and 4. table render ──────────────────────────
echo "[03,04] fitting RF / MLP baselines and rendering table"
python paper_plot/predictor_verify/table_a.py

echo ""
echo "[01_predictor_verification] done."
echo "  Markdown table -> $LLMFORGE_DIR/paper_plot/predictor_verify/table_a.md"
echo "  LaTeX table    -> $LLMFORGE_DIR/paper_plot/predictor_verify/table_a.tex"
