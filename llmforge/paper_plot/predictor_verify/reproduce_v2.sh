#!/usr/bin/env bash
# Reproduces predictor-verify table over multiple seeds for credibility (v2).
#
# Differences from reproduce.sh (v1):
#   * Trains a separate ForgeFormer per seed for each dataset (custom + HW-GPT-Bench)
#     so that ranking metrics reflect predictor variability, not a single point.
#   * Renders a per-seed Markdown / LaTeX table per seed.
#   * After all seeds finish, aggregates per-seed tables into a single
#     mean +/- std table (table_a_aggregated.md / .tex) for the paper.
#
# Stages per seed (each artifact is skipped if already on disk; pass --refresh
# in your own re-run loop or delete the matching ckpt to retrain):
#   1. Train ForgeFormer on the custom dataset (IHA, dataset_200M.csv).
#   2. Train ForgeFormer on HW-GPT-Bench gpt_l (5-field schema).
#   3. Fit + save RF and MLP baselines for both datasets (table_a.py auto-fits).
#   4. Render Markdown + LaTeX tables for that seed.
# After the loop:
#   5. Aggregate the per-seed tables into a single mean +/- std table.
#
# Usage:
#   bash paper_plot/predictor_verify/reproduce_v2.sh
#   BASE_SEED=42 bash paper_plot/predictor_verify/reproduce_v2.sh
#   BASE_SEED=42 N_SEEDS=10 bash paper_plot/predictor_verify/reproduce_v2.sh
#   SEEDS="100 200 300" bash paper_plot/predictor_verify/reproduce_v2.sh
#   TEST_RATIO=0.2 HWGPT_TEST_RATIO=0.3 \
#       bash paper_plot/predictor_verify/reproduce_v2.sh
#
# Seed sweep:
#   By default the script derives N_SEEDS=5 reproducible seeds from a single
#   BASE_SEED (default 100) using numpy's default RNG. Pass SEEDS="..." to
#   override with an explicit space-separated list.
#
set -euo pipefail
cd "$(dirname "$0")/../.."   # → nsga_search/

# ── Configuration (overridable via environment) ──────────────────────────
BASE_SEED=${BASE_SEED:-100}
N_SEEDS=${N_SEEDS:-5}
# Derive N_SEEDS reproducible seeds from BASE_SEED unless an explicit
# SEEDS list was passed in.
if [[ -z "${SEEDS:-}" ]]; then
    SEEDS=$(python3 -c "
import numpy as np
rng = np.random.default_rng(${BASE_SEED})
seeds = rng.integers(1, 1_000_000, size=${N_SEEDS}).tolist()
print(' '.join(str(s) for s in seeds))
")
fi
TEST_RATIO=${TEST_RATIO:-0.2}
HWGPT_SCALE=${HWGPT_SCALE:-l}
HWGPT_TEST_RATIO=${HWGPT_TEST_RATIO:-0.3}

CUSTOM_DATASET=surrogate/dataset/dataset_200M.csv
CUSTOM_CKPT_DIR=paper_plot/predictor_verify/ckpts/custom_seeds
HWGPT_CKPT_DIR=paper_plot/predictor_verify/hw_gpt_bench/ckpts
TABLES_DIR=paper_plot/predictor_verify/per_seed_tables
AGG_OUT_MD=paper_plot/predictor_verify/table_a_aggregated.md
AGG_OUT_TEX=paper_plot/predictor_verify/table_a_aggregated.tex

mkdir -p "$CUSTOM_CKPT_DIR" "$HWGPT_CKPT_DIR" "$TABLES_DIR"

echo "[reproduce-v2] seeds: $SEEDS"
echo "[reproduce-v2] custom test_ratio: $TEST_RATIO"
echo "[reproduce-v2] hwgpt scale=$HWGPT_SCALE test_ratio=$HWGPT_TEST_RATIO"

# ── Per-seed sweep ───────────────────────────────────────────────────────
for seed in $SEEDS; do
    echo
    echo "============================================================"
    echo "[reproduce-v2] seed=$seed"
    echo "============================================================"

    custom_ckpt="$CUSTOM_CKPT_DIR/forgeformer_seed${seed}.pt"
    hwgpt_ckpt="$HWGPT_CKPT_DIR/forgeformer_hwgpt_${HWGPT_SCALE}_seed${seed}.pt"

    # 1. ForgeFormer on the custom (IHA) dataset.
    if [[ ! -f "$custom_ckpt" ]]; then
        echo "[reproduce-v2] training custom ForgeFormer (seed=$seed) → $custom_ckpt"
        python -m surrogate.train \
            --csv_paths "$CUSTOM_DATASET" \
            --save_path "$custom_ckpt" \
            --max_layers 40 \
            --d_model 64 --nhead 4 --num_layers 4 --dropout 0.2 \
            --epochs 200 --batch_size 32 --lr 1e-4 \
            --test_ratio "$TEST_RATIO" --seed "$seed"
    else
        echo "[reproduce-v2] custom ckpt found: $custom_ckpt (delete to retrain)"
    fi

    # 2. ForgeFormer on HW-GPT-Bench gpt_<scale>.
    if [[ ! -f "$hwgpt_ckpt" ]]; then
        echo "[reproduce-v2] training HW-GPT-Bench ForgeFormer (seed=$seed) → $hwgpt_ckpt"
        python paper_plot/predictor_verify/hw_gpt_bench/train_forgeformer.py \
            --scale "$HWGPT_SCALE" --seed "$seed" --test_ratio "$HWGPT_TEST_RATIO" \
            --epochs 200 --batch_size 64 --lr 1e-4 \
            --d_model 256 --nhead 4 --num_layers 4 --dropout 0.2 \
            --save_path "$hwgpt_ckpt"
    else
        echo "[reproduce-v2] hwgpt ckpt found: $hwgpt_ckpt (delete to retrain)"
    fi

    # 3-4. Render per-seed table.
    out_md="$TABLES_DIR/table_a_seed${seed}.md"
    out_tex="$TABLES_DIR/table_a_seed${seed}.tex"
    echo "[reproduce-v2] rendering table for seed=$seed"
    python paper_plot/predictor_verify/table_a.py \
        --custom_seed "$seed" --custom_test_ratio "$TEST_RATIO" \
        --custom_ckpt "$custom_ckpt" \
        --hwgpt_scale "$HWGPT_SCALE" --hwgpt_ckpt "$hwgpt_ckpt" \
        --out_md "$out_md" --out_tex "$out_tex"
done

# ── Aggregate across seeds ───────────────────────────────────────────────
echo
echo "============================================================"
echo "[reproduce-v2] aggregating across seeds"
echo "============================================================"
python paper_plot/predictor_verify/aggregate_seed_tables.py \
    --in_glob "$TABLES_DIR/table_a_seed*.md" \
    --out_md  "$AGG_OUT_MD" \
    --out_tex "$AGG_OUT_TEX"

echo
echo "[reproduce-v2] done"
echo "  per-seed tables → $TABLES_DIR/"
echo "  aggregated      → $AGG_OUT_MD"
echo "                  → $AGG_OUT_TEX"
