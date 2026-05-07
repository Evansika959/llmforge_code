#!/usr/bin/env bash
# Multi-seed predictor-verification sweep (surrogate ablation).
#
# Differences from 01_predictor_verification.sh (single-seed):
#   * Trains a separate Forge-Former, RF, and MLP per seed for each dataset.
#   * Renders one Markdown / LaTeX table per seed.
#   * After all seeds finish, aggregates per-seed tables into a single
#     mean +/- std table for the paper.
#
# Why per-seed retraining matters
# -------------------------------
# The 80/20 split is reseeded for every run. ForgeFormer is retrained on each
# seed's train rows because its checkpoint path is keyed by the seed. The
# RF and MLP baselines must also be re-fit on those same rows. Reusing a
# baseline that was fit on a different seed's train rows produces silently
# inflated metrics, because most of the next seed's "test" rows will have
# been in the cached model's training set. table_a.py keys the baseline
# cache on the split seed for exactly this reason; this script never relies
# on cross-seed cache hits.
#
# Stages per seed (each artefact is skipped if already on disk; pass
# --refresh by deleting the matching ckpt to retrain):
#   1. Train Forge-Former on the bundled IHA dataset (dataset_200M.csv).
#   2. Train Forge-Former on HW-GPT-Bench gpt_l (5-field schema).
#   3. Fit + save RF and MLP baselines for both datasets (table_a.py).
#   4. Render the Markdown + LaTeX table for that seed.
# After the loop:
#   5. Aggregate the per-seed tables into a single mean +/- std table.
#
# Usage:
#   bash example_scripts/surrogate_ablation.bash
#   BASE_SEED=42 bash example_scripts/surrogate_ablation.bash
#   BASE_SEED=42 N_SEEDS=10 bash example_scripts/surrogate_ablation.bash
#   SEEDS="100 200 300" bash example_scripts/surrogate_ablation.bash
#
# Seed sweep:
#   By default the script derives N_SEEDS=5 reproducible seeds from a single
#   BASE_SEED (default 100) using numpy's default RNG. Pass SEEDS="..." to
#   override with an explicit space-separated list.

set -euo pipefail

# Resolve paths relative to this script so it can be invoked from any cwd.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
LLMFORGE_TRAIN_DIR="$REPO_ROOT/llmforge_train"

export PYTHONPATH="$LLMFORGE_TRAIN_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
cd "$LLMFORGE_DIR"

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

echo "[surrogate-ablation] seeds: $SEEDS"
echo "[surrogate-ablation] custom test_ratio: $TEST_RATIO"
echo "[surrogate-ablation] hwgpt scale=$HWGPT_SCALE test_ratio=$HWGPT_TEST_RATIO"

# ── Per-seed sweep ───────────────────────────────────────────────────────
for seed in $SEEDS; do
    echo
    echo "============================================================"
    echo "[surrogate-ablation] seed=$seed"
    echo "============================================================"

    custom_ckpt="$CUSTOM_CKPT_DIR/forgeformer_seed${seed}.pt"
    hwgpt_ckpt="$HWGPT_CKPT_DIR/forgeformer_hwgpt_${HWGPT_SCALE}_seed${seed}.pt"

    # 1. Forge-Former on the IHA dataset.
    if [[ ! -f "$custom_ckpt" ]]; then
        echo "[surrogate-ablation] training IHA Forge-Former (seed=$seed)"
        echo "[surrogate-ablation]   output: $custom_ckpt"
        python -m surrogate.train \
            --csv_paths "$CUSTOM_DATASET" \
            --save_path "$custom_ckpt" \
            --max_layers 40 \
            --d_model 64 --nhead 4 --num_layers 4 --dropout 0.2 \
            --epochs 200 --batch_size 32 --lr 1e-4 \
            --test_ratio "$TEST_RATIO" --seed "$seed"
    else
        echo "[surrogate-ablation] IHA ckpt found: $custom_ckpt"
        echo "                      delete to retrain"
    fi

    # 2. Forge-Former on HW-GPT-Bench gpt_<scale>.
    if [[ ! -f "$hwgpt_ckpt" ]]; then
        echo "[surrogate-ablation] training HW-GPT-Bench Forge-Former (seed=$seed)"
        echo "[surrogate-ablation]   output: $hwgpt_ckpt"
        python paper_plot/predictor_verify/hw_gpt_bench/train_forgeformer.py \
            --scale "$HWGPT_SCALE" --seed "$seed" --test_ratio "$HWGPT_TEST_RATIO" \
            --epochs 200 --batch_size 64 --lr 1e-4 \
            --d_model 256 --nhead 4 --num_layers 4 --dropout 0.2 \
            --save_path "$hwgpt_ckpt"
    else
        echo "[surrogate-ablation] HW-GPT-Bench ckpt found: $hwgpt_ckpt"
        echo "                      delete to retrain"
    fi

    # 3-4. Fit RF / MLP baselines per seed and render this seed's table.
    # table_a.py now keys the baseline cache on the seed, so this never
    # silently reuses a baseline fit on a different seed's train rows.
    out_md="$TABLES_DIR/table_a_seed${seed}.md"
    out_tex="$TABLES_DIR/table_a_seed${seed}.tex"
    echo "[surrogate-ablation] rendering table for seed=$seed"
    python paper_plot/predictor_verify/table_a.py \
        --custom_seed "$seed" --custom_test_ratio "$TEST_RATIO" \
        --custom_ckpt "$custom_ckpt" \
        --hwgpt_scale "$HWGPT_SCALE" --hwgpt_ckpt "$hwgpt_ckpt" \
        --out_md "$out_md" --out_tex "$out_tex"
done

# ── Aggregate across seeds ───────────────────────────────────────────────
echo
echo "============================================================"
echo "[surrogate-ablation] aggregating across seeds"
echo "============================================================"
python paper_plot/predictor_verify/aggregate_seed_tables.py \
    --in_glob "$TABLES_DIR/table_a_seed*.md" \
    --out_md  "$AGG_OUT_MD" \
    --out_tex "$AGG_OUT_TEX"

echo
echo "[surrogate-ablation] done"
echo "  per-seed tables -> $LLMFORGE_DIR/$TABLES_DIR/"
echo "  aggregated      -> $LLMFORGE_DIR/$AGG_OUT_MD"
echo "                  -> $LLMFORGE_DIR/$AGG_OUT_TEX"
