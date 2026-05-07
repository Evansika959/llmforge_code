#!/usr/bin/env bash
# Example 2: NSGA-II search smoke test.
#
# Runs a tiny end-to-end search (4 archs, 2 offspring, 1 generation,
# surrogate evaluator with no hardware backend) to confirm that the
# environment, search loop, and bundled Forge-Former checkpoint are all
# wired up. Completes in under a minute on CPU.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
LLMFORGE_TRAIN_DIR="$REPO_ROOT/llmforge_train"

export PYTHONPATH="$LLMFORGE_TRAIN_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
cd "$LLMFORGE_DIR"

EXP_NAME="example_search_smoke"
echo "[02] running tiny NSGA-II search (exp_name=$EXP_NAME)"
echo "     pop_size=4, offspring=2, generations=1, surrogate-only"

python run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 4 --offspring 2 --generations 1 \
    --sw_mode surrogate \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --hw_mode none \
    --objectives val_loss params_M \
    --constraint val_loss=3.8 \
    --seed 42

echo ""
echo "[02_search_smoke] done."
echo "  Generation checkpoints -> $LLMFORGE_DIR/ckpts/$EXP_NAME/"
echo ""
echo "Inspect the gen-0 and gen-1 JSON dumps to confirm:"
echo "  - 4 individuals each with a sampled IHA architecture spec,"
echo "  - val_loss predictions in roughly [2.5, 3.5],"
echo "  - params_M values in roughly [50, 300]."
