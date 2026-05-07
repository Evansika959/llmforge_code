#!/bin/bash
# Example 2: Search-strategy ablation (Section 4.4 of the paper).
#
# Renders the combined ablation figure (two HV-curve panels):
#   (a) ZEUS substrate, 2-objective vs 4-objective NSGA selection.
#   (b) IHA-class search-space, NSGA + IHA vs Random + IHA vs NSGA + GQA.
#
# This script is the figure renderer. The five NSGA-II runs whose
# checkpoints feed the figure are produced by:
#   llmforge/script/finetune_zeus_paramsloss.bash      (panel a, 2-obj)
#   llmforge/script/finetune_zeus_hwloss.bash          (panel a, 4-obj)
#   llmforge/script/ablations/nsga_iha_paramsloss.bash    (panel b)
#   llmforge/script/ablations/random_iha_paramsloss.bash  (panel b)
#   llmforge/script/ablations/nsga_gqa_paramsloss.bash    (panel b)
#
# Each run writes its checkpoints to llmforge/ckpts/<date>_<exp_name>/.
# Override the auto-discovered defaults by exporting any of the env
# variables below before running this script.
#
# Usage:
#   bash example_scripts/02_search_strategy_ablation.bash
#
#   # or with explicit checkpoint locations:
#   CKPT_4OBJ_ZEUS=/path/to/ckpts \
#   CKPT_DIR_NSGA_IHA=/path/to/ckpts \
#   ...                                                        \
#   bash example_scripts/02_search_strategy_ablation.bash

set -e

# Resolve paths relative to this script so it can be invoked from any cwd.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
EVO_GPT_DIR="$REPO_ROOT/evo_gpt"
PAPER_FIG_DIR="$REPO_ROOT/paper_figures"

export PYTHONPATH="$EVO_GPT_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"

# Where to look for NSGA-II checkpoints.
NSGA_CKPT_ROOT="${NSGA_CKPT_ROOT:-$LLMFORGE_DIR/ckpts}"

# Panel (a): 2-obj vs 4-obj on ZEUS substrate.
CKPT_2OBJ_ZEUS="${CKPT_2OBJ_ZEUS:-$(ls -dt ${NSGA_CKPT_ROOT}/*finetune_200m_paramsloss_zeus 2>/dev/null | head -1)}"
CKPT_4OBJ_ZEUS="${CKPT_4OBJ_ZEUS:-$(ls -dt ${NSGA_CKPT_ROOT}/*finetune_200m_hwloss_zeus 2>/dev/null | head -1)}"

# Panel (b): search-strategy ablation.
CKPT_DIR_NSGA_IHA="${CKPT_DIR_NSGA_IHA:-$(ls -dt ${NSGA_CKPT_ROOT}/*ablation_nsga_iha_paramsloss 2>/dev/null | head -1)}"
CKPT_DIR_RANDOM_IHA="${CKPT_DIR_RANDOM_IHA:-$(ls -dt ${NSGA_CKPT_ROOT}/*ablation_random_iha_paramsloss 2>/dev/null | head -1)}"
CKPT_DIR_NSGA_GQA="${CKPT_DIR_NSGA_GQA:-$(ls -dt ${NSGA_CKPT_ROOT}/*ablation_nsga_gqa_paramsloss 2>/dev/null | head -1)}"

missing=0
for var in CKPT_2OBJ_ZEUS CKPT_4OBJ_ZEUS \
           CKPT_DIR_NSGA_IHA CKPT_DIR_RANDOM_IHA CKPT_DIR_NSGA_GQA; do
    val="${!var}"
    if [[ -z "$val" || ! -d "$val" ]]; then
        echo "ERROR: $var is unset or does not exist: $val" >&2
        echo "  Run the corresponding NSGA-II search first; see header comments." >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "" >&2
    echo "Hint: NSGA_CKPT_ROOT=$NSGA_CKPT_ROOT" >&2
    echo "Override with NSGA_CKPT_ROOT=/path/to/ckpts or set the per-panel" >&2
    echo "variables (CKPT_2OBJ_ZEUS, CKPT_4OBJ_ZEUS, CKPT_DIR_NSGA_IHA, ...)." >&2
    exit 1
fi

OUT_PDF="${OUT_PDF:-$EXAMPLE_DIR/ablations_combined.pdf}"

echo "[panel a] 2-obj ZEUS  : ${CKPT_2OBJ_ZEUS}"
echo "[panel a] 4-obj ZEUS  : ${CKPT_4OBJ_ZEUS}"
echo "[panel b] NSGA + IHA  : ${CKPT_DIR_NSGA_IHA}"
echo "[panel b] Random + IHA: ${CKPT_DIR_RANDOM_IHA}"
echo "[panel b] NSGA + GQA  : ${CKPT_DIR_NSGA_GQA}"
echo "[out]                 : ${OUT_PDF}"

python3 "$PAPER_FIG_DIR/plot_ablations_combined.py" \
    --ckpt_2obj_zeus   "${CKPT_2OBJ_ZEUS}" \
    --ckpt_4obj_zeus   "${CKPT_4OBJ_ZEUS}" \
    --ckpt_nsga_iha    "${CKPT_DIR_NSGA_IHA}" \
    --ckpt_random_iha  "${CKPT_DIR_RANDOM_IHA}" \
    --ckpt_nsga_gqa    "${CKPT_DIR_NSGA_GQA}" \
    --out              "${OUT_PDF}"

echo ""
echo "[02_search_strategy_ablation] done."
echo "  Figure -> $OUT_PDF"
