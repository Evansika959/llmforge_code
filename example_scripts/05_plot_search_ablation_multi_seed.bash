#!/usr/bin/env bash
# Example 5: Render the multi-seed search-strategy ablation HV figure.
#
# Reads checkpoint directories produced by example 04 (the seed sweep) and
# renders one HV trajectory panel with three overlaid curves, each plotted
# as mean +/- 1 sigma over the sweep's seeds:
#   * NSGA-II + IHA      (main recipe)
#   * Random  + IHA      (search-structure floor)
#   * NSGA-II + GQA      (IHA-expansion isolation, snapped to GQA-feasible)
#
# The script auto-discovers all matching ckpt directories under
# llmforge/ckpts/ that follow the seed-sweep naming convention:
#   <date>_ablation_<config>_paramsloss_seed<N>/
# (set NSGA_CKPT_ROOT to point at a different parent if needed).
#
# Usage:
#   bash example_scripts/05_plot_search_ablation_multi_seed.bash
#
#   # or with explicit override:
#   DIRS_NSGA_IHA="/p/seed42 /p/seed1" \
#   DIRS_RANDOM_IHA="/q/seed42 /q/seed1" \
#   DIRS_NSGA_GQA="/r/seed42 /r/seed1" \
#   bash example_scripts/05_plot_search_ablation_multi_seed.bash

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
EVO_GPT_DIR="$REPO_ROOT/evo_gpt"

export PYTHONPATH="$EVO_GPT_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"

NSGA_CKPT_ROOT="${NSGA_CKPT_ROOT:-$LLMFORGE_DIR/ckpts}"

# Auto-discover ckpt dirs per condition (sorted for reproducibility).
# Each condition's DIRS_* variable is a space-separated list of directories.
# `|| true` guards against pipefail when the glob has zero matches.
discover () {
    ls -d "$NSGA_CKPT_ROOT"/*ablation_"$1"_paramsloss_seed* 2>/dev/null | sort || true
}
DIRS_NSGA_IHA="${DIRS_NSGA_IHA:-$(discover nsga_iha)}"
DIRS_RANDOM_IHA="${DIRS_RANDOM_IHA:-$(discover random_iha)}"
DIRS_NSGA_GQA="${DIRS_NSGA_GQA:-$(discover nsga_gqa)}"

missing=0
for var in DIRS_NSGA_IHA DIRS_RANDOM_IHA DIRS_NSGA_GQA; do
    val="${!var}"
    if [[ -z "$val" ]]; then
        echo "ERROR: $var resolved to no directories." >&2
        echo "  NSGA_CKPT_ROOT=$NSGA_CKPT_ROOT" >&2
        echo "  Run example_scripts/04_search_ablation_seed_sweep.bash first." >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

# Convert space-separated lists to comma-separated form expected by the
# multi-seed plotter's --run "label:color:dir1,dir2,..." syntax.
to_csv () { tr ' ' '\n' | grep -v '^$' | paste -sd,; }
CSV_NSGA_IHA="$(echo "$DIRS_NSGA_IHA"  | to_csv)"
CSV_RANDOM_IHA="$(echo "$DIRS_RANDOM_IHA" | to_csv)"
CSV_NSGA_GQA="$(echo "$DIRS_NSGA_GQA"  | to_csv)"

OUT_DIR="${OUT_DIR:-$EXAMPLE_DIR/figs}"
mkdir -p "$OUT_DIR"
OUT_PDF="${OUT_PDF:-$OUT_DIR/search_ablation_hv.pdf}"

n_ihas=$(echo "$DIRS_NSGA_IHA"  | wc -w)
n_rnds=$(echo "$DIRS_RANDOM_IHA" | wc -w)
n_gqas=$(echo "$DIRS_NSGA_GQA"  | wc -w)

echo "[info] NSGA  + IHA  : $n_ihas seeds"
echo "[info] Random + IHA : $n_rnds seeds"
echo "[info] NSGA  + GQA  : $n_gqas seeds"
echo "[info] Output       : $OUT_PDF"

python3 "$EXAMPLE_DIR/_lib/plot_hv_compare_multi_seed.py" \
    --run "NSGA + IHA:#1f77b4:${CSV_NSGA_IHA}" \
    --run "Random + IHA:#888888:${CSV_RANDOM_IHA}" \
    --run "NSGA + GQA:#ff7f0e:${CSV_NSGA_GQA}" \
    --out "$OUT_PDF" \
    --obj-keys val_loss params_M

echo ""
echo "[05_plot_search_ablation_multi_seed] done."
echo "  Figure -> $OUT_PDF"
