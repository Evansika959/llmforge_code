#!/bin/bash
# Visualize the final Pareto front of the ZEUS finetune co-search run
# (20260427_finetune_200m_paramsloss_zeus) via the interactive explorer.
#
# Picks the latest gen* checkpoint by generation number and renders an
# HTML scatter with dropdowns for x/y objective selection.  Click a
# Pareto point to see its full per-layer config + bar charts.
set -e
cd "$(dirname "$0")/.."

RUN_DIR="ckpts/20260427_finetune_200m_paramsloss_zeus"
OUT_DIR="htmls"
OUT_HTML="${OUT_DIR}/pareto_zeus_$(basename "$RUN_DIR").html"

# Pick the checkpoint with the highest gen number (handles the timestamp
# prefix change between gen35 and gen36 in this run).
CKPT="$(ls -1 "$RUN_DIR"/*_ckpt_gen*.json \
    | awk -F'gen' '{print $NF "\t" $0}' \
    | sort -n -k1 \
    | tail -1 \
    | cut -f2-)"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $RUN_DIR" >&2
    exit 1
fi

echo "Run dir : $RUN_DIR"
echo "Ckpt    : $CKPT"
echo "Output  : $OUT_HTML"

mkdir -p "$OUT_DIR"

python utils/pareto_front_explorer.py \
    --ckpt "$CKPT" \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --x val_loss --y energy_per_token_uJ \
    --output "$OUT_HTML" \
    --port 8000 \
    "$@"
