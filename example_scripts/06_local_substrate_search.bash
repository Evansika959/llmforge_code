#!/usr/bin/env bash
# Example 6: Local NSGA-II co-search on a Timeloop substrate, no cluster.
#
# Reproduces the structure of the production recipes
# (llmforge/script/finetune_<substrate>_paramsloss.bash) but with the active-
# learning real-train events disabled. The search drives entirely on the
# bundled Forge-Former checkpoint and on local Timeloop hardware-cost
# evaluation, so it needs only this single workstation. No remote H100
# pool, no SSH config, no conda env on remotes.
#
# Differences vs the production scripts:
#   * --sw_mode surrogate              (no surrogate fine-tune, no AL events)
#   * --realtrain_*  flags dropped     (no remote dispatch)
#   * --finetune_*   flags dropped     (no AL refits)
#   * --pop_size / --generations cut from (24, 40) to (12, 12) by default,
#     so the run finishes in ~1-2 hours of local Timeloop instead of ~18-24 h
#     of cluster compute. Override with POP_SIZE / GENERATIONS env vars.
#
# Supported substrates (must have a config under llmforge/hw_eval/arch/<name>):
#   gemmini, eyeriss, flat_edge, dxe_relaxed, simba, simba_edge
#
# Usage:
#   bash example_scripts/06_local_substrate_search.bash <substrate>
#   bash example_scripts/06_local_substrate_search.bash gemmini
#   POP_SIZE=8 GENERATIONS=6 bash example_scripts/06_local_substrate_search.bash eyeriss
#
# Outputs:
#   llmforge/ckpts/local_<substrate>_paramsloss/<ts>_ckpt_gen{N}.json
#   llmforge/logs/local_<substrate>_paramsloss_<ts>.log

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash $0 <substrate>" >&2
    echo "  e.g.: bash $0 gemmini" >&2
    exit 1
fi
SUBSTRATE="$1"

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
LLMFORGE_DIR="$REPO_ROOT/llmforge"
LLMFORGE_TRAIN_DIR="$REPO_ROOT/llmforge_train"

export PYTHONPATH="$LLMFORGE_TRAIN_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
cd "$LLMFORGE_DIR"

# Defaults: small enough to finish locally in a couple of hours.
POP_SIZE="${POP_SIZE:-12}"
OFFSPRING="${OFFSPRING:-12}"
GENERATIONS="${GENERATIONS:-12}"
PREFILL_LEN="${PREFILL_LEN:-128}"
DECODE_LEN="${DECODE_LEN:-32}"
SEQ_LEN="${SEQ_LEN:-256}"
SEED="${SEED:-42}"

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="local_${SUBSTRATE}_paramsloss"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

echo "[06] substrate=${SUBSTRATE}"
echo "[06] pop=${POP_SIZE} offspring=${OFFSPRING} gens=${GENERATIONS}"
echo "[06] prefill=${PREFILL_LEN} decode=${DECODE_LEN}"
echo "[06] log: $log"
echo "[06] (this is local-only -- no remote training, no AL events)"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed "$SEED" \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size "$POP_SIZE" --offspring "$OFFSPRING" --generations "$GENERATIONS" \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --sw_mode surrogate \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 5 --acquisition_beta 1.0 \
    --hw_mode timeloop --timeloop_substrate "$SUBSTRATE" \
    --prefill_len "$PREFILL_LEN" --decode_len "$DECODE_LEN" --seq_len "$SEQ_LEN" \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo ""
echo "[06_local_substrate_search] done."
echo "  Run dir: ckpts/${EXP_NAME}/"
echo "  Log:     ${log}"
echo ""
echo "Next steps for paper-scale reproduction:"
echo "  - With a remote H100 cluster, see llmforge/script/finetune_${SUBSTRATE}_paramsloss.bash"
echo "    for the full active-learning recipe (surrogate co-evolution + 8 H100s)."
