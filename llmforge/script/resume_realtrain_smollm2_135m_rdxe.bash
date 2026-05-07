#!/bin/bash
# Resume the SmolLM2-135M × rDXE × real-train co-search from gen 10 → gen 40.
#
# The original script (realtrain_smollm2_135m_rdxe.bash) ran 10 generations
# in ~7.5 h on 2026-05-01.  This script resumes from the gen-10 ckpt and
# runs 30 additional generations, leaving everything else identical:
#   - same EXP_NAME → all new ckpts (gen11..gen40) land alongside gen0..gen10
#     in ckpts/20260501_realtrain_smollm2_135m_rdxe/.  The run_time timestamp
#     prefix is regenerated (so the new ckpts are like 0501_1810_ckpt_gen11.json),
#     which is harmless — the loader picks the latest by gen number.
#   - same NSGA hyperparams (pop=16, offspring=16, crossover/mutation rates)
#   - same SW (real_train, max_iters=10000) and HW (rdxe) backends
#   - same 4 objectives + same 3 constraints
#
# Resume mechanics: --resume_ckpt restores the gen-10 population (16 archs +
# their evaluations + offspring buffer is empty post-elimination).  The init
# evaluation block is skipped (run_cosearch.py:529 checks `if not args.resume_ckpt`).
# The loop then runs `args.generations` times → set --generations 30 to land
# on gen 40.
#
# Cost: 30 gens × ~41 min/gen (8-host parallel real-train @ max_iters=10000) +
# trivial rDXE inner sweep (mostly cached from the first 10 gens) ≈ 20 h
# wall-clock.
set -e
cd "$(dirname "$0")/.."

# IMPORTANT: hard-coded to match the original 2026-05-01 run dir.  Do NOT
# regenerate from $(date +...) — that would point at a fresh dir and lose
# the gen0..gen10 history.
EXP_NAME="20260501_realtrain_smollm2_135m_rdxe"
RESUME_CKPT="ckpts/${EXP_NAME}/0501_0537_ckpt_gen10.json"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: resume ckpt not found: $RESUME_CKPT" >&2
    exit 1
fi

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/${EXP_NAME}_resume_${ts}.log"
mkdir -p logs

echo "Resuming from: $RESUME_CKPT"
echo "Log:           $log"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/smollm2_135m_space.yaml \
    --max_layers 40 --min_layers 8 \
    --resume_ckpt "$RESUME_CKPT" \
    --pop_size 16 --offspring 16 --generations 30 \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --sw_mode real_train \
    --realtrain_hosts_file script/examples/hosts_example.yaml \
    --realtrain_user ${USER:-anon} \
    --realtrain_ssh_key ~/.ssh/id_rsa \
    --realtrain_conda_env ${CONDA_ENV:-llmforge} \
    --realtrain_remote_llmforge_train_dir ${LLMFORGE_TRAIN_DIR:-$HOME/llmforge_train} \
    --realtrain_max_iters 10000 \
    --realtrain_timeout 10000 \
    --realtrain_poll_interval 300 \
    --hw_mode timeloop --timeloop_substrate rdxe \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    --constraint "total_area_mm2<=1000" \
    --constraint "total_area_mm2>=100" \
    --save_offspring \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Final ckpt:   ckpts/${EXP_NAME}/<ts>_ckpt_gen40.json"
echo "Combined Pareto over gen 0..gen 40: load all ckpt_gen{N}.json files in that dir."
