#!/bin/bash
# Production NSGA-II co-search — Eyeriss HW backend variant.
#   sw_mode = surrogate_finetune  (surrogate drives selection; every 5 gens
#                                  a batch of 8 archs is real-trained and
#                                  the surrogate is refit on the augmented
#                                  labels with 5:1 old:new buffer blend)
#   hw_mode = timeloop, substrate = eyeriss
#                                 (Timeloop run against the MIT Eyeriss v1
#                                  spatial dataflow accelerator config in
#                                  hw_eval/arch/eyeriss/arch.yaml.  Two-pass
#                                  evaluation (prefill at prefill_len,
#                                  decode at decode_len) → per-token totals.
#                                  Useful as a published-silicon baseline
#                                  alongside Gemmini and the in-house DXE
#                                  substrate.)
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  ttft and tpot are derived from prefill_cycles and
# decode_cycles assuming a 1 GHz clock (same convention as the dxe and
# gemmini variants — raw cycle counts are also in aux for re-clocking).
# energy_per_token_uJ averages prefill + decode energy across
# (prefill_len + decode_len) tokens.  Full per-arch breakdown — energy_uJ,
# cycles, total_ops, total_memory_accesses, fusion_saved_*, edp,
# prefill_*, decode_* — is preserved in aux for downstream analysis.
#
# Notes:
# - 4 objectives stretches NSGA-II's selection pressure.  pop_size=24 may
#   end up mostly Pareto-non-dominated by gen 10-15.  Increase --pop_size
#   and --offspring if exploration stalls.
# - Eyeriss has a relatively flexible mapper (row-stationary dataflow with
#   a generous mapping space), so unlike strict `dxe` it should map all
#   GEMM shapes from our search space cleanly.
#
# Cost: 24 init + 48 offspring × 40 gens ≈ 1944 evaluations.  With ~50%
# cache hits, expect ~5-8h of Timeloop compute on top of the ~13h cluster
# baseline → total ~18-21h.
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_finetune_200m_paramsloss_eyeriss"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/search_space_200M.yaml \
    --max_layers 40 --min_layers 8 \
    --pop_size 24 --offspring 48 --generations 40 \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --sw_mode surrogate_finetune \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 --acquisition_beta 1.0 \
    --finetune_every 5 --finetune_batch 8 \
    --finetune_base_csv surrogate/dataset/dataset_200M.csv \
    --finetune_old_to_new_ratio 5.0 \
    --realtrain_hosts_file script/examples/hosts_example.yaml \
    --realtrain_user ${USER:-anon} \
    --realtrain_ssh_key ~/.ssh/id_rsa \
    --realtrain_conda_env ${CONDA_ENV:-llmforge} \
    --realtrain_remote_evo_gpt_dir ${EVO_GPT_DIR:-$HOME/evo_gpt} \
    --realtrain_max_iters 20000 \
    --realtrain_timeout 16000 \
    --realtrain_poll_interval 600 \
    --hw_mode timeloop --timeloop_substrate eyeriss \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Fine-tuned surrogate ckpts: ckpts/${EXP_NAME}/surrogate/gen{N}.pt"
echo "Baseline surrogate at surrogate/ckpts/forgeformer.pt was NOT modified."
echo "Eyeriss metrics (ttft, tpot, energy_per_token_uJ, energy_uJ, cycles,"
echo "  prefill_*, decode_*, total_ops, edp) are recorded in each individual's aux."
