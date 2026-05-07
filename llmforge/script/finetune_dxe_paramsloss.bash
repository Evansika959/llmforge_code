#!/bin/bash
# Production NSGA-II co-search — DXE (relaxed) HW backend variant.
#   sw_mode = surrogate_finetune  (surrogate drives selection; every 5 gens
#                                  a batch of 8 archs is real-trained and
#                                  the surrogate is refit on the augmented
#                                  labels with 5:1 old:new buffer blend)
#   hw_mode = timeloop, substrate = dxe_relaxed
#                                 (Timeloop run against the DXE chip with
#                                  loosened mapper constraints — same chip
#                                  topology as `dxe`, but admits valid
#                                  mappings for arbitrary GEMM shapes from
#                                  our search space.  The strict `dxe`
#                                  alternative fails with "no valid mappings"
#                                  on output dims like 1152 = n_head·qk_dim
#                                  that don't tile cleanly into the 8×16×16
#                                  PE array.  `dxe_relaxed` is what the
#                                  legacy rDXE inner search uses for the
#                                  same reason (run_exp_hw.py:387).
#                                  Two-pass evaluation (prefill at
#                                  prefill_len, decode at decode_len),
#                                  metrics combined to per-token totals.)
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  ttft and tpot come from Timeloop's prefill_cycles and
# decode_cycles divided by an assumed 1 GHz clock; energy_per_token_uJ is
# the (prefill_energy + decode_energy * decode_len) / total_tokens average.
# Full per-arch breakdown — energy_uJ, cycles, total_ops,
# total_memory_accesses, fusion_saved_*, edp, prefill_energy_uJ /
# prefill_cycles, decode_energy_uJ / decode_cycles — is preserved in aux.
#
# Notes:
# - 4 objectives stretches NSGA-II's selection pressure.  pop_size=24 may
#   end up mostly Pareto-non-dominated by gen 10-15 (reduced selection
#   signal).  If exploration stalls, increase --pop_size and --offspring.
# - Timeloop caches per-GEMM-shape mappings aggressively, so after the
#   first ~20 unique shapes appear in the population, subsequent archs are
#   evaluated mostly out of cache.  Expect Timeloop overhead of ~10-30s/arch
#   for cold misses, near-instant for cache hits.
#
# Cost: 24 init + 48 offspring × 40 gens ≈ 1944 evaluations.  With ~50%
# cache hits, expect ~5-8h of Timeloop compute on top of the ~13h cluster
# baseline → total ~18-21h.
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_finetune_200m_paramsloss_dxe_relaxed"
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
    --hw_mode timeloop --timeloop_substrate dxe_relaxed \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    --save_offspring \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Fine-tuned surrogate ckpts: ckpts/${EXP_NAME}/surrogate/gen{N}.pt"
echo "Baseline surrogate at surrogate/ckpts/forgeformer.pt was NOT modified."
echo "DXE-relaxed metrics (ttft, tpot, energy_per_token_uJ, energy_uJ, cycles,"
echo "  prefill_*, decode_*, total_ops, edp) are recorded in each individual's aux."
