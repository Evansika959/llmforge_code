#!/bin/bash
# Production NSGA-II co-search — SmolLM2-135M seed × FLAT × real-train every arch.
#   sw_mode = real_train          (every individual gets a fresh remote
#                                  cluster training; NO surrogate, NO
#                                  fine-tune, NO MC-dropout. val_loss
#                                  comes straight from cluster CSVs.)
#   hw_mode = timeloop, substrate = flat_edge
#                                 (Timeloop on the FLAT edge-LLM accelerator;
#                                  same two-pass prefill/decode evaluation
#                                  as the *_paramsloss_flat run.)
#   seed_arch = SmolLM2-135M       (HuggingFaceTB/SmolLM2-135M canonical
#                                  config — 30 layers, n_embd=576, n_head=9,
#                                  n_kv_group=3 GQA, mlp=1536, head_dim=64,
#                                  attention_variant=infinite. Initial
#                                  population is the seed + jittered copies.)
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  ttft and tpot are derived from prefill/decode cycles at
# 1 GHz (same convention as the *_paramsloss_flat run).  energy_per_token_uJ
# averages prefill + decode energy across (prefill_len + decode_len) tokens.
#
# Cost model (real-train every arch is the dominant cost):
#   - Each arch trains for --realtrain_max_iters on the remote cluster;
#     wall-clock per arch is ~max_iters / 240 minutes (≈4.2k iter/min on
#     an H100 for a 135M-class model).  At max_iters=10000 → ~42 min/arch.
#   - --pop_size N + --offspring M × --generations G archs total; with 8
#     parallel hosts each running sequentially, total wall is roughly
#     (N + M*G) / 8 * (per_arch_minutes).
#   - Defaults below: 16 + 16*10 = 176 archs × 42 min ÷ 8 hosts ≈ 15.5 h.
#   - To go faster: drop --realtrain_max_iters to 5000 (~21 min/arch,
#     ~7.5h total); to go cheaper still: also drop --offspring to 8 and
#     --generations to 5 (≈ 56 archs, ~5h).
#   - To match the *_paramsloss_flat budget you'd need ~24+48*40=1944
#     archs × 42 min ÷ 8 ≈ 170 h (~7 days). Don't.
#
# Notes:
# - 4 objectives stretches NSGA-II's selection pressure even without
#   surrogate noise; smaller pop/offspring is fine because every label is
#   ground truth.
# - val_loss=3.8 is the same loss-budget cap used in the surrogate-driven
#   runs.  SmolLM2-135M baseline trained on FineWeb-Edu (the cluster's
#   default) reaches val_loss≈2.78 → the seed itself is well below the
#   cap, so the constraint mostly culls aggressively-shrunk offspring.
# - host_3 (1.2.3.5) was unreachable as of 2026-04-30 — verify the
#   cluster is healthy before launching.  hosts_8instances.yaml is the
#   default; switch to hosts_4instances.yaml if you only have 4 hosts up.
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_realtrain_smollm2_135m_flat"
log="logs/${EXP_NAME}_${ts}.log"
mkdir -p logs "ckpts/${EXP_NAME}"

python -u run_cosearch.py \
    --exp_name "$EXP_NAME" \
    --log_dir logs \
    --seed 42 \
    --search_space_config search_space_def/smollm2_135m_space.yaml \
    --max_layers 40 --min_layers 8 \
    --seed_arch reference_archs/smollm2_135m.yaml \
    --seed_p_mlp 0.15 --seed_p_head 0.10 --seed_p_kv 0.10 \
    --seed_p_qk_dim 0.05 --seed_p_v_dim 0.05 --seed_p_identity 0.03 \
    --pop_size 16 --offspring 16 --generations 10 \
    --crossover_rate 0.6 --mutation_rate 0.3 \
    --sw_mode real_train \
    --realtrain_hosts_file script/examples/hosts_example.yaml \
    --realtrain_user ${USER:-anon} \
    --realtrain_ssh_key ~/.ssh/id_rsa \
    --realtrain_conda_env ${CONDA_ENV:-llmforge} \
    --realtrain_remote_evo_gpt_dir ${EVO_GPT_DIR:-$HOME/evo_gpt} \
    --realtrain_max_iters 10000 \
    --realtrain_timeout 8000 \
    --realtrain_poll_interval 300 \
    --hw_mode timeloop --timeloop_substrate flat_edge \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    --save_offspring \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Per-gen Population JSONs (incl. offspring snapshots) in ckpts/${EXP_NAME}/."
echo "Real-train CSVs (one per gen, with 8/N val_loss labels) in train/${EXP_NAME}/."
echo "FLAT metrics (ttft, tpot, energy_per_token_uJ, energy_uJ, cycles,"
echo "  prefill_*, decode_*, total_ops, edp) are recorded in each individual's aux."
