#!/bin/bash
# Production NSGA-II co-search — SmolLM2-135M seed × rDXE × real-train every arch.
#   sw_mode = real_train          (every individual gets a fresh remote
#                                  cluster training; NO surrogate, NO
#                                  fine-tune, NO MC-dropout. val_loss
#                                  comes straight from cluster CSVs.)
#   hw_mode = timeloop, substrate = rdxe
#                                 (rDXE inner chip co-search via HwRdxeInner.
#                                  For each candidate arch we sweep a
#                                  (mac_per_vac, max_chips, wmem_per_core_KB)
#                                  chip-design grid through Timeloop; the
#                                  inner Pareto front is preserved per-arch
#                                  under aux['chip_pareto'], and the best
#                                  chip's metrics — ttft, tpot,
#                                  energy_per_token_uJ, area_mm2, power_W,
#                                  n_chips — are promoted to top-level
#                                  aux for outer-NSGA selection.
#                                  Edge envelope filter (area<=800mm2,
#                                  power<=100W) is on by default.)
#   seed_arch = SmolLM2-135M       (HuggingFaceTB/SmolLM2-135M canonical
#                                  config — 30 layers, n_embd=576, n_head=9,
#                                  n_kv_group=3 GQA, mlp=1536, head_dim=64,
#                                  attention_variant=infinite. Initial
#                                  population is the seed + jittered copies.)
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  ttft/tpot are the selected-chip's prefill/decode latencies
# (seconds, derived from cycles / chip clock), energy_per_token_uJ averages
# prefill + decode energy across (prefill_len + decode_len) tokens.  The
# full per-arch chip Pareto and analytical fields (n_chips, area_mm2,
# power_W, mac_util_pct, etc.) are preserved in aux['chip_pareto'] and aux
# for downstream analysis.
#
# Difference vs realtrain_smollm2_135m_flat.bash:
#   - flat: single-substrate Timeloop on a fixed FLAT chip; ttft/tpot/E
#           are the metrics of THAT specific chip running this arch.
#   - rdxe (this script): inner chip-design co-search per arch; each arch
#           is paired with its best chip from a (mac/max_chips/wmem) grid.
#           ttft/tpot/E are the selected chip's metrics — i.e. the search
#           jointly co-explores arch space AND chip space.
#
# Cost model (real-train + rDXE inner sweep, both substantial):
#   - Per-arch real-train: ~max_iters / 240 minutes (≈42 min @ max_iters=10000).
#   - Per-arch rDXE inner sweep: ~1–3 min after the chip × shape cache warms
#     (the first ~20 unique GEMM shapes are cold and slower).
#   - HW eval and real-train are sequential per gen (real-train first, then
#     local rDXE), so total wall ≈ (N+M*G)/H × (per_arch_realtrain) +
#                                  (N+M*G) × (per_arch_rdxe_avg).
#   - Defaults below: 16 + 16*10 = 176 archs.  Real-train dominates:
#     176/8 × 42 min + 176 × 2 min ≈ 15.5 h + 6 h ≈ 21 h total.
#   - To go faster: same levers as the FLAT variant — drop --realtrain_max_iters
#     to 5000 (~7.5 h real-train), drop pop/offspring/gens, etc.
#
# Notes:
# - 4 objectives stretches NSGA-II's selection pressure even without
#   surrogate noise; smaller pop/offspring is fine because every label is
#   ground truth.
# - val_loss=3.8 is the same loss-budget cap used in the surrogate-driven
#   runs.  SmolLM2-135M baseline reaches val_loss≈2.78 — well under the cap.
# - host_3 (1.2.3.5) was unreachable as of 2026-04-30 — verify the
#   cluster is healthy before launching.  hosts_8instances.yaml is the
#   default; switch to hosts_4instances.yaml if you only have 4 hosts up.
# - rDXE caches Timeloop mappings per (chip_config × GEMM_shape).  Sharing
#   hw_eval/runs/rdxe/ across rDXE runs gives big speedups on later runs.
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_realtrain_smollm2_135m_rdxe"
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
echo "Per-gen Population JSONs (incl. offspring snapshots) in ckpts/${EXP_NAME}/."
echo "Real-train CSVs (one per gen, with N val_loss labels) in train/${EXP_NAME}/."
echo "rDXE metrics: top-level aux carries the selected chip's ttft, tpot,"
echo "  energy_per_token_uJ, total_area_mm2, power_W, n_chips, mac_util_pct;"
echo "  full per-arch inner Pareto preserved under aux['chip_pareto']."
