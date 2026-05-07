#!/bin/bash
# Production NSGA-II co-search — ZEUS HW backend, 4-objective HW-loss search.
#   sw_mode = surrogate_finetune  (surrogate drives selection; every 5 gens
#                                  a batch of 8 archs is real-trained and
#                                  the surrogate is refit on the augmented
#                                  labels with 5:1 old:new buffer blend)
#   hw_mode = zeus                (every individual is measured on the local
#                                  GPU via ZEUS — ttft, tpot, decode energy,
#                                  power.  Unlike *_paramsloss_zeus*, here
#                                  the HW metrics drive NSGA selection.)
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  Same objective set as the Timeloop variants (eyeriss /
# gemmini / dxe_relaxed / flat_edge), so Pareto fronts are directly
# comparable across HW substrates.
#
# Difference vs finetune_zeus_paramsloss.bash:
#   - paramsloss: objectives = (val_loss, params_M); HW metrics recorded
#                 in aux but invisible to NSGA selection.
#   - hwloss (this script): objectives = (val_loss, ttft, tpot,
#                 energy_per_token_uJ); HW metrics directly steer the
#                 search frontier alongside val_loss.
# Aux fields (ttft_ms, tpot_ms, session_e_per_tok_uJ, power_W,
# total_area_mm2, n_chips) are still preserved for downstream analysis.
#
# Reproducibility / data flow:
#   --seed 42 controls NSGA evolution. Remote training process re-seeds
#   itself each invocation so per-arch dataset shuffles vary across gens.
#   ZEUS measurements are local A100, cached per arch hash (no re-measure
#   when an arch survives elimination across gens).
#
# Notes:
# - 4 objectives stretches NSGA-II's selection pressure.  pop_size=24 may
#   end up mostly Pareto-non-dominated by gen 10-15.  Increase --pop_size
#   and --offspring if exploration stalls.
# - Cost: 24 init + 48 offspring × 40 gens ≈ 1944 evaluations.  ZEUS is
#   ~25-40 s/arch on this A100, mostly amortized by per-hash caching of
#   surviving archs across gens → ~6-8 h of GPU measurement, plus the
#   ~13 h cluster real-train baseline → total ~20-22 h.
# - Host conflict: --realtrain_hosts_file points at hosts_8instances.yaml.
#   Do NOT launch concurrently with another run that also uses the same
#   8-host file (e.g. the FLAT timeloop run); they'd compete for AL job
#   slots.  Use hosts_4instances.yaml or wait for the conflicting run to
#   finish before launching this one.
#
# Outputs:
#   ckpts/<EXP_NAME>/                — per-gen Population JSON ckpts
#                                       (aux carries ttft_ms, tpot_ms,
#                                        energy_per_token_uJ, power_W in
#                                        addition to the analytical fields)
#   ckpts/<EXP_NAME>/{ts}_offspring_gen{N}.json  — pre-elim offspring snapshot
#   ckpts/<EXP_NAME>/surrogate/      — fine-tuned surrogate gen{N}.pt + .json
#                                       (baseline ckpt is NEVER overwritten)
#   ckpts/<EXP_NAME>/al_payloads/    — per-event training YAML payloads
#   train/<EXP_NAME>/gen{N}/         — remote-trainer working dirs (CSV results)
#   logs/<EXP_NAME>_<ts>.log         — full stdout/stderr
# ─────────────────────────────────────────────────────────────────────────
# PREREQUISITES
# ─────────────────────────────────────────────────────────────────────────
# This is a *production* search recipe. Running it requires:
#   1. An 8-host H100 pool (one machine per concurrent active-learning real-
#      train). Hosts are listed in --realtrain_hosts_file (use a real hosts
#      yaml; the bundled script/examples/hosts_example.yaml has placeholders).
#   2. SSH access (--realtrain_user / --realtrain_ssh_key) from this machine
#      to every host in (1).
#   3. The conda env named in --realtrain_conda_env (default "llmforge")
#      pre-installed on every remote host with llmforge_train/requirements_*.txt.
#   4. The llmforge_train/ package present at --realtrain_remote_llmforge_train_dir
#      on every remote host (clone of the same repo; the trainer does
#      `git pull` before each AL event).
#   5. Timeloop + Accelergy installed locally (only for hw_mode=timeloop):
#        pip install timeloopfe accelergy
#      plus the timeloop-{mapper,model} CLI binaries — see README.md.
#   6. ZEUS energy-measurement library locally (only for hw_mode=zeus):
#        pip install zeus-ml
#
# For a self-contained variant that does not need a cluster or remote
# training (no AL events, surrogate stays static), see
#   example_scripts/06_local_substrate_search.bash
# which runs a comparable search using only the local Timeloop install.
# ─────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_finetune_200m_hwloss_zeus"
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
    --realtrain_remote_llmforge_train_dir ${LLMFORGE_TRAIN_DIR:-$HOME/llmforge_train} \
    --realtrain_max_iters 20000 \
    --realtrain_timeout 16000 \
    --realtrain_poll_interval 600 \
    --hw_mode zeus \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --zeus_n_repeats 1 --zeus_warmup 1 --zeus_dtype bf16 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    --save_offspring \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Fine-tuned surrogate ckpts: ckpts/${EXP_NAME}/surrogate/gen{N}.pt"
echo "Baseline surrogate at surrogate/ckpts/forgeformer.pt was NOT modified."
echo "ZEUS metrics (ttft, tpot, energy_per_token_uJ, ttft_ms, tpot_ms,"
echo "  session_e_per_tok_uJ, power_W) are recorded in each individual's aux."
