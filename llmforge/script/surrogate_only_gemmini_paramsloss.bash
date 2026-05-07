#!/bin/bash
# Production NSGA-II co-search — Gemmini HW backend, FIXED surrogate (no fine-tune).
#   sw_mode = surrogate            (frozen base surrogate; MC-dropout for σ
#                                  used in NSGA acquisition, but NEVER refit.
#                                  No remote real-train, no AL events.)
#   hw_mode = timeloop, substrate = gemmini
#                                 (Timeloop run against UC Berkeley's
#                                  Gemmini systolic-array accelerator config
#                                  in hw_eval/arch/system_gemmini.yaml. Two-pass
#                                  evaluation (prefill at prefill_len,
#                                  decode at decode_len) → per-token totals.)
#
# Counterpart of finetune_gemmini_paramsloss.bash with the AL machinery
# disabled.  Use this to ablate the contribution of fine-tuning vs the
# frozen surrogate alone — same NSGA hyperparams, same objectives, same
# constraints, same Gemmini substrate.  Compare HV trajectories and final
# Pareto fronts to quantify how much AL adds.
#
# Search target: 4-objective NSGA-II over
#   (val_loss, ttft, tpot, energy_per_token_uJ)
# all minimized.  Same convention as the *_finetune_*_paramsloss runs;
# ttft/tpot from prefill/decode cycles ÷ 1 GHz, energy_per_token_uJ from
# (prefill_energy + decode_energy * decode_len) / total_tokens.
#
# Important difference vs the fine-tune variant:
#   - val_loss is a frozen-surrogate prediction throughout.  No real labels
#     are ever fetched.  All val_loss-driven NSGA selection rests on the
#     base surrogate's accuracy on the search-space distribution.
#   - The remote H100 cluster is NOT touched.  Don't pass any
#     --realtrain_* flags; they're ignored when sw_mode=surrogate but
#     omitting them keeps the cmdline honest about what's running.
#
# Cost: 24 init + 48 offspring × 40 gens ≈ 1944 evaluations.  Per-arch cost
# is just (a) Timeloop on Gemmini (mostly cached after first ~20 unique
# shapes), (b) surrogate forward pass (cheap, 10 MC-dropout samples).  No
# remote-cluster training → expect ~5-8h of Timeloop compute total, no
# cluster wait → total wall-clock dominated by the local Timeloop.
set -e
cd "$(dirname "$0")/.."

ts="$(date +'%Y%m%d_%H%M%S')"
EXP_NAME="$(date +%Y%m%d)_surrogate_200m_paramsloss_gemmini"
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
    --sw_mode surrogate \
    --surrogate_ckpt surrogate/ckpts/forgeformer.pt \
    --mc_dropout_n 10 --acquisition_beta 1.0 \
    --hw_mode timeloop --timeloop_substrate gemmini \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --objectives val_loss ttft tpot energy_per_token_uJ \
    --constraint val_loss=3.8 \
    --save_offspring \
    2>&1 | tee -a "$log"

echo
echo "Done. Run dir: ckpts/${EXP_NAME}/"
echo "Log:          ${log}"
echo "Surrogate ckpt used (frozen, NEVER modified):"
echo "  surrogate/ckpts/forgeformer.pt"
echo "Gemmini metrics (ttft, tpot, energy_per_token_uJ, energy_uJ, cycles,"
echo "  prefill_*, decode_*, total_ops, edp) are recorded in each individual's aux."
