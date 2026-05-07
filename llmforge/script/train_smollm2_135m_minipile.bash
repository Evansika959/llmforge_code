#!/bin/bash
# Real training of SmolLM2-135M on minipile (val_loss baseline).
#
# Recipe matches script/resume_finetune_zeus_paramsloss.bash:
#   - 20,000 iters
#   - dataset=minipile
#   - 8-host remote cluster (script/examples/hosts_example.yaml)
#   - llmforge conda env, ${EVO_GPT_DIR:-$HOME/evo_gpt}
#   - 16,000 s per-job timeout, 600 s poll interval
#
# Wall-clock: ~1-3 h depending on cluster load. Output is a single
# val_loss number plus the summary JSON.
#
# Outputs:
#   logs/smollm2_135m_minipile_<TS>.log
#   logs/smollm2_135m_minipile_<TS>.summary.json
set -e
cd "$(dirname "$0")/.."

EXP_NAME="smollm2_135m_minipile"
ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/${EXP_NAME}_${ts}.log"
summary_json="logs/${EXP_NAME}_${ts}.summary.json"
mkdir -p logs

python -u bench_smollm2_baseline.py \
    --ref_yaml reference_archs/smollm2_135m.yaml \
    --exp_name "$EXP_NAME" \
    --summary_json "$summary_json" \
    --no_zeus \
    --realtrain_hosts_file script/examples/hosts_example.yaml \
    --realtrain_user ${USER:-anon} \
    --realtrain_ssh_key ~/.ssh/id_rsa \
    --realtrain_conda_env ${CONDA_ENV:-llmforge} \
    --realtrain_remote_evo_gpt_dir ${EVO_GPT_DIR:-$HOME/evo_gpt} \
    --max_iters 20000 \
    --timeout 10000 \
    --poll_interval 600 \
    --dataset minipile \
    "$@" \
    2>&1 | tee -a "$log"

echo
echo "Log:        $log"
echo "Summary:    $summary_json"
