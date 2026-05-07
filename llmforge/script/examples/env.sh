#!/bin/bash
# Source this file before running any of the production scripts in
# ../finetune_*.bash, ../realtrain_*.bash, etc.
#
#   source script/examples/env.sh
#
# These environment variables are read by the bash scripts (with safe
# fallbacks if unset) so the same scripts work on any host without
# hard-coded paths.

# Path to the Evo_GPT training package on the local host.
export EVO_GPT_DIR="${EVO_GPT_DIR:-$HOME/anon_llmforge/evo_gpt}"

# Path to the LLMForge search package on the local host.
export LLMFORGE_DIR="${LLMFORGE_DIR:-$HOME/anon_llmforge/llmforge}"

# SSH user / key used by the active-learning real-train trainer to dispatch
# jobs to the remote H100 hosts listed in --realtrain_hosts_file.
export USER="${USER:-anon}"
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"

# Conda environment name on remote hosts.
export CONDA_ENV="${CONDA_ENV:-llmforge}"

# Make Evo_GPT and llmforge importable.
export PYTHONPATH="$EVO_GPT_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
