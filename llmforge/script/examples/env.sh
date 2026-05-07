#!/bin/bash
# Source this file before running any of the production scripts in
# ../finetune_*.bash, ../realtrain_*.bash, etc.
#
#   source script/examples/env.sh
#
# These environment variables are read by the bash scripts (with safe
# fallbacks if unset) so the same scripts work on any host without
# hard-coded paths.

# Path to the LLMForge training training package on the local host.
export LLMFORGE_TRAIN_DIR="${LLMFORGE_TRAIN_DIR:-$HOME/anon_llmforge/llmforge_train}"

# Path to the LLMForge search package on the local host.
export LLMFORGE_DIR="${LLMFORGE_DIR:-$HOME/anon_llmforge/llmforge}"

# SSH user / key used by the active-learning real-train trainer to dispatch
# jobs to the remote H100 hosts listed in --realtrain_hosts_file.
export USER="${USER:-anon}"
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"

# Conda environment name on remote hosts.
export CONDA_ENV="${CONDA_ENV:-llmforge}"

# Make LLMForge training and llmforge importable.
export PYTHONPATH="$LLMFORGE_TRAIN_DIR:$LLMFORGE_DIR:${PYTHONPATH:-}"
