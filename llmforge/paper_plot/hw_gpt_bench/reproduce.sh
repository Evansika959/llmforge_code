#!/usr/bin/env bash
# Reproduce HW-GPT-Bench gpt_l comparison: V1 (Ours, d=256) vs Paper-Net (MLP-128).
#
# Inputs (must exist):
#   $HOME/hw-gpt-bench/data_collection/gpt_datasets/gpt_l/stats.pkl
#
# Outputs (overwrite-safe):
#   hwgpt_l_v1_d256.pt          V1 checkpoint
#   hwgpt_l_preds_d256.npz      V1 predictions on test split
#   hwgpt_l_results_d256.csv    V1 + RF + sklearn-MLP held-out metrics
#   hwgpt_l_paper_net.pt        Paper-Net checkpoint
#   hwgpt_l_paper_preds.npz     Paper-Net predictions on test split
#   hwgpt_l_paper_baseline.csv  Paper-Net held-out metrics
#
# Both runs use identical splits (seed=7, test_ratio=0.3 → n_train=7000, n_test=3000).
# Total wallclock ≈ 45 min on a single H100 if launched in parallel.

set -e
cd "$(dirname "$0")"

# -- V1 (PerFieldMLPRanker, d=256, 4-layer/4-head, 4000 epochs, batch 1024, lr 6e-4) --
python -u run_v1_hwgpt.py l \
    --epochs 4000 --batch 1024 --lr 6e-4 \
    --d_model 256 --n_layer 4 --n_head 4 \
    --tag d256 &
V1_PID=$!

# -- Paper-Net (Net(num_layers=36, layer_size=128), 4000 epochs, batch 1024, lr 1e-3) --
python -u paper_baseline.py l &
PAPER_PID=$!

wait $V1_PID $PAPER_PID

# -- Print combined metrics --
python - <<'PY'
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
v1 = pd.read_csv("hwgpt_l_results_d256.csv")
paper = pd.read_csv("hwgpt_l_paper_baseline.csv")
print("\n=== HW-GPT-Bench gpt_l (n_test=3000, seed=7) ===")
print(pd.concat([paper, v1], ignore_index=True).to_string(index=False))
PY
