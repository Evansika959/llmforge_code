# LLMForge

Anonymous code release for the paper *LLMForge: Co-Evolving NAS with
Infinite-Head Attention for Edge LLMs*.

LLMForge is a hardware-aware neural architecture search framework with three
composable contributions: **Infinite-Head Attention (IHA)**, an attention
parameterization that decouples query heads, KV groups, and per-head Q/K and
V dimensions; **Forge-Former**, a Transformer-encoder accuracy surrogate; and
**Forge-DSE**, an NSGA-II design-space-exploration engine that pairs the
surrogate with a multi-backend hardware cost model and optionally co-evolves
the surrogate during search.

## Repository layout

```
llmforge/         Search engine, surrogate, evaluators, hardware configs.
llmforge_train/   Transformer training code with IHA support.
example_scripts/  Self-contained reproduction examples (run on one workstation).
paper_artifacts/  Per-model architecture specs and benchmark JSONs for the 9
                  FineWeb-Edu-10BT trained models reported in Table 2 of the
                  paper. Eval harness and provenance table are also bundled.
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the smoke-test example (verifies the surrogate + search loop in <1 min)
bash example_scripts/03_search_smoke.sh

# 3. Reproduce the predictor-verification table from the paper (Table 1)
bash example_scripts/01_predictor_verification.sh

# 4. Run the multi-seed NSGA-II search-strategy ablation, then render the figure
bash example_scripts/04_search_ablation_seed_sweep.bash
bash example_scripts/05_plot_search_ablation_multi_seed.bash

# 5. Local NSGA-II search on a Timeloop substrate (Gemmini / Eyeriss / FLAT / ...).
#    Requires Timeloop + Accelergy locally; see "Optional dependencies" below.
bash example_scripts/06_local_substrate_search.bash gemmini
```

Steps 1–4 are self-contained and do not need a remote cluster, ZEUS, or
Timeloop. Step 5 needs Timeloop + Accelergy locally. See
`example_scripts/README.md` for the full list. To launch the full
multi-substrate searches that produce the paper's Pareto fronts, use the
production scripts in `llmforge/script/finetune_*_paramsloss.bash` after
configuring a remote H100 cluster (each script's header lists the
prerequisites).

## Optional dependencies

The example scripts in steps 1–4 above need only the packages in
`requirements.txt`. The substrate search in step 5 and the production
scripts in `llmforge/script/` additionally require:

### Timeloop and Accelergy (only for `--hw_mode timeloop`)

[Timeloop](https://timeloop.csail.mit.edu/) is the analytical mapper used
to evaluate Gemmini, Eyeriss, FLAT, and the single-chip DXE substrate.
[Accelergy](https://accelergy.mit.edu/) provides the energy estimation
tables Timeloop reads from. The Python front-end `timeloopfe` wraps both.

Install paths:

```bash
# Python front-end + accelergy energy tables (PyPI)
pip install timeloopfe accelergy

# Timeloop CLI binaries (timeloop-mapper, timeloop-model, timeloop-metrics).
# The simplest path is the upstream Docker image:
docker pull mitdlh/timeloop-accelergy-pytorch:latest
# Or build from source: https://github.com/Accelergy-Project/timeloop-accelergy-tutorials

# Verify
which timeloop-mapper accelergy
python -c "import timeloopfe.v4 as tl; print('timeloopfe ok')"
```

If you see `ImportError: cannot import name 'get_energy' from 'model'` from
Accelergy when running a search, that is a `PYTHONPATH` collision between
this repo's `llmforge_train/model.py` and an Accelergy ADC plug-in. The
HW evaluator in `llmforge/hw_exp.py` clears `PYTHONPATH` for the Timeloop
subprocess to avoid this; if you reproduce the issue from a custom entry
point, pass `environment={"PYTHONPATH": ""}` to `timeloopfe.call_mapper`.

### ZEUS (only for `--hw_mode zeus`, GPU energy measurement)

```bash
pip install zeus-ml
```

ZEUS reads NVML energy counters on a local NVIDIA GPU. We tested on an
A100-SXM4-40GB with driver 550.90.07, CUDA 12.4, and PyTorch 2.6.0.

## Manual setup

```bash
export LLMFORGE_TRAIN_DIR=$(pwd)/llmforge_train
export PYTHONPATH=$LLMFORGE_TRAIN_DIR:$(pwd)/llmforge:$PYTHONPATH

# Train Forge-Former from the bundled labels
cd llmforge
python -m surrogate.train \
    --csv_paths surrogate/dataset/dataset_200M.csv \
    --save_path surrogate/ckpts/forgeformer.pt \
    --max_layers 40 \
    --d_model 64 --nhead 4 --num_layers 4 --dropout 0.2 \
    --epochs 200 --batch_size 32 --lr 1e-4 \
    --test_ratio 0.2 --seed 100

# Run an NSGA-II search with the Gemmini Timeloop substrate
bash script/finetune_gemmini_paramsloss.bash
```

## Reproducing the paper

| Paper artifact | Reproduce with |
|---|---|
| Forge-Former vs MLP / RF table (Table 1) | `bash llmforge/paper_plot/predictor_verify/reproduce.sh` |
| HW-GPT-Bench gpt_l comparison | `bash llmforge/paper_plot/hw_gpt_bench/reproduce.sh` |
| t-SNE embedding figure | `python llmforge/paper_plot/predictor_verify/embeddings_tsne.py` |
| Architectural fingerprint figure | `python llmforge/paper_plot/pareto_trends/pareto_trends.py` |
| Search-strategy ablation figure | `bash example_scripts/04_search_ablation_seed_sweep.bash && bash example_scripts/05_plot_search_ablation_multi_seed.bash` |
| Scaled-training validation table (Table 2) | See `paper_artifacts/README.md` — each of the 9 trained models has its `arch.yaml`, `best_val_loss_and_iter.txt`, and per-benchmark eval JSON (ARC-E, ARC-C, BoolQ, HellaSwag, SciQ) bundled. The full eval harness and a `COMPARISON_TABLE.md` are in `paper_artifacts/results/` and `paper_artifacts/scripts/`. |

The Forge-DSE searches that produce the multi-substrate Pareto fronts
are run by `llmforge/script/finetune_*_paramsloss.bash` (one script per
ZEUS / Gemmini / Eyeriss / FLAT substrate). The rDXE multi-chip ring-substrate
search and the rDXE simulator are described in the paper appendix and
are deferred to the supplementary material.

## What is included vs not

**Included:**
- The full NSGA-II search engine (`llmforge/run_cosearch.py` and
  `llmforge/evaluators/*`).
- Forge-Former (encoder, training, fine-tuning, inference).
- Pretrained Forge-Former checkpoint (`llmforge/surrogate/ckpts/forgeformer.pt`)
  and the labeled IHA training corpus (`llmforge/surrogate/dataset/`).
- Timeloop substrate configurations for ZEUS-paired GPU baseline, Gemmini,
  Eyeriss, FLAT, and a single-chip DXE building block
  (`llmforge/hw_eval/arch/`).
- Reference architecture YAMLs for SmolLM2-135M, SmolLM2-360M, Pythia-160M,
  Qwen-0.5B, GPT-2-small, and the LLMForge-discovered picks
  (`llmforge/reference_archs/`).
- Plot scripts under `llmforge/paper_plot/` that regenerate the
  predictor-verification, t-SNE, and architectural fingerprint figures.
- **Paper-artifact bundle** (`paper_artifacts/`) for the 9 FineWeb-Edu-10BT
  trained models in Table 2. Each model directory contains its `arch.yaml`
  spec, a `best_val_loss_and_iter.txt` summary, and per-benchmark eval JSONs
  (ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, SciQ). The aggregate
  `results/all_benchmarks_results.json`, the human-readable
  `results/COMPARISON_TABLE.md`, and the eval scripts under
  `scripts/` are also included. No model weights are bundled (they would
  exceed the proxy's size budget); the per-model `arch.yaml` is sufficient
  to retrain from scratch on a single H100.

**Not included (regenerable or external):**
- Timeloop and its dependencies. Install separately from
  https://timeloop.csail.mit.edu/ before running Backend-B Timeloop searches.
- ZEUS energy-measurement library. Install separately via `pip install zeus`
  before running Backend-A GPU measurements.
- Pretraining datasets (MiniPile, FineWeb-Edu-10BT). These are publicly
  available on the HuggingFace Hub; preparation scripts are in `llmforge_train/`.
- Per-generation NSGA checkpoints from the production search runs. Each search
  takes between 6 and 24 hours on a single A100 plus an 8-host H100 active
  learning pool. To reproduce a final Pareto front, run the corresponding
  `script/finetune_*_paramsloss.bash`.
- Multi-chip ring-substrate simulator. See the supplementary PDF for the
  parametric specification of that substrate.

## Compute requirements

- Forge-Former training: ~5 minutes on a single H100.
- Single Forge-DSE search with co-evolution (40 generations, 24 population):
  6 to 24 hours on one A100 plus eight H100s for active-learning real-trains.
- Pareto-front candidate retraining for the FineWeb-Edu-10BT validation
  table: ~24 hours per architecture on a single H100.

## License

This code is released for review purposes under the MIT License.

## Anonymity notice

This repository accompanies a double-blind peer-review submission and does
not include any author or affiliation identifying information. Comments,
issue trackers, and CI metadata were stripped during release preparation.
