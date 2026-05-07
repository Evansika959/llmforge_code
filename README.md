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
llmforge/        Search engine, surrogate, evaluators, hardware configs.
evo_gpt/         Transformer training code with IHA support.
example_scripts/ Self-contained reproduction examples (run on one workstation).
paper_figures/   Plot scripts for the figures and tables in the paper.
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the smoke-test example (verifies the surrogate + search loop in <1 min)
bash example_scripts/03_search_smoke.sh

# 3. Reproduce the predictor-verification table from the paper
bash example_scripts/01_predictor_verification.sh

# 4. Render the search-strategy ablation figure (after running the five
#    NSGA-II searches whose checkpoints feed it; see the script header)
bash example_scripts/02_search_strategy_ablation.bash
```

The two scripts above are self-contained and do not need a remote
cluster, ZEUS, or Timeloop. See `example_scripts/README.md` for the
full list. To launch the full multi-substrate searches that produce the
paper's Pareto fronts, use the production scripts in `llmforge/script/`
after configuring remote hosts (see `llmforge/script/examples/env.sh`).

## Manual setup

```bash
export EVO_GPT_DIR=$(pwd)/evo_gpt
export PYTHONPATH=$EVO_GPT_DIR:$(pwd)/llmforge:$PYTHONPATH

# Train Forge-Former from the bundled labels
cd llmforge
python -m surrogate.train \
    --csv_paths surrogate/dataset/dataset_200M.csv \
    --save_path surrogate/ckpts/forgeformer.pt \
    --max_layers 40 \
    --d_model 64 --nhead 4 --num_layers 4 --dropout 0.2 \
    --epochs 200 --batch_size 32 --lr 1e-4 \
    --test_ratio 0.2 --seed 100

# 4. Run an NSGA-II search with the Gemmini Timeloop substrate
bash script/finetune_gemmini_paramsloss.bash
```

## Reproducing the paper

| Paper artifact | Reproduce with |
|---|---|
| Forge-Former vs MLP / RF table | `bash llmforge/paper_plot/predictor_verify/reproduce.sh` |
| HW-GPT-Bench gpt_l comparison | `bash llmforge/paper_plot/hw_gpt_bench/reproduce.sh` |
| t-SNE embedding figure | `python llmforge/paper_plot/predictor_verify/embeddings_tsne.py` |
| Per-substrate Pareto fronts | `bash paper_figures/plot_substrate_pareto_4row.bash` |
| Architectural fingerprint figure | `python llmforge/paper_plot/pareto_trends/pareto_trends.py` |
| Per-substrate full search summaries | `bash paper_figures/plot_substrate_summaries.bash` |
| Search recipe ablations | `bash paper_figures/plot_ablations_combined.bash` |

The Forge-DSE searches that produce the multi-substrate Pareto fronts are
run by `llmforge/script/finetune_*_paramsloss.bash` (one script per
ZEUS / Gemmini / Eyeriss / FLAT substrate). The rDXE multi-chip ring-substrate
search and the rDXE simulator are described in the paper appendix and are
deferred to the supplementary material.

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
- Plot scripts that regenerate every figure in the paper.

**Not included (regenerable or external):**
- Timeloop and its dependencies. Install separately from
  https://timeloop.csail.mit.edu/ before running Backend-B Timeloop searches.
- ZEUS energy-measurement library. Install separately via `pip install zeus`
  before running Backend-A GPU measurements.
- Pretraining datasets (MiniPile, FineWeb-Edu-10BT). These are publicly
  available on the HuggingFace Hub; preparation scripts are in `evo_gpt/`.
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
