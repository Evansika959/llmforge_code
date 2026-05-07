# Example Experiments

Self-contained reproduction scripts that do not require remote-cluster
training, ZEUS GPU energy measurement, or Timeloop. Each script runs end
to end on a single workstation and writes its outputs back into the
repository tree.

## Examples

| Script | What it reproduces | Wall-clock |
|---|---|---|
| `01_predictor_verification.sh` | Forge-Former vs MLP / RF baselines on the IHA dataset and on HW-GPT-Bench `gpt_l`. Reproduces Table 1 of the paper. | ~30 min on one H100, ~2 h on CPU |
| `02_search_strategy_ablation.bash` | Combined ablation figure of Section 4.4: ZEUS 2-obj vs 4-obj NSGA, and IHA-class search-strategy comparison (NSGA + IHA / Random + IHA / NSGA + GQA). Renders the figure from existing NSGA-II checkpoints. | <1 min, after the five search runs that feed it have completed |
| `03_search_smoke.sh` | Tiny NSGA-II search (4 archs, 1 generation, surrogate-only, no HW backend). Verifies that the environment, search loop, and surrogate checkpoint are wired up correctly. | <1 min |
| `04_search_ablation_seed_sweep.bash` | Multi-seed sweep of the three search-strategy ablation conditions (NSGA + IHA, Random + IHA, NSGA + GQA), running each with `len(SEEDS)` independent NSGA-II runs. Output checkpoints feed `05_plot_search_ablation_multi_seed.bash`. | ~1-2 h on one A100 for 5 seeds x 3 conditions |
| `05_plot_search_ablation_multi_seed.bash` | Renders the multi-seed ablation HV figure. Each condition is plotted as the cross-seed mean trajectory with a shaded `+/-1 sigma` band, and the final-generation HV mean+/-std is printed to stdout. | <1 min, after `04_search_ablation_seed_sweep.bash` |
| `surrogate_ablation.bash` | Multi-seed extension of `01_predictor_verification.sh`. Retrains Forge-Former, RF, and MLP baselines for each seed and aggregates per-seed tables into a single mean +/- std summary. | ~N x (`01` wall-clock) |

Files under `_lib/` are Python helpers consumed by the example scripts;
they are not entry points themselves.

## Running

From the repository root:

```bash
# 1. Install dependencies.
pip install -r requirements.txt

# 2. Run an example.
bash example_scripts/01_predictor_verification.sh
```

Each script self-resolves paths and sets `PYTHONPATH` so it can be
invoked from any working directory.

`02_search_strategy_ablation.bash` requires NSGA-II checkpoints from five
production search runs. Run those first via the corresponding bash
scripts under `llmforge/script/` (see the script header for the mapping)
or override the auto-discovery with `CKPT_*` env vars.

## Dependencies that are NOT exercised by these examples

- Remote-cluster H100 hosts. Used only by the active-learning real-train
  evaluator in the production search recipes (`llmforge/script/finetune_*`).
- ZEUS energy measurement. Used only when `--hw_mode zeus` is set on the
  search driver.
- Timeloop. Used by the systolic substrate searches (Gemmini, Eyeriss,
  FLAT). Install separately from <https://timeloop.csail.mit.edu/> if you
  want to reproduce those Pareto fronts.
- HW-GPT-Bench raw data. The `01_predictor_verification.sh` script can
  optionally retrain on the HW-GPT-Bench `gpt_l` split if a local copy of
  the data is available; otherwise it skips that stage and reports only
  the IHA-dataset numbers.
