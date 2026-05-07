# Paper artifacts — FineWeb-Edu-10BT trained models + eval suite

Self-contained bundle for reproducing Table 2 of the LLMForge paper. **GitHub-friendly**:
no large model weights — only architecture YAML configs, eval scripts, and per-model
benchmark JSON outputs (~400 KB total).

To re-train from these YAMLs and reproduce the reported numbers, see "Reproducing the
training" below.

## Layout

```
paper_artifacts/
├── README.md
├── results/
│   ├── all_benchmarks_results.json        # aggregate (9 models × 5 benchmarks + val_loss + metadata)
│   └── COMPARISON_TABLE.md                # human-readable extended table
├── scripts/
│   ├── evaluate_custom_models.py          # main eval harness (ARC-E/C, BoolQ, HellaSwag, etc.)
│   ├── eval_wg_sciq_fixed.py              # lm-eval-harness-compatible WG (per-choice ctx) + open-book SciQ
│   ├── evaluate_huggingface_models.py     # shared dataset loaders / extractors
│   └── run_evals_generic.sh               # convenience wrapper running ARC-E + BoolQ + HellaSwag sweep
└── models/
    ├── SmolLM2-135M/
    ├── Pythia-160M/
    ├── LLMForge-Acc-123M/
    ├── LLMForge-Compact-106M/
    ├── SmolLM2-360M/
    ├── Qwen-0.5B/
    ├── LLMForge-Acc-347M/
    ├── LLMForge-Eco-294M/
    └── LLMForge-Fast-365M/
        ├── arch.yaml                      # architecture spec (per-layer for NSGA-evolved; flat for baselines)
        ├── best_val_loss_and_iter.txt     # one-line summary: val_loss, iter, tokens, params, ...
        ├── eval_arc_easy.json             # ARC-Easy validation acc
        ├── eval_arc_challenge.json        # ARC-Challenge validation acc
        ├── eval_boolq.json                # BoolQ validation acc
        ├── eval_hellaswag.json            # HellaSwag validation acc
        └── eval_sciq.json                 # SciQ validation acc (open-book scoring with `support` passage)
```

## Reproducing the training

Each `models/<name>/arch.yaml` is a standalone architecture spec consumable by
`optimization_and_search/run_from_yaml.py` in the main `llmforge_train` repository.
Train any one of the 9 models from scratch:

```bash
# Example: re-train LLMForge-Compact-106M (the smallest NSGA pick)
python optimization_and_search/run_from_yaml.py \
    --yaml paper_artifacts/models/LLMForge-Compact-106M/arch.yaml \
    --output_dir my_results/LLMForge-Compact-106M \
    --prefix retrain \
    --dataset fineweb-edu-sample-10BT \
    --override_args \
        max_iters=100000 batch_size=64 eval_interval=2500 eval_iters=200 log_interval=100 \
        learning_rate=3e-4 min_lr=3e-5 decay_lr=true warmup_iters=2000 \
        grad_clip=1.0 dropout=0.0 always_save_checkpoint=true
```

Total cost per model: ~12 hours on a single NVIDIA H100 80 GB.
Final checkpoint contains 100K-iter weights; best-val checkpoint also retained.

## Reproducing the evaluations

The eval scripts in `scripts/` require the main `llmforge_train` repository to be on the Python
path (they import `model.GPT`, `sample.get_tokenizer_functions`, etc.). They auto-detect
the repo root by walking up from `__file__` looking for `model.py`, or you can override
with `LLMFORGE_REPO_ROOT`:

```bash
# After re-training (or with downloaded weights), evaluate any benchmark:
python paper_artifacts/scripts/evaluate_custom_models.py \
    --out_dir my_results/LLMForge-Compact-106M/retrain-row0 \
    --benchmark arc-easy --split validation \
    --output_json /tmp/test_eval.json

# For SciQ (open-book) and the corrected WinoGrande, use the fixed harness:
python paper_artifacts/scripts/eval_wg_sciq_fixed.py \
    --out_dir my_results/LLMForge-Compact-106M/retrain-row0 \
    --benchmark sciq --split validation \
    --output_json /tmp/test_sciq.json
```

## Scoring conventions

All benchmarks use **length-normalized log-likelihood (`acc_norm`)** scoring on the
validation split, with `block_size = 1024` and `bfloat16`.

| Benchmark | Examples | Choices | Random | Notes |
|-----------|---------:|--------:|-------:|-------|
| ARC-Easy | 570 | 4 | 25% | Easy school science |
| ARC-Challenge | 299 | 4 | 25% | Hard school science (near-random for sub-200M models) |
| BoolQ | 3,270 | 2 | 50% | `Passage:..\nQuestion:..\nAnswer:` + [" yes", " no"] |
| HellaSwag | 10,042 | 4 | 25% | 4-choice continuation |
| SciQ | 1,000 | 4 | 25% | **Open-book**: context = `support` passage + question (lm-eval-harness compatible) |

WinoGrande was evaluated under the lm-eval-harness convention (per-choice context
`before + option_i`, target = `after`) but **omitted from the table** — at-or-near
random (49–53%) for sub-1B models trained on 13.1B tokens, even with the fix.

## Training recipe (identical across all 9 models)

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.99, weight_decay=0.1) |
| Schedule | Cosine, 3e-4 → 3e-5 over 100K iters with 2K warmup |
| Batch | 64 × 1024 × 2 grad-accum = 131,072 tokens/step |
| Total tokens | 100K × 131,072 ≈ 13.1B |
| Precision | bfloat16 + `torch.compile` |
| Init | from scratch (no pretraining/finetuning) |
| Tokenizer | GPT-2 BPE (`tiktoken`), `vocab_size = 50257` |
| Dataset | FineWeb-Edu sample-10BT (Penedo et al., 2024) |
| Hardware | single NVIDIA H100 80 GB per training run |

NSGA-evolved architectures (LLMForge-* picks) use `mlp_variant=swiglu`, RMSNorm, RoPE,
peri-LN, GQA per-layer; the standard baselines preserve their canonical recipes.
The full per-layer architecture is in each model's `arch.yaml`.
