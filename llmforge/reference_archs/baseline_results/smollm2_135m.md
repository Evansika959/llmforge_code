# SmolLM2-135M — baseline results

| artifact                   | path |
|----------------------------|------|
| Reference YAML             | [`../smollm2_135m.yaml`](../smollm2_135m.yaml) |
| **Machine-readable JSON**  | [`smollm2_135m.json`](smollm2_135m.json) |
| Combined runner            | [`../../bench_smollm2_baseline.py`](../../bench_smollm2_baseline.py) |
| ZEUS + train wrapper       | [`../../script/bench_smollm2_135m_baseline.bash`](../../script/bench_smollm2_135m_baseline.bash) |
| Train-only wrapper         | [`../../script/train_smollm2_135m_minipile.bash`](../../script/train_smollm2_135m_minipile.bash) |

> **Source of truth:** the JSON next to this file. Numbers below are
> snapshots quoted at write-time; treat the JSON as the live record.

## Headline numbers

| metric                    | value         |
|---------------------------|---------------|
| `val_loss` (minipile, 20k iters)        | **2.7794**    |
| `ttft_ms`                 |  236.23       |
| `tpot_ms`                 |  181.29       |
| `energy_per_token_uJ`     |  10,234,844   |
| `power_W`                 |  56.5         |
| `decode_energy_J`         |  2,620.12     |
| `zeus_kv_cache_used`      |  True         |
| Total params (built)      |  126.63 M     |
| Non-embedding (built)     |  97.35 M      |
| `params_M_est`            |  106.17       |

## ⚠ GPU-contention caveat on the HW numbers

The ZEUS measurements above were captured while another long-running
training process was actively sharing the same A100. Concurrent compute
serializes at the SM level, inflating wall-clock per-token by 5–7×. A
contention-free reference snapshot of the same model under the same
workload (captured earlier in the day) lands at:

| metric        | contention-free   | contended (current JSON) | ratio |
|---------------|-------------------|--------------------------|-------|
| `ttft_ms`     |  36.24            |  236.23                  | ×6.5  |
| `tpot_ms`     |  25.26            |  181.29                  | ×7.2  |
| `power_W`     |  58.8             |   56.5                   | ×0.96 |
| `E/tok_uJ`    |  1.48 M           |   10.23 M                | ×6.9  |
| `val_loss`    |  2.7794           |    2.7794 (injected)     | —     |

`power_W` barely changes — both runs see roughly the same instantaneous
GPU utilization while the kernel is on. Time × power = energy, so the
energy-per-token figure inflates with the time inflation.

If the JSON's `tpot_ms` is in the same ballpark as 25 ms, the GPU was
quiet during the run and the numbers are usable as a baseline. If it's
the 100+ ms regime, the GPU was busy — re-run when load is lower:

```bash
python bench_smollm2_baseline.py \
    --ref_yaml reference_archs/smollm2_135m.yaml \
    --exp_name smollm2_135m_baseline \
    --summary_json reference_archs/baseline_results/smollm2_135m.json \
    --prefill_len 256 --decode_len 256 --seq_len 512 \
    --zeus_n_repeats 3 --zeus_warmup 2 --zeus_dtype bf16 \
    --inject_val_loss 2.7794
```

Or — once the val_loss training is done elsewhere — the same wrapper
form via `script/bench_smollm2_135m_baseline.bash --no_train --inject_val_loss 2.7794`.

## Workload context

| field             | value         |
|-------------------|---------------|
| `prefill_len`     | 256           |
| `decode_len`      | 256           |
| `seq_len`         | 512           |
| `dtype`           | bf16          |
| `n_repeats`       | 3             |
| `warmup`          | 2             |
| `no_kv_cache`     | False (cached decode via `zeus_kv_cache.py`) |

## Architecture

| field                  | value     |
|------------------------|-----------|
| `n_embd`               | 576       |
| `block_size`           | 512       |
| `n_layer` (active)     | 30        |
| `n_head` per layer     | 9         |
| `n_kv_group` per layer | 3 (GQA 3:1) |
| `n_qk_head_dim`        | 64        |
| `n_v_head_dim`         | 64        |
| `mlp_size`             | 1,536     |
| `mlp_variant`          | swiglu    |
| `attention_variant`    | infinite  |

## val_loss provenance

`val_loss = 2.7794` is the result of training this exact arch on the
remote 8-host cluster via `script/train_smollm2_135m_minipile.bash`,
20,000 iterations on `minipile`. The number was injected into the JSON
via `--inject_val_loss 2.7794` (see `train.source` field in the JSON);
the cluster job itself was not re-run by `bench_smollm2_baseline.py`.

The trained model uses `_FIXED_MODEL_OVERRIDES` from
`remote_trainer.py` (SwiGLU + RoPE + no abs-pos + no bias + tied wte +
pre/peri-LN). Our ZEUS build now also uses SwiGLU
(`zeus_eval.build_model_from_individual` defaults to
`mlp_variant="swiglu"`); the remaining structural differences (RoPE vs
abs-pos, bias on/off, norm placement) do not affect parameter count
materially but do shift HW timing slightly. Closing those is a
straightforward follow-up if exact HW-vs-trained parity is required.

## Why ~127 M, not 135 M? (~9 M gap)

Inspecting `transformer.h.*.attn.c_proj.weight` shows it materializing
at shape `(n_v_head_dim=64) × (n_embd=576) = 0.037 M` instead of the
Llama-style `(n_head × n_v_head_dim = 576) × (n_embd = 576) = 0.332 M`.
That's because `zeus_eval.build_model_from_individual` doesn't propagate
`use_concat_heads` from the YAML into `GPTConfig`, so IHA falls through
to its `n_cproj == 1` branch.

Cost: ~0.295 M per layer × 30 layers ≈ **8.85 M** params underweighted
vs Llama. One-line fix in `build_model_from_individual`:

```python
cfg_kwargs = dict(
    ...,
    use_concat_heads=bool(g.get("use_concat_heads", True)),  # ← add
    mlp_variant=mlp_variant,
)
```

Out of scope for this baseline; flagged so the gap doesn't look
unexplained. Note `params_M_est` (106.17 M, attn+MLP only) does assume
`use_concat_heads=True`, so it represents what the model *would* report
once the propagation is fixed — that's why estimate (106.17) and
non-embedding measured (97.35) differ by ~9 M.
