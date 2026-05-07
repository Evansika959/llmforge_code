# Reference-arch baseline results

This directory holds measured baseline metrics for each reference architecture
in `reference_archs/`. One JSON file per arch. Numbers are produced by
`bench_smollm2_baseline.py` (or `bench_hw_eval.py` for multi-platform runs)
and serve as anchor points the cosearch fronts can be compared against.

## ZEUS measurements (local A100)

Workload: `prefill=256, decode=256, bf16, KV-cache ON` (cached decode via `zeus_kv_cache.py`).
ttft / tpot in ms, E/tok in µJ, power in W.

| Arch                       | n_embd | n_layer | n_kv_group | mlp_size | mlp_variant     | ttft  | tpot  | E/tok    | power | val_loss   | Files                                                                                                                  |
|----------------------------|--------|---------|------------|----------|-----------------|-------|-------|----------|-------|------------|------------------------------------------------------------------------------------------------------------------------|
| Pythia-160M                | 768    | 12      | 12 (MHA)   | 3,072    | mlp (GeLU 2-mat) | 13.09 |  8.88 | 0.556 M  | 62.6  | —          | [.json](pythia_160m.json)                                                                                              |
| SmolLM2-135M               | 576    | 30      |  3 (3:1)   | 1,536    | swiglu (3-mat)   | 27.71 | 24.70 | 1.470 M  | 59.5  | **2.7794** | [.md](smollm2_135m.md) · [.json](smollm2_135m.json) · [hw_all.json](smollm2_135m_hw_all.json)                          |
| Qwen-2.5-7B-scaled         | 768    | 22      |  2 (6:1)   | 3,840    | swiglu           | 20.92 | 17.68 | 1.086 M  | 61.7  | **2.7071** | [.json](qwen2_5_7b_scaled.json)                                                                                        |
| LLaMA-2-7B-scaled          | 768    | 32      |  4 (3:1)   | 2,048    | swiglu           | 29.68 | 25.60 | 1.549 M  | 60.8  | **2.7296** | [.json](llama2_7b_scaled.json)                                                                                         |
| LLaMA-3-8B-scaled          | 768    | 30      |  3 (4:1)   | 2,560    | swiglu           | 25.43 | 24.30 | 1.453 M  | 60.2  | **2.7649** | [.json](llama3_8b_scaled.json)                                                                                         |
| SmolLM2-360M               | (not yet measured)                                                                                                                                                                                                              ||||||||||


8 DXE chips in a token-level pipeline, with each chip holding a subset
of layers' weights on-chip (WMEM bypass-DRAM). Sweeps
`(n_mac_per_vac, max_chips, wmem_per_core_KB)` per arch and selects the
envelope-feasible chip with min `per_tok_uJ`.

| Arch                  | ttft_ms | tpot_ms | E/tok (µJ) | n_chips | total_area_mm² | mac_util |
|-----------------------|--------:|--------:|-----------:|--------:|---------------:|---------:|
| **SmolLM2-135M**      |   **4.00** | **3.21** |     **56.58** |       8 |        454.72  | 33.2%   |
| Pythia-160M           | (rerun pending) | | | | | |
| Qwen-2.5-7B-scaled    | (rerun pending) | | | | | |
| LLaMA-2-7B-scaled     | (rerun pending) | | | | | |
| LLaMA-3-8B-scaled     | (rerun pending) | | | | | |

Selected config for SmolLM2-135M: `mac_per_vac=16, max_chips=8,
wmem_per_core=24 KB`, 17.6 mW @ 2,493 tok/s. Envelope-feasible (under
800 mm² area, 100 W power) with 33% MAC utilization. Compared to the
single-DXE `dxe_relaxed` (which has weights via DRAM and silent-zero
mitigations turned on):

| metric              | single dxe_relaxed | rDXE ring (8 chips) | improvement |
|---------------------|------------------:|--------------------:|------------:|
| ttft_ms             | 11.82             | 4.00                | **2.95×**   |
| tpot_ms             | 21.86             | 3.21                | **6.81×**   |
| energy_per_token_uJ | 2,876.12          | 56.58               | **50.83×**  |

n_head=6, d_ffn=1024) while we measure full-stack SmolLM2-135M
(30 layers, n_embd=576, ~126M params via SwiGLU). Per-layer-equivalent:
56.58 / 30 ≈ 1.89 µJ/token-layer, well below the paper's per-layer
implied number — consistent with rDXE's design intent for SLM decode.

## DXE-relaxed (single Timeloop substrate)

Same workload (`prefill=256, decode=256`). After the constraints + D-axis
padding patch (2026-05-01), `dxe_relaxed` produces honest numbers
without any special-case fusion-savings code. SmolLM2-135M is the only
one re-run on this branch so far.

| Arch                  | ttft_ms  | tpot_ms  | E/tok (µJ) | padded_op_count |
|-----------------------|---------:|---------:|-----------:|----------------:|
| **SmolLM2-135M**      |  **11.82** | **21.86** | **2,876.12** | **180** |
| Pythia-160M           | (rerun pending) | | | |
| Qwen-2.5-7B-scaled    | (rerun pending) | | | |
| LLaMA-2-7B-scaled     | (rerun pending) | | | |
| LLaMA-3-8B-scaled     | (rerun pending) | | | |

### What the patch does

**1. Updated DXE constraints YAMLs** (`hw_eval/arch/dxe_relaxed*/constraints.yaml`):
weights now flow through DRAM along with Inputs/Outputs (`keep:
[Inputs, Outputs, Weights]`, dropping the prior `bypass: [Weights]`
that modeled a hypothetical 3 MB-WMEM scale where all layer weights
fit on-chip — that assumption doesn't hold at 100M+ param baselines
since 100M × 2 bytes = 200 MB ≫ 3 MB). With weights through DRAM, raw
GEMM cycles include weight-fetch bandwidth and the silent-zero
`max(0, raw − saved)` clamp no longer trips.

**2. D-axis padding** (`hw_exp.py:_pad_D_for_arch`):
when `cfg.d_axis_spatial` is set (the four DXE variants, mesh = 8 DXT
× 16 VAC = 128), GEMM output dims `> 128 and not divisible by 128`
are rounded up to the next multiple. Sidecar `padding.json` is
dropped into the gemm work-dir for traceability; `padded_ops:
[[orig, padded], ...]` propagates up through `aggregate_stats` →
`evaluate_layer` → `evaluate_population` → `HwTimeloop.evaluate` to the
per-arch result; the headline JSON exposes `padded_op_count`. Cost:
~6–11% over-estimate on inflated GEMMs (SmolLM2-135M: Op 1 V_gen
192→256 +33%, Op 4/Op 6 ATTN_proj/MLP_FC2 576→640 +11%; 6 ops × 30
layers × {prefill, decode} = 90+90 = 180 unique padded ops).

`_estimate_saved_cycles` and `compute_fusion_savings` are unchanged
from their pre-patch form on this branch — no special-case code path
for DXE. Eyeriss/simba/gemmini/flat_edge produce **bit-identical**
numbers to pre-patch runs (verified against
`ckpts_paper/20260428_finetune_200m_paramsloss_eyeriss/0428_0630_ckpt_gen40.json`
ind[0]: every metric matched exactly).

### DXE numbers in context

`dxe_relaxed`'s `ttft_ms = 11.82` and `tpot_ms = 21.86` now sit in the
expected range relative to other Timeloop substrates:

- **prefill (ttft)**: 11.82 ms, between flat_edge (15.77 — high BW=25)
  and the others (53–125 ms — BW=4). Consistent with DXE's BW=4 plus
  some constant tile-overhead from the 8 DXT × 16 VAC mesh.
- **decode (tpot)**: 21.86 ms, comparable to eyeriss / simba / gemmini
  (~20.6 ms) — same DRAM bandwidth, similar attention/MLP shapes. Much
  higher than flat_edge (3.29 ms) because flat_edge has 6× higher BW.
- **energy/token**: 2,876 µJ, in line with the 2.7–3.1 K µJ bracket of
  the other 4 Timeloop substrates. Energy modeling was always
  substrate-agnostic; the new dxe_relaxed value lands where you'd

The two larger MAC-array variants (`dxe_relaxed_m32`, `dxe_relaxed_m64`)
got the same constraints update; their constraints YAMLs now also
have `keep: [Inputs, Outputs, Weights]`.

The "scaled" rows are scaled-down mappings of larger published models
into the cosearch n_embd=768 search space, mapped to preserve GQA ratio
and MLP ratio (see comments at the top of each YAML for the mapping
choices). val_loss for the three scaled archs comes from prior 20k-iter
training runs on minipile, injected into the JSON via
`--inject_val_loss`. SmolLM2-135M's val_loss comes from the same
training pipeline, run via `script/train_smollm2_135m_minipile.bash`.

## To add a new entry

1. Drop a reference YAML in `reference_archs/<name>.yaml` (use one of
   the existing files as a template — `mlp_variant: mlp` if the source
   model uses 2-matrix GeLU, otherwise omit and inherit the SwiGLU default).
2. Run ZEUS:
   ```bash
   python -u bench_smollm2_baseline.py \
       --ref_yaml reference_archs/<name>.yaml \
       --exp_name <name>_baseline \
       --summary_json reference_archs/baseline_results/<name>.json \
       --prefill_len 256 --decode_len 256 --seq_len 512 \
       --zeus_n_repeats 3 --zeus_warmup 2 --zeus_dtype bf16 \
       --inject_val_loss <known_val_loss>     # if you have one
   ```
3. Append a row to the table above.

Always tag entries with the workload (prefill/decode/dtype/kv_cache mode)
the ZEUS numbers came from — these knobs change absolute values
dramatically, so a number without context is not a number.
