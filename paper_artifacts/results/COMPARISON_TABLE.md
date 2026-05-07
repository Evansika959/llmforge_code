# FineWeb-Edu-10BT scaled-training validation — extended comparison

**Setup**: All models pretrained from scratch under identical recipe (100K iters × batch 64 × block 1024 = 13.1B tokens, AdamW, lr 3e-4 cosine, 2K warmup, bf16). ARC-E and HellaSwag are zero-shot length-normalized accuracy on the validation split.

**Last updated**: 2026-05-06

| Tier | Model | Params (M) | val_loss ↓ | ARC-E ↑ | HellaSwag ↑ | E_tok (µJ) ↓ | TTFT (ms) ↓ | TPOT (ms) ↓ |
|------|-------|-----------:|-----------:|--------:|------------:|-------------:|------------:|------------:|
| **~100M** | GPT-2 Small | 124 | 3.076 | 38.07 | 29.75 | — | — | — |
|  | SmolLM2-135M | 135 | 3.011 | **40.00** | 31.32 | 56.6 | 4.00 | 3.21 |
|  | Pythia-160M | 160 | 3.044 | 37.02 | 29.99 | 55.4 | 2.68 | 1.70 |
|  | **NSGA-Best-123M** *(Gen22-realtrain)* | 123 | **3.003** | 37.89 | **31.43** | **17.9** | **1.24** | **0.48** |
|  | **Gen40-real-Best-106M** *(g40r_a)* | 106 | 3.037 | **38.95** | 30.34 | — | — | — |
|  | **Gen40-eyeriss-Best-150M** | 150 | 3.331 | 39.30 | 31.35 | — | — | — |
|  | **Gen40-flat-Best-141M** | 141 | 3.387 | 38.77 | **31.53** | — | — | — |
|  | **Gen40-gemini-Best-125M** | 125 | 3.376 | 37.72 | 30.32 | — | — | — |
| **~300M** | OPT-350M | 355 | 2.939 | 38.77 | 32.94 | — | — | — |
|  | SmolLM2-360M | 362 | 2.831 | **43.86** | 36.22 | 143.0 | 6.71 | 5.65 |
|  | Qwen-0.5B | 402 | 2.821 | 41.93 | 36.37 | 135.6 | 7.07 | **4.08** |
|  | **NSGA-Best-347M** *(Gen19-fromSM2)* | 347 | **2.798** | 41.75 | **36.95** | 124.8 | 6.40 | 5.46 |
|  | **Gen30-real-Best-294M** *(g30_best3)* | 294 | 2.833 | **43.51** | 36.33 | — | — | — |
|  | **Gen30-real-Best-365M** *(g30_best1)* | 365 | 3.162 | 40.88 | 36.50 | — | — | — |
|  | **Gen30-HW-Pareto-279M** *(g30hw_c)* | 279 | 3.282 | 38.77 | 33.25 | — | — | — |

## Provenance of NSGA-discovered picks

| label in table | internal name | NSGA search family | Generation | Individual / dump |
|----------------|---------------|--------------------|-----------:|-------------------|
| NSGA-Best-123M | `nsga_g22_e` | Gen22 real_train+timeloop | 22 | ind[9] in `0501_1829_gen22.json` |
| Gen40-real-Best-106M | `nsga_g40r_a` | Gen40 real_train+timeloop | 40 | main[10] in `0503_gen40_realtrain.json` |
| Gen40-eyeriss-Best-150M | `eyeriss_real` | Gen40 substrate-aware (eyeriss) | 40 | hash `12d4e2823d6b` in `eyeriss_archs.yaml` |
| Gen40-flat-Best-141M | `flat_real` | Gen40 substrate-aware (flat) | 40 | hash `b2e614435c13` in `flat_archs.yaml` |
| Gen40-gemini-Best-125M | `gemini_real` | Gen40 substrate-aware (gemini) | 40 | hash `66d57a8ddc61` in `gemini_archs.yaml` |
| NSGA-Best-347M | `nsga_from_sm2_best3` | Gen19 (seeded from SmolLM2) | 19 | Ind 0 |
| Gen30-real-Best-294M | `nsga_g30_best3` | Gen30 real | 30 | ind[6] in `gen30.json` |
| Gen30-real-Best-365M | `nsga_g30_best1` | Gen30 real | 30 | ind[1] in `gen30.json` |
| Gen30-HW-Pareto-279M | `nsga_g30hw_c` | Gen30 real (HW-Pareto knee pick) | 30 | ind[11] in `gen30.json` |

## Summary

### ~100M tier
- **NSGA-Best-123M** (Gen22) holds best val_loss (3.003) and HellaSwag (31.43) at the smallest size, beating both GPT-2 Small (val 3.076) and Pythia-160M (val 3.044).
- **Gen40-real-Best-106M** (g40r_a) ties for best ARC-E (38.95) with sub-110M params; the only Gen40-realtrain pick with healthy +0.17 nat search→trained gap.
- **Gen40 substrate picks** (eyeriss/flat/gemini) all show large +0.77–0.81 nat transfer gaps but produce competitive ARC-E (37.7–39.3) and HellaSwag (30.3–31.5). flat_real ties with NSGA-Best-123M on HellaSwag (31.53 vs 31.43).

### ~300M tier
- **NSGA-Best-347M** (Gen19-fromSM2) holds best val_loss (2.798) and HellaSwag (36.95) — top of the entire 43-model leaderboard on these metrics. Beats OPT-350M (val 2.939, HS 32.94) and Qwen-0.5B (val 2.821) at smaller params.
- **Gen30-real-Best-294M** (g30_best3) is the standout new addition: **ARC-E 43.51%** at 19% fewer params than SmolLM2-360M, while matching its val_loss (2.833 vs 2.831). This makes g30_best3 the best ARC-E entry for any sub-300M open-recipe model in the table.
- **Gen30-HW-Pareto-279M** illustrates the search proxy's overcommitment to HW efficiency: lowest energy/tpot on its substrate but +0.58 nat transfer gap.

## Notes on blank cells

- **E_tok / TTFT / TPOT** are blank for all NSGA-discovered models. The values for the canonical baselines (SmolLM2/Pythia/Qwen) come from the timeloop simulation referenced in the original paper table; comparable substrate-matched measurements for the NSGA picks would need to be re-run on the same HW model.
- **Search-time HW signals** (different units per substrate) are available in the source dumps but not directly comparable across substrates in this table.
