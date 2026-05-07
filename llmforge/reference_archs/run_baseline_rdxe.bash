#!/bin/bash
# Run rDXE HW evaluation on a single reference architecture YAML.
# Matches the rDXE NSGA search workload (prefill=512, decode=256, seq=768)
# so the numbers are apples-to-apples with NSGA-Best-347M's
# energy/TTFT/TPOT reported in Table 3 of the paper draft.
#
# Default target: Qwen-2.5-0.5B. Override with REF env var to evaluate
# any other reference architecture, e.g.:
#   REF=gpt2_small      bash run_baseline_rdxe.bash
#   REF=gemma_3_270m    bash run_baseline_rdxe.bash
#   REF=smollm2_360m    bash run_baseline_rdxe.bash
#
# rDXE envelope and selection criterion match the inner-search defaults
# used during the cosearch: select the chip Pareto point with the lowest
# per_tok_uJ that fits the (area<=800mm^2, power<=100W) envelope. The
# selected chip's metrics are promoted to top-level keys of
# headline.rdxe in the output JSON for direct table use.

set -e
cd "$(dirname "$0")/.."  # nsga_search/

REF="${REF:-qwen2_5_0_5b}"
REF_YAML="reference_archs/${REF}.yaml"
OUT_JSON="reference_archs/baseline_results/${REF}_rdxe.json"
EXP_NAME="${REF}_rdxe"

if [[ ! -f "${REF_YAML}" ]]; then
    echo "ERROR: reference YAML not found: ${REF_YAML}" >&2
    exit 1
fi
mkdir -p "$(dirname "${OUT_JSON}")"

echo "=== rDXE HW eval ==="
echo "  ref_yaml : ${REF_YAML}"
echo "  exp_name : ${EXP_NAME}"
echo "  out_json : ${OUT_JSON}"
echo "  workload : prefill=256, decode=256, seq=512  (matches rDXE NSGA search)"
echo

python3 -u bench_hw_eval.py \
    --ref_yaml      "${REF_YAML}" \
    --exp_name      "${EXP_NAME}" \
    --out_json      "${OUT_JSON}" \
    --hw            rdxe \
    --prefill_len   256 \
    --decode_len    256 \
    --seq_len       512 \
    --rdxe_select_by per_tok_uJ \
    --rdxe_area_max  800.0 \
    --rdxe_power_max 100.0 \
    --verbose

echo
echo "=== rdxe headline + selected chip config (extracted) ==="
python3 -c "
import json
d = json.load(open('${OUT_JSON}'))
h = d.get('headline', {}).get('rdxe', {})
hw = d.get('hw', {}).get('rdxe', {})
if not h and not hw:
    print('(no rdxe results; check errors block)')
    print(json.dumps(d.get('errors', {}), indent=2))
else:
    def fmt(val, spec):
        return format(val, spec) if isinstance(val, (int, float)) else '--'
    # headline metrics
    print(f'  energy_per_token_uJ : {fmt(h.get(\"energy_per_token_uJ\"), \".3f\")}')
    print(f'  ttft_ms             : {fmt(h.get(\"ttft_ms\"), \".3f\")}')
    print(f'  tpot_ms             : {fmt(h.get(\"tpot_ms\"), \".3f\")}')
    print(f'  power_W             : {fmt(h.get(\"power_W\"), \".4f\")}')
    # extra chip-config fields (from hw.rdxe, not headline)
    print(f'  total_area_mm2      : {fmt(hw.get(\"total_area_mm2\"), \".2f\")}')
    print(f'  n_chips             : {hw.get(\"n_chips\", \"--\")}')
    print(f'  mac_util_pct        : {fmt(hw.get(\"mac_util_pct\"), \".2f\")}')
    print(f'  selected_mac_per_vac: {hw.get(\"selected_mac_per_vac\", \"--\")}')
    print(f'  selected_max_chips  : {hw.get(\"selected_max_chips\", \"--\")}')
    print(f'  selected_wmem_KB    : {hw.get(\"selected_wmem_KB\", \"--\")}')
"
