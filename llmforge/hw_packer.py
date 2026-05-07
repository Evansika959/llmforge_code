"""Layer-to-chip packing for rDXE chiplet ring topology.

Handles multi-resource feasibility (WMEM, KV cache, activation workspace)
and balanced contiguous partitioning for pipeline latency optimization.

Resources per chip:
  - WMEM: weight memory (static, holds all assigned layers' weights)
  - KV cache: per-layer KV entries × max context (for infinite attention)
  - Activation scratchpad: peak intermediate buffer during any single op

Packing strategy: contiguous balanced partitioning
  - Layers are assigned in order (no reordering) to preserve pipeline locality
  - Objective: minimize max per-chip decode time (TPOT)
  - Subject to: WMEM, KV cache, and scratchpad constraints per chip
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerCost:
    """Full resource profile for one active layer."""
    layer_idx: int
    attention_variant: str
    weight_bytes: int
    kv_bytes_per_token: int
    peak_activation_bytes: int
    decode_compute_ops: int
    prefill_compute_ops_per_token: int


@dataclass
class ChipSpec:
    """Chip resource budget."""
    wmem_bytes: int
    kv_cache_bytes: int
    n_cores: int
    total_macs: int
    scratchpad_per_core_bytes: int = 8192  # accumulator: 2048 entries × 4 bytes


@dataclass
class PackingResult:
    """Result of layer-to-chip assignment."""
    n_chips: int
    assignment: List[int]
    chip_layers: List[List[int]]
    feasible: bool
    infeasibility_reason: str = ""

    # Per-chip resource usage
    chip_weight_bytes: List[int] = field(default_factory=list)
    chip_kv_bytes: List[int] = field(default_factory=list)
    chip_decode_ops: List[int] = field(default_factory=list)
    chip_prefill_ops: List[int] = field(default_factory=list)

    # Balance metrics
    decode_balance: float = 0.0  # max/mean ratio (1.0 = perfect)
    prefill_balance: float = 0.0


def profile_layer(layer: dict, n_embd: int, max_context: int = 512) -> LayerCost:
    """Compute full resource profile for one layer."""
    nh = layer['n_head']
    nkv = layer['n_kv_group']
    qk = layer['n_qk_head_dim']
    vd = layer['n_v_head_dim']
    mlp = layer['mlp_size']
    attn = layer.get('attention_variant', 'infinite')

    # Weight bytes (INT8)
    if attn == 'infinite':
        weight_bytes = (n_embd * qk * (nh + nkv) +  # Q, K projections
                       n_embd * vd * nkv +            # V projection
                       vd * nh * n_embd +              # output projection
                       2 * n_embd * mlp)               # MLP FC1 + FC2
    else:  # identity
        weight_bytes = 2 * n_embd * mlp

    # KV cache per token (only for infinite attention)
    if attn == 'infinite':
        kv_bytes_per_token = nkv * (qk + vd)
    else:
        kv_bytes_per_token = 0

    # Peak activation bytes during any single sub-operation
    # For a GEMM(M, N, K): need input(M×K) + output(M×N) buffers
    # Peak is typically the largest intermediate activation
    if attn == 'infinite':
        # QK_gen output: seq × qk × (nh + nkv) -- largest projection
        # Attention scores: seq × seq × nh (can be huge for prefill!)
        # But DXE tiles this, so peak = one tile
        # Conservative: max of all projection output sizes
        peak_activation_bytes = max(
            n_embd + qk * (nh + nkv),   # QK_gen: input + output (per token)
            n_embd + vd * nkv,           # V_gen
            qk * nh + nh,               # QK_attn (per query token)
            vd * nh + n_embd,           # attn_proj output
            n_embd + mlp,               # MLP_FC1
            mlp + n_embd,               # MLP_FC2
        )
    else:
        peak_activation_bytes = max(n_embd + mlp, mlp + n_embd)

    # Compute ops for decode (single token, forward pass)
    if attn == 'infinite':
        # Projections: GEMV ops
        qk_gen_ops = n_embd * qk * (nh + nkv)
        v_gen_ops = n_embd * vd * nkv
        # Attention: dot product with KV cache (context length matters)
        qk_attn_ops = qk * max_context * nh  # Q @ K^T
        pv_attn_ops = max_context * vd * nh   # scores @ V
        attn_proj_ops = vd * nh * n_embd
        mlp_ops = 2 * n_embd * mlp
        decode_ops = qk_gen_ops + v_gen_ops + qk_attn_ops + pv_attn_ops + attn_proj_ops + mlp_ops
    else:
        decode_ops = 2 * n_embd * mlp

    # Compute ops for prefill (per token, amortized)
    # Same as decode but attention ops scale differently
    if attn == 'infinite':
        prefill_ops_per_token = (n_embd * qk * (nh + nkv) +
                                 n_embd * vd * nkv +
                                 qk * max_context * nh +
                                 max_context * vd * nh +
                                 vd * nh * n_embd +
                                 2 * n_embd * mlp)
    else:
        prefill_ops_per_token = 2 * n_embd * mlp

    return LayerCost(
        layer_idx=0,
        attention_variant=attn,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        peak_activation_bytes=peak_activation_bytes,
        decode_compute_ops=decode_ops,
        prefill_compute_ops_per_token=prefill_ops_per_token,
    )


def profile_model(individual: dict, max_context: int = 512) -> List[LayerCost]:
    """Profile all active layers in a model."""
    g = individual["globals"]
    layers = individual["layers"]
    mask = g.get("layer_mask", [True] * len(layers))
    n_embd = g["n_embd"]

    costs = []
    for li, (layer, m) in enumerate(zip(layers, mask)):
        if not m:
            continue
        cost = profile_layer(layer, n_embd, max_context)
        cost.layer_idx = li
        costs.append(cost)
    return costs


def check_chip_feasibility(layer_costs: List[LayerCost], chip: ChipSpec,
                           max_context: int = 512) -> Tuple[bool, str]:
    """Check if a set of layers can feasibly run on one chip."""
    total_weight = sum(lc.weight_bytes for lc in layer_costs)
    if total_weight > chip.wmem_bytes:
        return False, f"weights {total_weight/1e6:.2f}MB > WMEM {chip.wmem_bytes/1e6:.2f}MB"

    total_kv = sum(lc.kv_bytes_per_token * max_context for lc in layer_costs)
    if total_kv > chip.kv_cache_bytes:
        return False, f"KV cache {total_kv/1e6:.2f}MB > KV SRAM {chip.kv_cache_bytes/1e6:.2f}MB"

    # Per-operator scratchpad check: any single layer's peak activation must fit
    total_scratchpad = chip.scratchpad_per_core_bytes * chip.n_cores
    for lc in layer_costs:
        if lc.peak_activation_bytes > total_scratchpad:
            return False, (f"layer {lc.layer_idx} activation {lc.peak_activation_bytes/1e3:.1f}KB "
                          f"> scratchpad {total_scratchpad/1e3:.1f}KB")

    return True, ""


# ── Balanced contiguous partitioning ───────────────────────────────────────

def _greedy_contiguous_partition(layer_costs: List[LayerCost], chip: ChipSpec,
                                 max_stage_ops: int, max_context: int = 512
                                 ) -> Optional[List[List[int]]]:
    """Try to partition layers contiguously so no chip exceeds max_stage_ops decode ops.
    Returns list of layer index groups, or None if infeasible."""
    n = len(layer_costs)
    chips = [[]]
    chip_weight = 0
    chip_kv = 0
    chip_ops = 0

    for i in range(n):
        lc = layer_costs[i]
        new_weight = chip_weight + lc.weight_bytes
        new_kv = chip_kv + lc.kv_bytes_per_token * max_context
        new_ops = chip_ops + lc.decode_compute_ops

        fits_resources = (new_weight <= chip.wmem_bytes and
                         new_kv <= chip.kv_cache_bytes)
        fits_ops = (new_ops <= max_stage_ops)

        if fits_resources and fits_ops:
            chips[-1].append(i)
            chip_weight = new_weight
            chip_kv = new_kv
            chip_ops = new_ops
        else:
            # Start new chip
            chips.append([i])
            chip_weight = lc.weight_bytes
            chip_kv = lc.kv_bytes_per_token * max_context
            chip_ops = lc.decode_compute_ops

            # Check single layer feasibility
            if chip_weight > chip.wmem_bytes:
                return None
            if chip_kv > chip.kv_cache_bytes:
                return None

    return chips


def balanced_contiguous_pack(layer_costs: List[LayerCost], chip: ChipSpec,
                              max_context: int = 512) -> PackingResult:
    """Find the best contiguous partition that minimizes max per-chip decode time.

    Uses binary search on the max allowed per-chip decode ops, with greedy
    contiguous assignment as the feasibility check.
    """
    n = len(layer_costs)
    if n == 0:
        return PackingResult(n_chips=0, assignment=[], chip_layers=[], feasible=False,
                            infeasibility_reason="no layers")

    # Check if any single layer exceeds chip resources
    for lc in layer_costs:
        ok, reason = check_chip_feasibility([lc], chip, max_context)
        if not ok:
            return PackingResult(n_chips=0, assignment=[0]*n, chip_layers=[],
                               feasible=False, infeasibility_reason=reason)

    total_decode_ops = sum(lc.decode_compute_ops for lc in layer_costs)
    max_single_layer_ops = max(lc.decode_compute_ops for lc in layer_costs)

    # Binary search: find minimum max_stage_ops that produces a feasible partition
    lo = max_single_layer_ops
    hi = total_decode_ops
    best_partition = None

    while lo <= hi:
        mid = (lo + hi) // 2
        partition = _greedy_contiguous_partition(layer_costs, chip, mid, max_context)
        if partition is not None:
            best_partition = partition
            hi = mid - 1
        else:
            lo = mid + 1

    if best_partition is None:
        # Fallback: try with no ops limit (just resource constraints)
        best_partition = _greedy_contiguous_partition(
            layer_costs, chip, total_decode_ops * 2, max_context)

    if best_partition is None:
        return PackingResult(n_chips=0, assignment=[0]*n, chip_layers=[],
                           feasible=False, infeasibility_reason="cannot partition within resource constraints")

    # Build result
    n_chips = len(best_partition)
    assignment = [0] * n
    for ci, group in enumerate(best_partition):
        for li in group:
            assignment[li] = ci

    chip_weight = []
    chip_kv = []
    chip_decode = []
    chip_prefill = []
    for group in best_partition:
        group_costs = [layer_costs[i] for i in group]
        chip_weight.append(sum(lc.weight_bytes for lc in group_costs))
        chip_kv.append(sum(lc.kv_bytes_per_token * max_context for lc in group_costs))
        chip_decode.append(sum(lc.decode_compute_ops for lc in group_costs))
        chip_prefill.append(sum(lc.prefill_compute_ops_per_token for lc in group_costs))

    mean_decode = sum(chip_decode) / max(n_chips, 1)
    max_decode = max(chip_decode) if chip_decode else 0
    decode_balance = max_decode / mean_decode if mean_decode > 0 else 1.0

    mean_prefill = sum(chip_prefill) / max(n_chips, 1)
    max_prefill = max(chip_prefill) if chip_prefill else 0
    prefill_balance = max_prefill / mean_prefill if mean_prefill > 0 else 1.0

    return PackingResult(
        n_chips=n_chips,
        assignment=assignment,
        chip_layers=best_partition,
        feasible=True,
        chip_weight_bytes=chip_weight,
        chip_kv_bytes=chip_kv,
        chip_decode_ops=chip_decode,
        chip_prefill_ops=chip_prefill,
        decode_balance=decode_balance,
        prefill_balance=prefill_balance,
    )


# ── FFD packing (legacy, for comparison) ──────────────────────────────────

def greedy_ffd_pack(layer_costs: List[LayerCost], chip: ChipSpec,
                     max_context: int = 512) -> PackingResult:
    """First-fit decreasing by weight bytes. Non-contiguous. Minimizes chip count."""
    n = len(layer_costs)
    if n == 0:
        return PackingResult(n_chips=0, assignment=[], chip_layers=[], feasible=False)

    # Sort by weight descending
    sorted_indices = sorted(range(n), key=lambda i: -layer_costs[i].weight_bytes)
    assignment = [0] * n
    chip_remaining_wmem = []
    chip_remaining_kv = []
    chip_layer_lists = []

    for idx in sorted_indices:
        lc = layer_costs[idx]
        placed = False
        for ci in range(len(chip_remaining_wmem)):
            if (chip_remaining_wmem[ci] >= lc.weight_bytes and
                chip_remaining_kv[ci] >= lc.kv_bytes_per_token * max_context):
                assignment[idx] = ci
                chip_remaining_wmem[ci] -= lc.weight_bytes
                chip_remaining_kv[ci] -= lc.kv_bytes_per_token * max_context
                chip_layer_lists[ci].append(idx)
                placed = True
                break
        if not placed:
            if (lc.weight_bytes > chip.wmem_bytes or
                lc.kv_bytes_per_token * max_context > chip.kv_cache_bytes):
                return PackingResult(n_chips=0, assignment=[0]*n, chip_layers=[],
                                   feasible=False,
                                   infeasibility_reason=f"layer {lc.layer_idx} exceeds single chip")
            assignment[idx] = len(chip_remaining_wmem)
            chip_remaining_wmem.append(chip.wmem_bytes - lc.weight_bytes)
            chip_remaining_kv.append(chip.kv_cache_bytes - lc.kv_bytes_per_token * max_context)
            chip_layer_lists.append([idx])

    # Sort each chip's layers by original order
    for cl in chip_layer_lists:
        cl.sort()

    n_chips = len(chip_layer_lists)
    chip_decode = [sum(layer_costs[i].decode_compute_ops for i in group) for group in chip_layer_lists]
    chip_prefill = [sum(layer_costs[i].prefill_compute_ops_per_token for i in group) for group in chip_layer_lists]
    chip_weight = [sum(layer_costs[i].weight_bytes for i in group) for group in chip_layer_lists]
    chip_kv = [sum(layer_costs[i].kv_bytes_per_token * max_context for i in group) for group in chip_layer_lists]

    mean_d = sum(chip_decode) / max(n_chips, 1)
    max_d = max(chip_decode) if chip_decode else 0

    return PackingResult(
        n_chips=n_chips,
        assignment=assignment,
        chip_layers=chip_layer_lists,
        feasible=True,
        chip_weight_bytes=chip_weight,
        chip_kv_bytes=chip_kv,
        chip_decode_ops=chip_decode,
        chip_prefill_ops=chip_prefill,
        decode_balance=max_d / mean_d if mean_d > 0 else 1.0,
        prefill_balance=max(chip_prefill) / (sum(chip_prefill) / max(n_chips, 1)) if chip_prefill and sum(chip_prefill) > 0 else 1.0,
    )
