# hardware exploration by TimeLoop

from search_space import Individual
import yaml
import os
import math
import time
import logging
import timeloopfe.v4 as tl
from hw_eval.parse_timeloop_stats import parse_timeloop_stats, parse_dram_dataspace_stats
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress verbose timeloopfe / Specification / accelergy logs
for _lg in ("timeloopfe", "Specification", "accelergy"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

class _TimeloopFilter(logging.Filter):
    """Block root-logger messages from timeloopfe internals."""
    _KEYWORDS = ("Loading yaml file", "Found top-key", "Found extra top-key",
                 "Specification:", "Processor ", "parsed-processed",
                 "Parsing extra attributes", "Calculated Specification",
                 "Calling timeloop", "Calling Timeloop",
                 "Dataspace2BranchProcessor", "Branch ", "keeps {", "bypasses {")
    def filter(self, record):
        msg = record.getMessage()
        return not any(kw in msg for kw in self._KEYWORDS)

logging.getLogger().addFilter(_TimeloopFilter())

# ---------------------------------------------------------------------------
# Architecture configuration registry
# ---------------------------------------------------------------------------

_BASE = os.path.join(os.curdir, "hw_eval")

class ArchConfig:
    """Holds all file paths and parameters for a hardware architecture."""
    def __init__(self, name: str, arch_path: str, components_path: str,
                 constraints_path: str, variables_path: str,
                 mapper_path: str, runs_dir: str,
                 dram_read_bw: float = 4, dram_write_bw: float = 4,
                 d_axis_spatial: Optional[int] = None):
        self.name = name
        self.arch_path = arch_path
        self.components_path = components_path
        self.constraints_path = constraints_path
        self.variables_path = variables_path
        self.mapper_path = mapper_path
        self.runs_dir = runs_dir
        self.dram_read_bw = dram_read_bw
        self.dram_write_bw = dram_write_bw
        # When set, GEMM output dim D is rounded up to the next multiple
        # of `d_axis_spatial` if D > d_axis_spatial AND D % d_axis_spatial != 0.
        # Works around Timeloop's "residual ends not supported for Whoop
        # output" mapper failure on substrates with strict spatial-mesh
        # constraints (currently the four DXE variants, mesh = 8 DXT × 16 VAC = 128).
        # Other substrates (eyeriss / simba / gemmini / flat_edge) leave this
        # unset and the GEMM out_channel passes through unchanged.
        self.d_axis_spatial = d_axis_spatial

# Shared paths
_COMPONENTS = f"{_BASE}/arch/components/*.yaml"
_MAPPER = f"{_BASE}/mapper/mapper.yaml"
_PROBLEM_PATH = f"{_BASE}/prob/generic_GEMM.yaml"

ARCH_CONFIGS: Dict[str, ArchConfig] = {
    # Legacy Gemmini at 32nm (existing cached results)
    "gemmini": ArchConfig(
        name="gemmini",
        arch_path=f"{_BASE}/arch/system_gemmini.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/constraints/constraints.yaml",
        variables_path=f"{_BASE}/mapper/variables.yaml",
        mapper_path=_MAPPER,
        runs_dir=f"{_BASE}/runs",
    ),
    "eyeriss": ArchConfig(
        name="eyeriss",
        arch_path=f"{_BASE}/arch/eyeriss/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/eyeriss/constraints.yaml",
        variables_path=f"{_BASE}/arch/eyeriss/variables.yaml",
        mapper_path=_MAPPER,
        runs_dir=f"{_BASE}/runs/eyeriss",
    ),
    "simba": ArchConfig(
        name="simba",
        arch_path=f"{_BASE}/arch/simba/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/simba/constraints.yaml",
        variables_path=f"{_BASE}/arch/simba/variables.yaml",
        mapper_path=_MAPPER,
        runs_dir=f"{_BASE}/runs/simba",
    ),
    # FLAT-Edge: fused attention dataflow (Kao et al., ASPLOS 2023)
    "flat_edge": ArchConfig(
        name="flat_edge",
        arch_path=f"{_BASE}/arch/flat_edge/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/flat_edge/constraints.yaml",
        variables_path=f"{_BASE}/arch/flat_edge/variables.yaml",
        mapper_path=f"{_BASE}/arch/flat_edge/mapper.yaml",
        runs_dir=f"{_BASE}/runs/flat_edge",
        dram_read_bw=25,
        dram_write_bw=25,
    ),
    "simba_edge": ArchConfig(
        name="simba_edge",
        arch_path=f"{_BASE}/arch/simba_edge/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/simba_edge/constraints.yaml",
        variables_path=f"{_BASE}/arch/simba_edge/variables.yaml",
        mapper_path=_MAPPER,
        runs_dir=f"{_BASE}/runs/simba_edge",
    ),
    # DXE with relaxed constraints for general IHA GEMM evaluation
    "dxe_relaxed": ArchConfig(
        name="dxe_relaxed",
        arch_path=f"{_BASE}/arch/dxe_relaxed/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/dxe_relaxed/constraints.yaml",
        variables_path=f"{_BASE}/arch/dxe_relaxed/variables.yaml",
        mapper_path=f"{_BASE}/arch/dxe_relaxed/mapper.yaml",
        runs_dir=f"{_BASE}/runs/dxe_relaxed",
        dram_read_bw=4,
        dram_write_bw=4,
        d_axis_spatial=128,                # 8 DXT × 16 VAC; pads to avoid Whoop residual-ends
    ),
    # DXE relaxed with 2x mac_lane width (4096 MACs)
    "dxe_relaxed_m32": ArchConfig(
        name="dxe_relaxed_m32",
        arch_path=f"{_BASE}/arch/dxe_relaxed_m32/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/dxe_relaxed_m32/constraints.yaml",
        variables_path=f"{_BASE}/arch/dxe_relaxed_m32/variables.yaml",
        mapper_path=f"{_BASE}/arch/dxe_relaxed_m32/mapper.yaml",
        runs_dir=f"{_BASE}/runs/dxe_relaxed_m32",
        dram_read_bw=4,
        dram_write_bw=4,
        d_axis_spatial=128,                # same DXT/VAC mesh as dxe_relaxed
    ),
    # DXE relaxed with 4x mac_lane width (8192 MACs) — edge-NPU target
    "dxe_relaxed_m64": ArchConfig(
        name="dxe_relaxed_m64",
        arch_path=f"{_BASE}/arch/dxe_relaxed_m64/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/dxe_relaxed_m64/constraints.yaml",
        variables_path=f"{_BASE}/arch/dxe_relaxed_m64/variables.yaml",
        mapper_path=f"{_BASE}/arch/dxe_relaxed_m64/mapper.yaml",
        runs_dir=f"{_BASE}/runs/dxe_relaxed_m64",
        dram_read_bw=4,
        dram_write_bw=4,
        d_axis_spatial=128,                # same DXT/VAC mesh as dxe_relaxed
    ),
    "dxe": ArchConfig(
        name="dxe",
        arch_path=f"{_BASE}/arch/DXE/arch.yaml",
        components_path=_COMPONENTS,
        constraints_path=f"{_BASE}/arch/DXE/constraints.yaml",
        variables_path=f"{_BASE}/arch/DXE/variables.yaml",
        mapper_path=_MAPPER,
        runs_dir=f"{_BASE}/runs/dxe",
        dram_read_bw=2,
        dram_write_bw=2,
        d_axis_spatial=128,                # strict variant — pads same way as relaxed
    ),
}

DEFAULT_ARCH = "gemmini"

# Legacy aliases for backward compatibility
ARCH_PATH = ARCH_CONFIGS[DEFAULT_ARCH].arch_path
COMPONENTS_PATH = ARCH_CONFIGS[DEFAULT_ARCH].components_path
PROBLEM_PATH = _PROBLEM_PATH
MAPPER_PATH = ARCH_CONFIGS[DEFAULT_ARCH].mapper_path
CONSTRAINTS_PATH = ARCH_CONFIGS[DEFAULT_ARCH].constraints_path
VARIABLES_PATH = ARCH_CONFIGS[DEFAULT_ARCH].variables_path
DRAM_READ_BW = ARCH_CONFIGS[DEFAULT_ARCH].dram_read_bw
DRAM_WRITE_BW = ARCH_CONFIGS[DEFAULT_ARCH].dram_write_bw


def get_arch_config(arch: str = DEFAULT_ARCH) -> ArchConfig:
    if arch not in ARCH_CONFIGS:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(ARCH_CONFIGS.keys())}")
    return ARCH_CONFIGS[arch]


def _pad_D_for_arch(D: int, cfg: ArchConfig) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Round GEMM output dim D up to next multiple of `cfg.d_axis_spatial`
    when needed.

    Some substrates (the four DXE variants) hard-code spatial factors on
    the D axis (DXT × VAC = 128 on dxe_relaxed*); when the workload's D
    is `> mesh AND not divisible by mesh`, Timeloop's mapper raises
    "residual ends not supported for Whoop output" and the entire arch
    eval fails. For those substrates we round D up to the next multiple
    of the mesh — this overestimates compute by at most
    `(mesh − D mod mesh) / D` (~6–11% on common search-space shapes),
    which is much better than losing the data point entirely. For
    `D ≤ mesh` we pass through unchanged: the mapper handles small-D
    natively (low utilization, real cycles) and padding would inflate.
    Substrates without `d_axis_spatial` set (eyeriss / simba / gemmini /
    flat_edge) always pass through.

    Returns (D_to_use, padding_info or None). padding_info is
    `(D_orig, D_padded)` only when padding was actually applied, so
    callers can surface the inflation in audit fields.
    """
    mesh = getattr(cfg, "d_axis_spatial", None)
    if mesh and D > mesh and D % mesh != 0:
        D_pad = math.ceil(D / mesh) * mesh
        return D_pad, (D, D_pad)
    return D, None


def _prepare_gemm_spec(in_channel: int, out_channel: int, seq_length: int,
                       work_dir: str, cfg: ArchConfig) -> Tuple[str, str, Optional[Tuple[int, int]]]:
    """Prepare problem YAML and Timeloop spec files. Returns (out_dir, output_file, padding_info).

    `padding_info` is `(D_orig, D_padded)` when the substrate has a
    spatial-mesh constraint and the GEMM's output dim was rounded up,
    otherwise None. The on-disk gemm directory is named with the
    *padded* dim so the cache key stays consistent for the actual
    Timeloop run.
    """
    os.makedirs(work_dir, exist_ok=True)
    out_channel_used, padding_info = _pad_D_for_arch(out_channel, cfg)
    out_dir = os.path.join(work_dir, f"gemm_{in_channel}i_{out_channel_used}o_{seq_length}l")
    os.makedirs(out_dir, exist_ok=True)
    problem_file = os.path.join(out_dir, "generic_GEMM.yaml")
    with open(_PROBLEM_PATH, 'r') as f:
        problem_data = f.read()
        problem_data = problem_data.replace("$IN_CHANNELS", str(in_channel))
        problem_data = problem_data.replace("$OUT_CHANNELS", str(out_channel_used))
        problem_data = problem_data.replace("$OUT_HEIGHT", str(seq_length))
    with open(problem_file, 'w') as f:
        f.write(problem_data)
    if padding_info is not None:
        # Sidecar so the padding is auditable from disk for any cache hit.
        import json as _json
        with open(os.path.join(out_dir, "padding.json"), 'w') as f:
            _json.dump({"D_orig": padding_info[0], "D_padded": padding_info[1],
                        "mesh": cfg.d_axis_spatial, "arch": cfg.name}, f)
    return out_dir, problem_file, padding_info


def _run_mapper(out_dir: str, problem_file: str, cfg: ArchConfig,
                log_path: str = "/tmp/timeloop.log") -> str:
    """Run Timeloop mapper if not already cached. Returns stats output file path."""
    output_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(output_file):
        spec = tl.Specification.from_yaml_files(
            cfg.arch_path, cfg.components_path, cfg.mapper_path,
            problem_file, cfg.constraints_path, cfg.variables_path
        )
        spec.mapspace.template = 'uber'
        constrained_factors = ["D=1", "E=1"]
        tl.constraints.Factors(constrained_factors)
        if spec.constraints['targets'] is None:
            spec.constraints['targets'] = tl.constraints.ConstraintsList()

        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("")
        tl.call_mapper(spec, output_dir=out_dir, log_to=log_path)
    return output_file


def run_GEMM_evaluation(in_channel: int, out_channel: int, seq_length: int,
                        work_dir: str, log_path: str = "/tmp/timeloop.log",
                        arch: str = DEFAULT_ARCH) -> dict:
    cfg = get_arch_config(arch)
    # Use arch-specific runs directory unless work_dir was explicitly overridden
    if work_dir == "./hw_eval/runs":
        work_dir = cfg.runs_dir
    out_dir, problem_file, padding_info = _prepare_gemm_spec(
        in_channel, out_channel, seq_length, work_dir, cfg)
    output_file = _run_mapper(out_dir, problem_file, cfg, log_path)
    summary = parse_timeloop_stats(output_file)
    if padding_info is not None:
        summary['padded_D'] = list(padding_info)        # [orig, padded]
    return summary


def run_GEMM_evaluation_detailed(in_channel: int, out_channel: int, seq_length: int,
                                  work_dir: str, log_path: str = "/tmp/timeloop.log",
                                  arch: str = DEFAULT_ARCH) -> Tuple[dict, dict]:
    """Run GEMM evaluation and return both summary stats and per-dataspace DRAM stats.

    Returns:
        (summary_stats, dram_stats) where dram_stats is keyed by dataspace name
        ('Weights', 'Inputs', 'Outputs') with per-dataspace access counts and energy.
    """
    cfg = get_arch_config(arch)
    if work_dir == "./hw_eval/runs":
        work_dir = cfg.runs_dir
    out_dir, problem_file, padding_info = _prepare_gemm_spec(
        in_channel, out_channel, seq_length, work_dir, cfg)
    output_file = _run_mapper(out_dir, problem_file, cfg, log_path)
    summary = parse_timeloop_stats(output_file)
    dram = parse_dram_dataspace_stats(output_file)
    if padding_info is not None:
        summary['padded_D'] = list(padding_info)        # [orig, padded]
    return summary, dram


# ---------------------------------------------------------------------------
# Fusion savings calculation
# ---------------------------------------------------------------------------

def _dram_output_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Outputs dataspace of a GEMM.

    This is the energy for writing partial sums + reading back for reduction.
    In a fused chain, the producer's output stays on-chip, so this is saved.
    """
    out = dram_stats.get('Outputs', {})
    return out.get('energy_pJ', 0) or 0


def _dram_input_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Inputs dataspace of a GEMM.

    In a fused chain, the consumer reads its inputs from on-chip instead of DRAM.
    """
    inp = dram_stats.get('Inputs', {})
    return inp.get('energy_pJ', 0) or 0


def _dram_output_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar accesses for Outputs (reads + updates)."""
    out = dram_stats.get('Outputs', {})
    reads = out.get('scalar_reads', 0) or 0
    updates = out.get('scalar_updates', 0) or 0
    return reads + updates


def _dram_input_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar reads for Inputs."""
    inp = dram_stats.get('Inputs', {})
    return inp.get('scalar_reads', 0) or 0


def _estimate_saved_cycles(producer_dram: dict, consumer_dram: dict,
                           dram_read_bw: float = 4, dram_write_bw: float = 4) -> float:
    """Estimate cycle savings from avoiding DRAM round-trip for intermediate data."""
    out = producer_dram.get('Outputs', {})
    inp = consumer_dram.get('Inputs', {})

    out_writes = (out.get('scalar_updates', 0) or 0)
    out_reads = (out.get('scalar_reads', 0) or 0)
    inp_reads = (inp.get('scalar_reads', 0) or 0)

    saved_write_cycles = out_writes / dram_write_bw
    saved_read_cycles = (out_reads + inp_reads) / dram_read_bw

    return saved_write_cycles + saved_read_cycles


def compute_fusion_savings(
    op_stats: List[Tuple[dict, dict]],
    fusion_edges: List[Tuple[int, int]],
    scale_factors: Optional[Dict[int, float]] = None,
    dram_read_bw: float = 4,
    dram_write_bw: float = 4,
) -> Tuple[float, float]:
    """Compute energy and cycle savings from fusing consecutive operations.

    Args:
        op_stats: List of (summary_stats, dram_stats) per operation.
        fusion_edges: List of (producer_idx, consumer_idx) pairs defining
            which operations share intermediate data on-chip.
        scale_factors: Optional dict mapping op index to a scaling factor
            (e.g., for n_kv_groups scaling on QK_attn/PV_attn).
        dram_read_bw: DRAM read bandwidth in bytes/cycle.
        dram_write_bw: DRAM write bandwidth in bytes/cycle.

    Returns:
        (saved_energy_uJ, saved_cycles): Total savings from fusion.
    """
    if scale_factors is None:
        scale_factors = {}

    total_saved_energy_pJ = 0.0
    total_saved_cycles = 0.0

    # Compute total unfused energy to cap savings
    total_unfused_energy_pJ = 0.0
    for summary, _ in op_stats:
        e = summary.get('energy_uJ')
        if e is not None:
            total_unfused_energy_pJ += e * 1e6
    for idx, sc in scale_factors.items():
        e = op_stats[idx][0].get('energy_uJ')
        if e is not None:
            # Already counted once; add the (scale-1) extra
            total_unfused_energy_pJ += e * 1e6 * (sc - 1)

    for prod_idx, cons_idx in fusion_edges:
        _, prod_dram = op_stats[prod_idx]
        _, cons_dram = op_stats[cons_idx]

        saved_energy = _dram_output_energy(prod_dram) + _dram_input_energy(cons_dram)
        saved_cycles = _estimate_saved_cycles(prod_dram, cons_dram, dram_read_bw, dram_write_bw)

        # Apply scaling factors (e.g., n_kv_groups for attention ops)
        prod_scale = scale_factors.get(prod_idx, 1.0)
        cons_scale = scale_factors.get(cons_idx, 1.0)
        # Use the max scale since both endpoints contribute
        edge_scale = max(prod_scale, cons_scale)

        total_saved_energy_pJ += saved_energy * edge_scale
        total_saved_cycles += saved_cycles * edge_scale

    # Cap savings: can't save more than total DRAM energy (avoid negative energy)
    max_savings_pJ = total_unfused_energy_pJ * 0.9  # cap at 90% of total
    total_saved_energy_pJ = min(total_saved_energy_pJ, max_savings_pJ)
    total_saved_energy_uJ = total_saved_energy_pJ / 1e6
    return total_saved_energy_uJ, total_saved_cycles


# ---------------------------------------------------------------------------
# Layer evaluation with fusion
# ---------------------------------------------------------------------------

def evaluate_layer(layer: dict, n_embd: int, seq_length: int, work_dir: str,
                   fused: bool = True, arch: str = DEFAULT_ARCH,
                   mode: str = "prefill") -> dict:
    """Evaluate a single layer's hardware metrics.

    Args:
        mode: "prefill" — all ops use seq_length (GEMM, batch of tokens)
              "decode"  — projections use L=1 (GEMV, one token),
                          attention uses L=seq_length as context/KV-cache length
    """
    cfg = get_arch_config(arch)
    try:
        n_head = layer['n_head']
        n_kv_groups = layer['n_kv_group']
        n_qk_head_dim = layer['n_qk_head_dim']
        n_v_head_dim = layer['n_v_head_dim']
        n_cproj = layer['n_cproj']
        attn_variant = layer['attention_variant']
        mlp_size = layer['mlp_size']
    except KeyError as e:
        raise KeyError(f"Missing key in layer definition: {e}")

    # In decode mode: projections are GEMV (L=1), attention uses KV cache length
    proj_seq = 1 if mode == "decode" else seq_length
    attn_ctx = seq_length  # KV cache context length (used in both modes)

    if attn_variant == 'infinite':
        # Run all 7 GEMMs with detailed DRAM stats
        # Op 0: QK_gen  [embd -> qk*(h+kv), proj_seq]
        qk_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_qk_head_dim * (n_head + n_kv_groups),
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 1: V_gen   [embd -> v*kv, proj_seq]
        v_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_v_head_dim * n_kv_groups,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 2: QK_attn [qk -> attn_ctx, h//kv]  (scaled by n_kv_groups)
        #   decode: q[1,qk] × K_cache[qk,ctx] -> scores[1,ctx]
        qk_attn = run_GEMM_evaluation_detailed(
            in_channel=n_qk_head_dim, out_channel=attn_ctx,
            seq_length=n_head // n_kv_groups, work_dir=work_dir, arch=arch)
        # Op 3: PV_attn [attn_ctx -> v, h//kv]  (scaled by n_kv_groups)
        #   decode: scores[1,ctx] × V_cache[ctx,v] -> attended[1,v]
        pv_attn = run_GEMM_evaluation_detailed(
            in_channel=attn_ctx, out_channel=n_v_head_dim,
            seq_length=n_head // n_kv_groups, work_dir=work_dir, arch=arch)
        # Op 4: ATTN_proj [v*h -> embd, proj_seq]
        attn_proj = run_GEMM_evaluation_detailed(
            in_channel=n_v_head_dim * n_head, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 5: MLP_FC1 [embd -> mlp, proj_seq]
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 6: MLP_FC2 [mlp -> embd, proj_seq]
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)

        all_ops = [qk_gen, v_gen, qk_attn, pv_attn, attn_proj, mlp_fc1, mlp_fc2]

        # Apply n_kv_groups scaling to QK_attn (idx 2) and PV_attn (idx 3)
        for idx in [2, 3]:
            summary = all_ops[idx][0]
            for key in ['cycles', 'energy_uJ', 'total_ops', 'total_memory_accesses']:
                if summary[key] is not None:
                    summary[key] *= n_kv_groups

        # Extract summary stats for aggregation
        all_summaries = [op[0] for op in all_ops]

        if fused:
            # Define the producer->consumer fusion edges:
            #
            # Data flow graph:
            #   hidden -> QK_gen(0) -> Q,K --> QK_attn(2) -> scores --> PV_attn(3)
            #   hidden -> V_gen(1)  -> V   --------------------------> PV_attn(3)
            #   PV_attn(3) -> attended --> ATTN_proj(4)
            #   ATTN_proj(4) -> hidden' --> MLP_FC1(5)
            #   MLP_FC1(5) -> expanded --> MLP_FC2(6)
            #
            # Fusible edges (producer output = consumer input, stays on-chip):
            fusion_edges = [
                (0, 2),  # QK_gen outputs -> QK_attn inputs (Q,K projections)
                (1, 3),  # V_gen outputs -> PV_attn inputs (V values)
                (2, 3),  # QK_attn outputs -> PV_attn inputs (attention scores)
                (3, 4),  # PV_attn outputs -> ATTN_proj inputs (attended values)
                (4, 5),  # ATTN_proj outputs -> MLP_FC1 inputs (hidden states)
                (5, 6),  # MLP_FC1 outputs -> MLP_FC2 inputs (expanded activations)
            ]

            # Scale factors for ops that were scaled by n_kv_groups
            scale_factors = {2: n_kv_groups, 3: n_kv_groups}

            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                all_ops, fusion_edges, scale_factors,
                dram_read_bw=cfg.dram_read_bw, dram_write_bw=cfg.dram_write_bw)

            layer_stats = aggregate_stats(all_summaries)

            # Subtract fusion savings
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            # Store savings for debugging
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    else:
        # Identity or causal: only MLP
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)

        all_summaries = [mlp_fc1[0], mlp_fc2[0]]

        if fused:
            fusion_edges = [(0, 1)]  # MLP_FC1 -> MLP_FC2
            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                [mlp_fc1, mlp_fc2], fusion_edges,
                dram_read_bw=cfg.dram_read_bw, dram_write_bw=cfg.dram_write_bw)
            layer_stats = aggregate_stats(all_summaries)
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    return layer_stats


def eval_individual(individual: Individual, work_dir: str, fused: bool = True, arch: str = DEFAULT_ARCH, mode: str = "prefill") -> dict:
    global_spec = individual["globals"]
    layer_spec = individual["layers"]
    n_embd = global_spec["n_embd"]
    seq_length = global_spec["block_size"]
    layer_mask = global_spec.get("layer_mask", None)
    if layer_mask is None:
        raise ValueError("layer_mask is not defined in global_spec")

    hw_eval_list = []
    for i, layer in enumerate(layer_spec):
        if layer_mask[i] == 1:
            layer_stats = evaluate_layer(layer, n_embd, seq_length, work_dir, fused=fused, arch=arch, mode=mode)
            hw_eval_list.append(layer_stats)

    aggregated_stats = aggregate_stats(hw_eval_list)

    # average over sequence length
    aggregated_stats['cycles_per_token'] = aggregated_stats['cycles'] / seq_length if aggregated_stats['cycles'] is not None else None
    aggregated_stats['token_delay'] = aggregated_stats['cycles_per_token'] / 1e9  # assuming 1GHz clock
    aggregated_stats['energy_per_token_uJ'] = aggregated_stats['energy_uJ'] / seq_length if aggregated_stats['energy_uJ'] is not None else None
    aggregated_stats['edp_per_token'] = aggregated_stats['edp'] / seq_length if aggregated_stats['edp'] is not None else None
    return aggregated_stats


def evaluate_population(population: list, base_work_dir: str, fused: bool = True, arch: str = DEFAULT_ARCH, mode: str = "prefill") -> list:
    n = len(population)
    results = []
    for i, individual in enumerate(population):
        individual_stats = eval_individual(individual, work_dir=base_work_dir, fused=fused, arch=arch, mode=mode)
        results.append(individual_stats)
        print(f"\r  HW eval [{i+1}/{n}]", end="", flush=True)
    print()
    return results


def aggregate_stats(stats_list: list) -> dict:
    aggregated_stats = {}
    for key in stats_list[0].keys():
        # Only sum scalar numeric stats. Non-numeric / structural fields
        # (e.g. `padded_D = [orig, padded]`) are collected separately
        # below so we don't accidentally concatenate lists.
        vals = [s[key] for s in stats_list if s.get(key) is not None]
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            aggregated_stats[key] = sum(vals)
        elif vals:
            aggregated_stats[key] = vals[0]      # carry first non-None as a representative

    # If any input op was padded, surface the per-op padding records.
    # Two cases:
    #   1) GEMM-level: each `s` carries a single `padded_D = [orig, padded]`.
    #   2) Layer-level: each `s` carries `padded_ops` (list of records) +
    #      `padded_op_count` from a previous aggregate_stats call.
    # Concatenate either form into a flat list at this level.
    flat = []
    for s in stats_list:
        if s.get('padded_ops'):
            flat.extend(s['padded_ops'])
        elif s.get('padded_D'):
            flat.append(s['padded_D'])
    if flat:
        aggregated_stats['padded_ops'] = flat
        aggregated_stats['padded_op_count'] = len(flat)

    # recalculate derived metrics
    if aggregated_stats['total_ops'] is not None and aggregated_stats['total_memory_accesses'] is not None and aggregated_stats['total_memory_accesses'] != 0:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = aggregated_stats['total_ops'] / aggregated_stats['total_memory_accesses']
    else:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = None
    aggregated_stats['algorithmic_intensity_ops_per_byte'] = aggregated_stats['algorithmic_intensity_ops_per_access']
    aggregated_stats['edp'] = aggregated_stats['energy_uJ'] * aggregated_stats['cycles'] / 10e6 if aggregated_stats['energy_uJ'] is not None and aggregated_stats['cycles'] is not None else None  # J*ns

    total_cycle = aggregated_stats['cycles']
    aggregated_stats['utilization_pct'] = 0
    aggregated_stats['gflops'] = 0
    for stats in stats_list:
        aggregated_stats['utilization_pct'] += (stats['utilization_pct'] * stats['cycles'] / total_cycle) if stats['utilization_pct'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0
        aggregated_stats['gflops'] += (stats['gflops'] * stats['cycles'] / total_cycle) if stats['gflops'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0

    return aggregated_stats
