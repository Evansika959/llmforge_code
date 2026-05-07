import os
import re
from typing import Dict, Optional


def _num(s: str) -> Optional[float]:
    """Parse a number that may contain commas and unit suffixes.

    Returns a float or None if parsing fails.
    """
    if s is None:
        return None
    s = s.strip()
    # remove commas
    s = s.replace(',', '')
    # strip common units (uJ, mm^2, %) and parentheses
    s = re.sub(r"\s*(uJ|mJ|J|mm\^2|mm2|%|GHz)\s*$", '', s, flags=re.IGNORECASE)
    try:
        return float(s)
    except Exception:
        return None


def parse_timeloop_stats(path: str) -> Dict[str, Optional[float]]:
    """Parse a timeloop-mapper.stats.txt file and extract useful metrics.

    Extracted fields (when present):
      - gflops (GFLOPs @1GHz)
      - utilization_pct
      - cycles
      - energy_uJ
      - edp
      - area_mm2
      - total_ops
      - total_memory_accesses
      - optimal_ops_per_byte
      - algorithmic_intensity_ops_per_access = total_ops / total_memory_accesses
      - algorithmic_intensity_ops_per_byte = optimal_ops_per_byte if present, else same as ops_per_access

    Returns a dict mapping keys to floats or None.
    """
    metrics = {
        'gflops': None,
        'utilization_pct': None,
        'cycles': None,
        'energy_uJ': None,
        'edp': None,
        'area_mm2': None,
        'total_ops': None,
        'total_memory_accesses': None,
        'optimal_ops_per_byte': None,
        'algorithmic_intensity_ops_per_access': None,
        'algorithmic_intensity_ops_per_byte': None,
    }

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Summary Stats block
    m = re.search(r'GFLOPs\s*\(@1GHz\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['gflops'] = _num(m.group(1))

    m = re.search(r'Utilization:\s*([0-9.,]+)%', text)
    if m:
        metrics['utilization_pct'] = _num(m.group(1))

    m = re.search(r'Cycles:\s*([0-9,]+)', text)
    if m:
        metrics['cycles'] = _num(m.group(1))

    m = re.search(r'Energy:\s*([0-9.,Ee+-]+)\s*uJ', text, flags=re.IGNORECASE)
    if m:
        metrics['energy_uJ'] = _num(m.group(1))

    m = re.search(r'EDP\(J\*cycle\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['edp'] = _num(m.group(1))

    m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm', text)
    if m:
        metrics['area_mm2'] = _num(m.group(1))
    else:
        # try matching "Area: 0.00 mm^2" or similar
        m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm\^2', text)
        if m:
            metrics['area_mm2'] = _num(m.group(1))

    # Operational Intensity & totals
    m = re.search(r'Total ops\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_ops'] = _num(m.group(1))

    m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_memory_accesses'] = _num(m.group(1))

    m = re.search(r'Optimal Op per Byte\s*:\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['optimal_ops_per_byte'] = _num(m.group(1))

    # Fallbacks: sometimes words are slightly different
    if metrics['total_ops'] is None:
        m = re.search(r'Total elementwise ops\s*:\s*([0-9,]+)', text)
        if m:
            total_elem = _num(m.group(1))
        else:
            total_elem = None
        m = re.search(r'Total reduction ops\s*:\s*([0-9,]+)', text)
        if m:
            total_red = _num(m.group(1))
        else:
            total_red = None
        if total_elem is not None and total_red is not None:
            metrics['total_ops'] = total_elem + total_red

    if metrics['total_memory_accesses'] is None:
        m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
        if m:
            metrics['total_memory_accesses'] = _num(m.group(1))

    # Compute algorithmic intensity
    if metrics['total_ops'] and metrics['total_memory_accesses']:
        try:
            metrics['algorithmic_intensity_ops_per_access'] = (
                float(metrics['total_ops']) / float(metrics['total_memory_accesses'])
            )
        except Exception:
            metrics['algorithmic_intensity_ops_per_access'] = None

    # If optimal op/byte present, prefer that as ops/byte metric
    if metrics['optimal_ops_per_byte'] is not None:
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['optimal_ops_per_byte']
    elif metrics['algorithmic_intensity_ops_per_access'] is not None:
        # We don't strictly know whether "memory accesses" is in bytes or words.
        # Use ops_per_access as a fallback for ops/byte.
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['algorithmic_intensity_ops_per_access']

    return metrics


def parse_dram_dataspace_stats(path: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Parse per-dataspace DRAM access stats from timeloop-mapper.stats.txt.

    Returns a dict keyed by dataspace name ('Weights', 'Inputs', 'Outputs'),
    each containing:
      - scalar_reads: number of scalar reads from DRAM
      - scalar_updates: number of scalar writes/updates to DRAM
      - scalar_fills: number of scalar fills from DRAM
      - energy_pJ: total energy in picojoules for this dataspace at DRAM
      - energy_per_access_pJ: energy per scalar access
      - partition_size: total elements in this dataspace
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    result: Dict[str, Dict[str, Optional[float]]] = {}

    # Find the DRAM STATS section
    dram_match = re.search(r'=== DRAM ===.*?STATS\s*\n\s*-----\s*\n(.*?)(?:\nNetworks|\nLevel|\Z)',
                           text, re.DOTALL)
    if not dram_match:
        return result

    dram_section = dram_match.group(1)

    # Parse each dataspace block within DRAM STATS
    # Pattern: "    Weights:" or "    Inputs:" or "    Outputs:" followed by indented lines
    dataspace_pattern = re.compile(
        r'^\s{4}(\w+):\s*$'
        r'((?:\n\s{8}.+)+)',
        re.MULTILINE
    )

    for ds_match in dataspace_pattern.finditer(dram_section):
        ds_name = ds_match.group(1)
        ds_block = ds_match.group(2)

        ds_stats: Dict[str, Optional[float]] = {
            'scalar_reads': None,
            'scalar_updates': None,
            'scalar_fills': None,
            'energy_pJ': None,
            'energy_per_access_pJ': None,
            'partition_size': None,
        }

        m = re.search(r'Partition size\s*:\s*([0-9,]+)', ds_block)
        if m:
            ds_stats['partition_size'] = _num(m.group(1))

        m = re.search(r'Scalar reads \(per-instance\)\s*:\s*([0-9,]+)', ds_block)
        if m:
            ds_stats['scalar_reads'] = _num(m.group(1))

        m = re.search(r'Scalar updates \(per-instance\)\s*:\s*([0-9,]+)', ds_block)
        if m:
            ds_stats['scalar_updates'] = _num(m.group(1))

        m = re.search(r'Scalar fills \(per-instance\)\s*:\s*([0-9,]+)', ds_block)
        if m:
            ds_stats['scalar_fills'] = _num(m.group(1))

        m = re.search(r'Energy \(per-scalar-access\)\s*:\s*([0-9.,Ee+-]+)\s*pJ', ds_block)
        if m:
            ds_stats['energy_per_access_pJ'] = _num(m.group(1))

        m = re.search(r'Energy \(total\)\s*:\s*([0-9.,Ee+-]+)\s*pJ', ds_block)
        if m:
            ds_stats['energy_pJ'] = _num(m.group(1))

        result[ds_name] = ds_stats

    return result


def parse_buffer_stats(path: str) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Parse per-buffer, per-dataspace access stats from timeloop-mapper.stats.txt.

    Returns a nested dict:
        {buffer_name: {dataspace_name: {metric: value}}}

    Buffer names: 'acc_buffer', 'wmem', 'head_sram', 'global_sram', 'DRAM'
    Dataspace names: 'Weights', 'Inputs', 'Outputs'
    Metrics per dataspace: scalar_reads, scalar_fills, scalar_updates,
        temporal_reductions, utilized_capacity, partition_size,
        instances, energy_per_access_pJ, energy_total_pJ

    Also includes a top-level 'total_scalar_accesses' and 'op_per_byte'
    from the Operational Intensity section, stored under buffer_name directly.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    result: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    buffer_names = ['acc_buffer', 'wmem', 'head_sram', 'global_sram', 'DRAM']

    # --- Per-buffer STATS sections (detailed per-dataspace) ---
    for buf in buffer_names:
        buf_pattern = re.compile(
            r'=== ' + re.escape(buf) + r' ===.*?STATS\s*\n\s*-----\s*\n(.*?)(?=\nLevel |\nNetworks|\Z)',
            re.DOTALL
        )
        buf_match = buf_pattern.search(text)
        if not buf_match:
            continue

        buf_section = buf_match.group(1)
        result[buf] = {}

        # Parse cycles and bandwidth throttling at buffer level
        m = re.search(r'Cycles\s*:\s*([0-9,]+)', buf_section)
        buf_cycles = _num(m.group(1)) if m else None

        # Parse each dataspace block
        ds_pattern = re.compile(
            r'^\s{4}(\w+):\s*$'
            r'((?:\n\s{8}.+)+)',
            re.MULTILINE
        )
        for ds_match in ds_pattern.finditer(buf_section):
            ds_name = ds_match.group(1)
            ds_block = ds_match.group(2)

            ds: Dict[str, Optional[float]] = {}

            for field, regex in [
                ('partition_size', r'Partition size\s*:\s*([0-9,]+)'),
                ('utilized_capacity', r'Utilized capacity\s*:\s*([0-9,]+)'),
                ('instances', r'Utilized instances \(max\)\s*:\s*([0-9,]+)'),
                ('scalar_reads', r'Scalar reads \(per-instance\)\s*:\s*([0-9,]+)'),
                ('scalar_fills', r'Scalar fills \(per-instance\)\s*:\s*([0-9,]+)'),
                ('scalar_updates', r'Scalar updates \(per-instance\)\s*:\s*([0-9,]+)'),
                ('temporal_reductions', r'Temporal reductions \(per-instance\)\s*:\s*([0-9,]+)'),
                ('energy_per_access_pJ', r'Energy \(per-scalar-access\)\s*:\s*([0-9.,Ee+-]+)\s*pJ'),
                ('energy_total_pJ', r'Energy \(total\)\s*:\s*([0-9.,Ee+-]+)\s*pJ'),
            ]:
                m = re.search(regex, ds_block)
                ds[field] = _num(m.group(1)) if m else None

            # Compute total accesses across all instances
            inst = ds.get('instances') or 1
            reads = (ds.get('scalar_reads') or 0) * inst
            fills = (ds.get('scalar_fills') or 0) * inst
            updates = (ds.get('scalar_updates') or 0) * inst
            ds['total_reads'] = reads
            ds['total_fills'] = fills
            ds['total_updates'] = updates
            ds['total_accesses'] = reads + fills + updates

            result[buf][ds_name] = ds

    # --- Operational Intensity section (summary per buffer) ---
    oi_match = re.search(r'Operational Intensity Stats.*?\n(.*?)(?=\nSummary Stats|\Z)',
                         text, re.DOTALL)
    if oi_match:
        oi_section = oi_match.group(1)
        for buf in buffer_names:
            m = re.search(
                r'=== ' + re.escape(buf) + r' ===\s*\n'
                r'\s*Total scalar accesses\s*:\s*([0-9,]+)\s*\n'
                r'\s*Op per Byte\s*:\s*([0-9.,Ee+-]+)',
                oi_section
            )
            if m:
                if buf not in result:
                    result[buf] = {}
                result[buf]['_summary'] = {
                    'total_scalar_accesses': _num(m.group(1)),
                    'op_per_byte': _num(m.group(2)),
                }

    return result


def parse_art_area(stats_dir: str) -> Optional[float]:
    """Parse total accelerator area (in mm²) from the ART YAML in a Timeloop output directory.

    Returns total area in mm² or None if ART file not found.
    """
    import yaml as _yaml

    art_path = os.path.join(stats_dir, "timeloop-mapper.ART.yaml")
    if not os.path.exists(art_path):
        return None

    with open(art_path, 'r') as f:
        art = _yaml.safe_load(f)

    if not art or 'ART' not in art or 'tables' not in art['ART']:
        return None

    total_area_um2 = 0.0
    for entry in art['ART']['tables']:
        area = entry.get('area', 0.0)
        name = entry.get('name', '')
        m = re.search(r'\[(\d+)\.\.(\d+)\]', name)
        instances = int(m.group(2)) if m else 1
        total_area_um2 += area * instances

    return total_area_um2 / 1e6  # convert um² to mm²


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python parse_timeloop_stats.py /path/to/timeloop-mapper.stats.txt')
        raise SystemExit(2)
    path = sys.argv[1]
    out = parse_timeloop_stats(path)
    # Print a compact report
    print('Parsed Timeloop stats:')
    for k, v in out.items():
        print(f'  {k}: {v}')
