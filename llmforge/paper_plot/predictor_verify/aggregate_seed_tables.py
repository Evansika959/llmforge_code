#!/usr/bin/env python3
"""Aggregate per-seed predictor-verify tables into a single mean +/- std table.

Reads one or more `table_a_seed<N>.md` files produced by `reproduce_v2.sh`,
parses the metric grid (rows = metrics, columns = methods) from each, and
emits a single Markdown + LaTeX table where every cell shows mean +/- std
across seeds.

Each input table has the structure (from table_a.py):

    | Metric          | ForgeFormer (IHA) | MLP | RF | ForgeFormer (HW-GPT-Bench) | MLP | RF |
    |-----------------|------------------|-----|----|---------------------------|-----|----|
    | MAE             | 0.2055           | ... | ... | ...                       | ... | ... |
    | MAE@5%          | 0.0335           | ... | ... | ...                       | ... | ... |
    | Spearman $\\rho$ | +0.8053          | ... | ... | ...                       | ... | ... |
    | Kendall $\\tau$  | +0.6265          | ... | ... | ...                       | ... | ... |
    | $k$@1%          | 49               | ... | ... | ...                       | ... | ... |
    | $k$@5%          | 183              | ... | ... | ...                       | ... | ... |

The aggregated table preserves the same row/column layout and replaces each
numeric cell with `mean ± std` formatted to a sensible precision per metric.
The bold-best convention is also applied: per row, the method with the best
mean wins the bolding (lower-better for MAE / k, higher-better for Spearman /
Kendall).
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


# Metric -> (lower_is_better, decimals for mean, decimals for std)
METRIC_FMT = {
    "MAE":              (True,  4, 4),
    "MAE@5%":           (True,  4, 4),
    "Spearman $\\rho$": (False, 4, 4),
    "Kendall $\\tau$":  (False, 4, 4),
    "$k$@1%":           (True,  1, 1),
    "$k$@5%":           (True,  1, 1),
}


def parse_md_table(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Return (column_headers, {metric_name: [val_per_method, ...]}).

    Headers exclude the leading 'Metric' column. Values are floats.
    """
    text = path.read_text()
    lines = [ln.rstrip() for ln in text.splitlines()]
    rows = [ln for ln in lines if ln.startswith("|") and not ln.startswith("|---")
            and not ln.startswith("|-----") and "---" not in ln]
    if len(rows) < 2:
        raise RuntimeError(f"{path}: could not find a Markdown table with header + body")

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    header = cells(rows[0])
    if header[0].lower() != "metric":
        raise RuntimeError(f"{path}: expected first column 'Metric', got {header[0]!r}")
    column_headers = header[1:]

    grid: dict[str, list[float]] = {}
    for line in rows[1:]:
        c = cells(line)
        if len(c) != len(header):
            continue
        metric = c[0]
        try:
            vals = [float(v.replace("+", "")) for v in c[1:]]
        except ValueError:
            continue
        grid[metric] = vals
    return column_headers, grid


def aggregate(per_seed_grids: list[dict[str, list[float]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {metric: (mean, std)} stacked across seeds."""
    if not per_seed_grids:
        return {}
    metrics = list(per_seed_grids[0].keys())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in metrics:
        stack = np.array([g[m] for g in per_seed_grids if m in g], dtype=float)
        if stack.size == 0:
            continue
        out[m] = (stack.mean(axis=0), stack.std(axis=0, ddof=0))
    return out


def fmt_cell(mean: float, std: float, dec_mean: int, dec_std: int,
             is_best: bool, signed: bool = False) -> str:
    sign = "+" if (signed and mean >= 0) else ""
    body = f"{sign}{mean:.{dec_mean}f} $\\pm$ {std:.{dec_std}f}"
    return f"**{body}**" if is_best else body


def render_md(columns: list[str], agg: dict[str, tuple[np.ndarray, np.ndarray]],
              n_seeds: int, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Predictor accuracy on held-out test (multi-seed aggregate)")
    lines.append("")
    lines.append(f"_Aggregate of {n_seeds} seed runs. Each cell is mean $\\pm$ std across seeds._")
    lines.append("")
    head = "| Metric          | " + " | ".join(columns) + " |"
    sep  = "|-----------------|" + "|".join(["-" * (len(c) + 2) for c in columns]) + "|"
    lines.append(head)
    lines.append(sep)
    for metric, (mu, sigma) in agg.items():
        lower_better, dm, ds = METRIC_FMT.get(metric, (True, 4, 4))
        signed = metric.startswith("Spearman") or metric.startswith("Kendall")
        # Best-per-row across the methods on the same dataset block, where
        # the IHA block is columns [0, 1, 2] and the HW-GPT-Bench block is
        # columns [3, 4, 5] in the standard layout. We bold within each
        # block independently so per-dataset best is highlighted.
        bold_idx: set[int] = set()
        for block in ((0, 1, 2), (3, 4, 5)):
            block = [i for i in block if i < len(mu)]
            if not block:
                continue
            block_means = [mu[i] for i in block]
            best_local = (np.argmin(block_means) if lower_better
                          else np.argmax(block_means))
            bold_idx.add(block[int(best_local)])

        cells = [fmt_cell(mu[i], sigma[i], dm, ds, i in bold_idx, signed)
                 for i in range(len(mu))]
        lines.append(f"| {metric:<15} | " + " | ".join(cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))


def render_tex(columns: list[str], agg: dict[str, tuple[np.ndarray, np.ndarray]],
               n_seeds: int, out_path: Path) -> None:
    """Emit a booktabs LaTeX table mirroring the Markdown layout."""
    lines: list[str] = []
    lines.append("% Predictor accuracy on held-out test, aggregate of multiple seeds")
    lines.append("\\begin{tabular}{l" + "c" * len(columns) + "}")
    lines.append("\\toprule")
    lines.append("Metric & " + " & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    for metric, (mu, sigma) in agg.items():
        lower_better, dm, ds = METRIC_FMT.get(metric, (True, 4, 4))
        signed = metric.startswith("Spearman") or metric.startswith("Kendall")
        bold_idx: set[int] = set()
        for block in ((0, 1, 2), (3, 4, 5)):
            block = [i for i in block if i < len(mu)]
            if not block:
                continue
            block_means = [mu[i] for i in block]
            best_local = (np.argmin(block_means) if lower_better
                          else np.argmax(block_means))
            bold_idx.add(block[int(best_local)])
        cells: list[str] = []
        for i, (m_, s_) in enumerate(zip(mu, sigma)):
            sign = "+" if (signed and m_ >= 0) else ""
            body = f"{sign}{m_:.{dm}f} $\\pm$ {s_:.{ds}f}"
            cells.append(f"\\textbf{{{body}}}" if i in bold_idx else body)
        lines.append(f"{metric} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% aggregate over {n_seeds} seeds")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_glob", required=True,
                   help="Glob pattern matching per-seed table_a_seed*.md files.")
    p.add_argument("--out_md", type=Path, required=True)
    p.add_argument("--out_tex", type=Path, required=True)
    args = p.parse_args()

    in_paths = sorted(Path(p_).resolve() for p_ in glob.glob(args.in_glob))
    if not in_paths:
        raise SystemExit(f"no files matched: {args.in_glob}")
    print(f"[aggregate] {len(in_paths)} per-seed tables found:")
    for p_ in in_paths:
        print(f"  - {p_.name}")

    columns: list[str] | None = None
    grids: list[dict[str, list[float]]] = []
    for p_ in in_paths:
        cols, grid = parse_md_table(p_)
        if columns is None:
            columns = cols
        elif cols != columns:
            raise RuntimeError(f"{p_.name}: column header mismatch with first table\n"
                               f"  this:  {cols}\n"
                               f"  first: {columns}")
        grids.append(grid)

    agg = aggregate(grids)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    render_md(columns or [], agg, len(grids), args.out_md)
    render_tex(columns or [], agg, len(grids), args.out_tex)
    print(f"[aggregate] wrote {args.out_md}")
    print(f"[aggregate] wrote {args.out_tex}")


if __name__ == "__main__":
    main()
