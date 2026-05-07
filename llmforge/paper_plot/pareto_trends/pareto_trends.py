"""Layer-wise fingerprint of top-N non-dominated winners across HW substrates.

Per-substrate steps:
  1. Load the last-generation cosearch ckpt.
  2. Run iterated NSGA-II non-dominated sort to assign each arch a front
     rank, then take the top TOP_N=100 by rank ascending (front 0, then
     front 1, …, walking fronts until we accumulate TOP_N picks).
  3. For each pick, extract the active layers' per-field values in their
     original order (using globals.layer_mask).  Importantly, when a
     layer's `attention_variant == "identity"`, its n_head/n_kv_group/
     n_qk_head_dim/n_v_head_dim values are *dead* genome bits (no Q/K/V
     projection runs, no head computation), so we mark them as NaN at
     that position. Only mlp_size remains live in identity layers.
  4. Interpolate each field onto a fixed relative-depth grid (0=first
     active, 1=last active) so variable-depth archs are comparable.
     interp_to_grid drops NaN points before interpolation, so dead
     attention fields are excluded from each arch's contribution.
  5. Aggregate np.nanmean ± np.nanstd across the top-N pool at each grid
     point — positions where every arch is identity show as NaN gaps.

Output: figs/fig_pareto_fingerprint.pdf — 5 fields + 1 identity-rate
panel × N substrates fingerprint.

Source: top-N non-dominated archs from the actual NSGA-II runs (no
surrogate sampling), so trends are framework outputs.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT))

from nsga2 import dominates  # noqa: E402

# ── Inputs: per-substrate latest cosearch checkpoints ───────────────────────
SUBSTRATES = [
    ("ZEUS / GPU",          PROJECT / "ckpts/cosearch_200m_zeus/0424_1946_ckpt_gen30.json"),
    ("Gemmini / systolic",  PROJECT / "ckpts_paper/20260428_finetune_200m_paramsloss_gemmini/0428_0602_ckpt_gen40.json"),
    ("Eyeriss / row-stat",  PROJECT / "ckpts_paper/20260428_finetune_200m_paramsloss_eyeriss/0428_0630_ckpt_gen40.json"),
    ("FLAT / fused",        PROJECT / "ckpts/hw_search_d512/0416_1908_ckpt_gen20.json"),
]

FIELDS = [
    ("n_head",        r"$\mathbf{n_h}$"),
    ("n_kv_group",    r"$\mathbf{n_{kv}}$"),
    ("n_qk_head_dim", r"$\mathbf{d_{qk}}$"),
    ("n_v_head_dim",  r"$\mathbf{d_v}$"),
    ("mlp_size",      r"$\mathbf{d_{mlp}}$"),
]

OUT_DIR = THIS_DIR / "figs"
GRID = np.linspace(0.0, 1.0, 21)  # 21-point relative-depth grid
TOP_N = 50                         # walk fronts 0,1,... until this many archs

# Okabe-Ito subset, per-substrate colour
SUBSTRATE_COLOR = {
    "ZEUS / GPU":         "#0072B2",  # blue
    "Gemmini / systolic": "#009E73",  # bluish-green
    "Eyeriss / row-stat": "#CC79A7",  # reddish purple
    "FLAT / fused":       "#D55E00",  # vermilion
}

# NeurIPS-style typography: Times-family serif, 10 pt body, with axis-
# label / title text bolded for emphasis. Bumped sizes throughout so the
# figure stays legible when scaled to a single column or half-page.
STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11.0,
    "axes.titlesize": 13.0, "axes.titleweight": "bold",
    "axes.labelsize":  12.0, "axes.labelweight": "bold",
    "legend.fontsize": 10.0, "legend.frameon": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.30, "grid.linewidth": 0.45,
    "lines.linewidth": 1.8,
}


# ── Non-dominated sort: rank every individual, then take top-N ─────────────

def top_n_indices(objs: List[List[float]], cons: List[List[float]],
                  top_n: int) -> List[int]:
    """Walk NSGA-II non-dominated fronts and return the first `top_n` indices.

    Standard NSGA-II elitist accumulator: compute domination counts and
    "dominated by p" lists, peel off front 0 (those with count=0), then
    decrement neighbours, repeat. Stop once we've picked top_n indices
    (returns fewer if the population is smaller).
    """
    N = len(objs)
    dominated_by: List[List[int]] = [[] for _ in range(N)]   # who p dominates
    domcount = [0] * N                                        # # times p is dominated
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            d = dominates(objs[p], cons[p], objs[q], cons[q])
            if d == 1:
                dominated_by[p].append(q)
            elif d == -1:
                domcount[p] += 1

    picks: List[int] = []
    current_front = [i for i in range(N) if domcount[i] == 0]
    front_id = 0
    while current_front and len(picks) < top_n:
        picks.extend(current_front)
        nxt: List[int] = []
        for p in current_front:
            for q in dominated_by[p]:
                domcount[q] -= 1
                if domcount[q] == 0:
                    nxt.append(q)
        current_front = nxt
        front_id += 1
    return picks[:top_n]


# ── Per-architecture: active-layer field sequences ─────────────────────────
# Attention fields that are dead when `attention_variant == "identity"`:
ATTN_FIELDS = {"n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim"}


def active_layer_sequence(individual: dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return (fields, identity_mask) in active-layer order.

    fields[f] is a 1-D array of length n_active. For attention fields
    (n_head/n_kv_group/n_qk_head_dim/n_v_head_dim), positions where the
    layer's attention_variant == "identity" are NaN — those fields are
    dead genome bits in identity layers (no Q/K/V projection / head
    computation runs), so they should not contribute to the cross-arch
    mean. mlp_size is always live and never NaN'd.

    identity_mask is a length-n_active 0/1 indicator — 1 where the
    active layer is identity-attention, used for the identity-rate panel.
    """
    mask = individual["globals"]["layer_mask"]
    layers = individual["layers"]
    out: Dict[str, list] = {f: [] for f, _ in FIELDS}
    ident: List[float] = []
    for li, active in enumerate(mask):
        if not active:
            continue
        if li >= len(layers):
            break
        is_ident = layers[li].get("attention_variant", "infinite") == "identity"
        ident.append(1.0 if is_ident else 0.0)
        for f, _ in FIELDS:
            if is_ident and f in ATTN_FIELDS:
                out[f].append(float("nan"))
            else:
                out[f].append(float(layers[li][f]))
    return ({f: np.asarray(v, dtype=np.float32) for f, v in out.items()},
            np.asarray(ident, dtype=np.float32))


def interp_to_grid(seq: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample a length-L sequence onto the fixed relative-depth grid.

    NaN entries (e.g., dead attention fields in identity layers) are
    dropped before interpolation. If all entries are NaN, returns NaN
    everywhere on the grid; if only some are NaN, those positions are
    interpolated from the live neighbours, and downstream nanmean will
    weight the grid point only where the arch contributed a live value.
    """
    finite = np.isfinite(seq)
    if not finite.any():
        return np.full_like(grid, np.nan)
    seq = seq[finite]
    if len(seq) == 1:
        return np.full_like(grid, seq[0])
    src = np.linspace(0.0, 1.0, int(finite.sum()))
    # Build a mask grid: a grid-point gets NaN if it's outside the live-
    # entry span. With drop-NaN, src already spans [0,1], so np.interp's
    # extrapolation just clamps to endpoints — fine for reasonable archs.
    return np.interp(grid, src, seq)


def interp_indicator_to_grid(indicator: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample a 0/1 indicator (e.g., identity_mask) onto the grid.

    Linear interpolation on a 0/1 sequence yields a [0,1] curve whose
    average across archs at each grid point is the *identity rate* at
    that relative depth.
    """
    if len(indicator) == 0:
        return np.full_like(grid, np.nan)
    if len(indicator) == 1:
        return np.full_like(grid, float(indicator[0]))
    src = np.linspace(0.0, 1.0, len(indicator))
    return np.interp(grid, src, indicator)


def _arch_hash(ind: dict) -> str:
    return hashlib.sha256(json.dumps(ind, sort_keys=True).encode()).hexdigest()[:16]


def collect_unique_archs(ckpt_path: Path) -> Tuple[List[dict], List[List[float]], List[List[float]]]:
    """Walk every ckpt_gen*.json (and offspring_gen*.json if present) in the
    same dir as `ckpt_path`, dedupe individuals by hash, return aligned
    (individuals, objs, cons) lists for cumulative non-dominated sort.

    The `ckpt_path` itself is one of those files; we use its parent dir
    + basename pattern to find siblings.
    """
    d0 = ckpt_path.parent
    # Filename pattern: <ts_prefix>_ckpt_gen<N>.json  (and *_offspring_gen<N>.json
    # if the run was launched with --save_offspring).
    pat = re.compile(r"_(ckpt|offspring)_gen\d+\.json$")
    files = sorted(p for p in d0.iterdir() if pat.search(p.name))
    seen: Dict[str, Tuple[dict, List[float], List[float]]] = {}
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        # population first
        for ind, ev in zip(d.get("individuals", []), d.get("evaluations", [])):
            h = _arch_hash(ind)
            if h not in seen:
                seen[h] = (ind, ev["objs"], ev["cons"])
        # offspring (if present in this ckpt — saved by --save_offspring)
        for ind, ev in zip(d.get("offspring") or [], d.get("offspring_evaluations") or []):
            h = _arch_hash(ind)
            if h not in seen:
                seen[h] = (ind, ev["objs"], ev["cons"])
    individuals = [v[0] for v in seen.values()]
    objs = [v[1] for v in seen.values()]
    cons = [v[2] for v in seen.values()]
    return individuals, objs, cons


def pareto_fingerprint(ckpt_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Returns {field: (mean_curve, std_curve, n_picks)} along the grid.

    Selects the top-TOP_N archs by NSGA-II non-dominated rank from the
    *cumulative search history* (all unique archs across every gen ckpt
    in the run dir, deduplicated by hash). This guarantees the top-N
    pool isn't capped at the per-gen pop_size. Aggregation uses NaN-aware
    reductions so dead-attention positions in identity layers don't bias
    the curves; adds a synthetic identity-rate panel.
    """
    individuals, objs, cons = collect_unique_archs(ckpt_path)
    picks = top_n_indices(objs, cons, TOP_N)
    inds = [individuals[i] for i in picks]
    print(f"  {ckpt_path.name} (+siblings): "
          f"|unique_pool|={len(individuals)}, top_n={len(picks)} (target {TOP_N})")

    field_curves: Dict[str, list] = {f: [] for f, _ in FIELDS}
    ident_curves: List[np.ndarray] = []
    n_actives: List[int] = []
    for ind in inds:
        seqs, ident = active_layer_sequence(ind)
        n_actives.append(int(len(ident)))
        for f, _ in FIELDS:
            field_curves[f].append(interp_to_grid(seqs[f], GRID))
        ident_curves.append(interp_indicator_to_grid(ident, GRID))
    print(f"    n_active depth: median={int(np.median(n_actives))}, "
          f"min={min(n_actives)}, max={max(n_actives)}")
    n_ident_layers = sum(int(np.sum(np.round(ic) == 1.0)) for ic in ident_curves)
    print(f"    avg identity rate across picks: "
          f"{float(np.nanmean([c.mean() for c in ident_curves])):.3f}")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for f, _ in FIELDS:
        stack = np.stack(field_curves[f], axis=0)
        mean = np.nanmean(stack, axis=0)
        # Per-arch spread (sample standard deviation), NaN-aware. Shows
        # the within-substrate range of individual top-N picks at each
        # grid point — wider than SEM by √N, so substrate curves may
        # visually overlap even though their means are well-separated.
        std = np.nanstd(stack, axis=0)
        out[f] = (mean, std, len(inds))
    # Identity-rate panel: mean = fraction of picks in identity at this depth
    ident_stack = np.stack(ident_curves, axis=0)
    out["identity_rate"] = (np.nanmean(ident_stack, axis=0),
                             np.nanstd(ident_stack, axis=0), len(inds))
    return out


# ── Plotting ────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fingerprints: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, int]]] = {}
    for label, path in SUBSTRATES:
        if not path.exists():
            print(f"[skip] {label}: {path} missing")
            continue
        print(f"[load] {label}")
        fingerprints[label] = pareto_fingerprint(path)

    # 1×5 figure: one panel per architectural field, with NaN-aware
    # aggregation so identity-attention layers don't pollute the
    # attention-related field curves (their dead genome bits are NaN'd
    # in active_layer_sequence; np.nanmean skips them).
    fig, axes = plt.subplots(
        1, len(FIELDS), figsize=(12.5, 3.2),
        gridspec_kw=dict(wspace=0.42, left=0.045, right=0.995,
                         top=0.78, bottom=0.24),
    )
    for ax, (field, ylabel) in zip(axes, FIELDS):
        for label, fp in fingerprints.items():
            mean, sem, n = fp[field]
            color = SUBSTRATE_COLOR[label]
            ax.plot(GRID, mean, color=color,
                    label=f"{label}  ($n_{{top}}{{=}}{n}$)")
            ax.fill_between(GRID, mean - sem, mean + sem,
                             color=color, alpha=0.18, linewidth=0)
        ax.set_xlabel("relative depth $\\ell / L$", labelpad=4)
        ax.set_title(ylabel, pad=5)
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", which="major", length=3.0, pad=2)

    # Single legend centred above all panels.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, 0.99),
                     ncol=len(handles), handlelength=1.9, columnspacing=2.4,
                     frameon=False,
                     prop={"weight": "bold", "size": 10.0})
    # Also set the math-symbol fonts in the legend (the n_top=100
    # parenthetical contains a subscript) to bold.
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    out_pdf = OUT_DIR / "fig_pareto_fingerprint.pdf"
    out_png = OUT_DIR / "fig_pareto_fingerprint.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
