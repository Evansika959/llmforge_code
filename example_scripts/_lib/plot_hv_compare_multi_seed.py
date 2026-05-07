#!/usr/bin/env python3
"""Multi-seed companion of plot_hv_compare.py.

Each run is specified by a (label, color, ckpt_dirs) triple where ckpt_dirs is
a comma-separated list of NSGA-II checkpoint directories, one per seed. The
script computes one hypervolume trajectory per (label, seed) pair, then plots
the per-generation mean across seeds with a shaded +/-1 standard deviation
band. Per-axis min and max for HV normalization are taken over the union of
all (run, seed) populations so HV values are directly comparable across the
overlaid curves.

Example:
    python _lib/plot_hv_compare_multi_seed.py \\
        --run "NSGA + IHA:#1f77b4:/path/seed42,/path/seed1,/path/seed7" \\
        --run "Random + IHA:#888888:/path/seed42,/path/seed1,/path/seed7" \\
        --run "NSGA + GQA:#ff7f0e:/path/seed42,/path/seed1,/path/seed7" \\
        --obj-keys val_loss params_M \\
        --out search_ablation_hv_multi_seed.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pymoo.indicators.hv import HV

DEFAULT_OBJ_KEYS = ("val_loss", "e_per_tok_uJ", "tpot_ms", "ttft_ms")
HV_REF_MARGIN = 1.05  # Reference point: 5% past the worst observed value per axis.

KEY_FALLBACKS = {
    "val_loss":     [("val_loss", 1.0)],
    "params_M":     [("params_M", 1.0), ("params", 1.0)],
    "e_per_tok_uJ": [("energy_per_token_uJ", 1.0)],
    "ttft_ms":      [("ttft_ms", 1.0), ("ttft", 1e3)],
    "tpot_ms":      [("tpot_ms", 1.0), ("tpot", 1e3)],
}


# --------------------------------------------------------------------------- #
# Data loading (per-seed)
# --------------------------------------------------------------------------- #

def _extract(aux: dict, canonical_key: str) -> float:
    for raw_key, scale in KEY_FALLBACKS.get(canonical_key, [(canonical_key, 1.0)]):
        v = aux.get(raw_key)
        if v is None:
            continue
        try:
            f = float(v) * float(scale)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return f
    return float("nan")


def load_gen(gen: int, ckpt_dir: str, obj_keys: tuple[str, ...]) -> list[dict]:
    matches = sorted(glob.glob(f"{ckpt_dir}/*ckpt_gen{gen}.json"))
    plain = [p for p in matches if Path(p).name == f"ckpt_gen{gen}.json"]
    path = (plain or matches or [None])[0]
    if path is None:
        return []
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for ev in d.get("evaluations", []):
        if ev is None:
            continue
        a = ev.get("aux") or {}
        row = {k: _extract(a, k) for k in obj_keys}
        if all(np.isfinite(row[k]) for k in obj_keys):
            out.append(row)
    return out


def autodetect_max_gen(ckpt_dir: str) -> int:
    files = glob.glob(f"{ckpt_dir}/*ckpt_gen*.json")
    gens = []
    for f in files:
        name = Path(f).stem
        try:
            gens.append(int(name.rsplit("gen", 1)[-1]))
        except ValueError:
            pass
    return max(gens) if gens else 0


def load_seed(ckpt_dir: str, obj_keys: tuple[str, ...]) -> dict[int, list[dict]]:
    g_max = autodetect_max_gen(ckpt_dir)
    gens = {}
    for g in range(g_max + 1):
        pop = load_gen(g, ckpt_dir, obj_keys)
        if pop:
            gens[g] = pop
    if not gens:
        raise SystemExit(
            f"No usable checkpoints under {ckpt_dir} for objectives {obj_keys}. "
            f"Check that aux fields contain values for these keys."
        )
    return gens


# --------------------------------------------------------------------------- #
# Hypervolume aggregation across seeds
# --------------------------------------------------------------------------- #

def compute_hv_curves(curves: list[dict], obj_keys: tuple[str, ...]) -> list[dict]:
    """For each curve, populate `xs` (gen indices), `mean` (HV mean across
    seeds at each gen), and `std` (HV std across seeds). Per-axis lo/hi
    normalization is shared across all (run, seed) populations."""
    n_obj = len(obj_keys)
    all_pts = np.array([
        [d[k] for k in obj_keys]
        for c in curves for seed_gens in c["seeds"]
        for p in seed_gens.values() for d in p
    ])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    indicator = HV(ref_point=np.full(n_obj, HV_REF_MARGIN))

    for c in curves:
        # Compute HV per (seed, gen).
        per_seed_hv = []
        for seed_gens in c["seeds"]:
            sorted_gens = sorted(seed_gens)
            hv_seq = []
            for g in sorted_gens:
                m = np.array([[d[k] for k in obj_keys] for d in seed_gens[g]])
                hv_seq.append(float(indicator((m - lo) / span)))
            per_seed_hv.append({"xs": sorted_gens, "ys": hv_seq})

        # Align to common gen axis -- take the intersection of generations
        # observed across all seeds for this run so the mean/std are computed
        # over the same generation indices in all seeds.
        common_gens = sorted(set.intersection(*[set(p["xs"]) for p in per_seed_hv]))
        if not common_gens:
            raise SystemExit(
                f"No common generations across seeds for run {c['label']!r}; "
                f"per-seed gens: {[p['xs'] for p in per_seed_hv]}"
            )
        gen_to_idx = [{g: i for i, g in enumerate(p["xs"])} for p in per_seed_hv]
        mat = np.array([
            [per_seed_hv[s]["ys"][gen_to_idx[s][g]] for g in common_gens]
            for s in range(len(per_seed_hv))
        ])  # shape (n_seeds, n_gens)
        c["xs"] = common_gens
        c["mean"] = mat.mean(axis=0)
        c["std"] = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
        c["per_seed_final"] = mat[:, -1]
    return curves


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def style_neurips():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })


def plot_compare(curves: list[dict], out_pdf: Path, title: str | None = None):
    style_neurips()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for c in curves:
        xs = np.array(c["xs"])
        mean = np.array(c["mean"])
        std = np.array(c["std"])
        ax.fill_between(xs, mean - std, mean + std, color=c["color"],
                        alpha=0.20, linewidth=0, zorder=2)
        ax.plot(xs, mean, "-", color=c["color"], lw=2.0,
                marker="o", ms=4.0, mec="white", mew=0.5,
                label=c["label"], zorder=3)
    # Bold axis labels: rely on labelweight for the Latin parts and wrap
    # mathtext explicitly in \mathbf so the math glyphs render bold too.
    ax.set_xlabel(r"NSGA-II generation")
    ax.set_ylabel(r"Hypervolume (mean $\mathbf{\pm 1\sigma}$) $\mathbf{\uparrow}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, alpha=0.30, linewidth=0.5, color="#d0d0d0", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(loc="lower right", framealpha=0.9, edgecolor="none",
                    handletextpad=0.5, borderpad=0.4,
                    prop={"weight": "bold", "size": 16})
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if title:
        ax.set_title(title, pad=8)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    print(f"[ok] wrote {out_pdf}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_run_spec(spec: str) -> dict:
    """`label:color:dir1,dir2,...` -> dict. Label may not contain a colon."""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(
            f"--run must be 'label:color:dir1,dir2,...', got {spec!r}"
        )
    label = parts[0].strip()
    color = parts[1].strip()
    dirs = [p.strip() for p in parts[2].split(",") if p.strip()]
    if not dirs:
        raise SystemExit(f"--run {spec!r} has no checkpoint directories")
    return {"label": label, "color": color, "dirs": dirs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL:COLOR:DIR1,DIR2,...",
                    help="Repeatable. Comma-separated DIRs are independent "
                         "seeds of the same configuration.")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--title", default=None,
                    help="Optional axis title (kept short)")
    ap.add_argument("--obj-keys", nargs="+", default=list(DEFAULT_OBJ_KEYS),
                    help="Objective keys to compute HV over. Default: 4D HW+loss "
                         "set (val_loss, e_per_tok_uJ, tpot_ms, ttft_ms). For "
                         "search-strategy ablations on (val_loss, params_M) use "
                         "'--obj-keys val_loss params_M'.")
    args = ap.parse_args()

    if len(args.run) < 2:
        raise SystemExit("Need at least two --run entries to compare.")

    obj_keys = tuple(args.obj_keys)
    print(f"[hv] objective space: {obj_keys}")

    curves = []
    for spec in args.run:
        r = parse_run_spec(spec)
        seeds = [load_seed(d, obj_keys) for d in r["dirs"]]
        curves.append({"label": r["label"], "color": r["color"],
                       "dirs": r["dirs"], "seeds": seeds})
        print(f"[load] {r['label']!r}: {len(seeds)} seeds, "
              f"{[len(s) for s in seeds]} gens each")

    compute_hv_curves(curves, obj_keys)

    print()
    print("[hv] final-generation HV summary (across seeds):")
    for c in curves:
        finals = c["per_seed_final"]
        print(f"  {c['label']:<25s}  HV(final) = {finals.mean():.3f} "
              f"+/- {finals.std(ddof=1) if len(finals) > 1 else 0.0:.3f}  "
              f"(n_seeds = {len(finals)}, per-seed = "
              f"{', '.join(f'{v:.3f}' for v in finals)})")

    plot_compare(curves, Path(args.out), title=args.title)


if __name__ == "__main__":
    main()
