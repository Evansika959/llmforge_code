"""Quantitative depth-shape analysis on top-100 non-dominated archs per substrate.

Computes three substrate-discriminating signatures the qualitative
fingerprint plot only hints at:

  1. Slope / curvature / range per (substrate, field): describe whether
     the depth profile rises, falls, or peaks in the middle.
  2. Compute concentration index: fraction of per-arch active-layer
     FLOPs in the bottom-third / middle-third / top-third of depth.
  3. Cross-field interaction: correlation between attention-compute and
     MLP-compute along depth, within-arch and across the top-100 pool.

Reuses the same top-N selection logic and ckpt set as pareto_trends.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from pareto_trends import (  # noqa: E402
    SUBSTRATES, TOP_N, GRID,
    collect_unique_archs, top_n_indices,
    active_layer_sequence,
)


# ── Per-layer FLOP proxies (per token; constants cancel in fractions) ──────

def attention_flops(layer: dict, d_embd: int, seq: int) -> float:
    if layer.get("attention_variant", "infinite") == "identity":
        return 0.0
    h = layer["n_head"]
    kv = layer["n_kv_group"]
    dqk = layer["n_qk_head_dim"]
    dv = layer["n_v_head_dim"]
    # Q-proj + K-proj + V-proj (input × hidden_in × hidden_out, fwd ×2)
    qkv = 2.0 * d_embd * (h * dqk + kv * dqk + kv * dv)
    # QK^T and PV (per token, h heads, seq context)
    attn_kernel = 2.0 * h * seq * (dqk + dv)
    # Output projection
    oproj = 2.0 * d_embd * h * dv
    return qkv + attn_kernel + oproj


def mlp_flops(layer: dict, d_embd: int) -> float:
    return 4.0 * d_embd * layer["mlp_size"]   # FC1 + FC2


# ── Per-arch metrics ───────────────────────────────────────────────────────

def per_arch_layer_flops(individual: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (attn_flops_per_active_layer, mlp_flops_per_active_layer)."""
    g = individual["globals"]
    layers = individual["layers"]
    mask = g.get("layer_mask", [True] * len(layers))
    d = g["n_embd"]
    seq = g.get("block_size", 256)
    attn, mlp = [], []
    for i, active in enumerate(mask):
        if not active or i >= len(layers):
            continue
        attn.append(attention_flops(layers[i], d, seq))
        mlp.append(mlp_flops(layers[i], d))
    return np.asarray(attn), np.asarray(mlp)


def compute_concentration(individual: dict) -> Tuple[float, float, float]:
    """Per-arch fraction of total layer-FLOPs in (bottom, middle, top) thirds of depth."""
    a, m = per_arch_layer_flops(individual)
    total_per_layer = a + m
    n = len(total_per_layer)
    if n == 0 or total_per_layer.sum() == 0:
        return 0.0, 0.0, 0.0
    b = n // 3
    t = n - n // 3
    bottom = total_per_layer[:b].sum()
    middle = total_per_layer[b:t].sum()
    top = total_per_layer[t:].sum()
    s = bottom + middle + top
    return bottom / s, middle / s, top / s


def shape_signature(curve: np.ndarray) -> Tuple[float, float, float]:
    """slope, curvature, range — where slope = late-third mean − early-third mean,
    curvature = middle-third mean − ½(early+late), range = max − min, all in raw units."""
    finite = np.isfinite(curve)
    if finite.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    n = len(curve)
    third = n // 3
    early = np.nanmean(curve[:third])
    middle = np.nanmean(curve[third:n-third])
    late = np.nanmean(curve[n-third:])
    slope = late - early
    curvature = middle - 0.5 * (early + late)
    rng = float(np.nanmax(curve) - np.nanmin(curve))
    return slope, curvature, rng


# ── Per-substrate aggregation ──────────────────────────────────────────────

def analyze(label: str, ckpt_path: Path):
    inds, objs, cons = collect_unique_archs(ckpt_path)
    picks = top_n_indices(objs, cons, TOP_N)
    pool = [inds[i] for i in picks]

    # 1. Slope / curvature / range from the per-substrate fingerprint mean
    fields = ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]
    pretty = {"n_head": "n_h", "n_kv_group": "n_kv",
              "n_qk_head_dim": "d_qk", "n_v_head_dim": "d_v",
              "mlp_size": "d_mlp"}
    field_curves = {f: [] for f in fields}
    for ind in pool:
        seqs, _ = active_layer_sequence(ind)
        for f in fields:
            arch_seq = seqs[f]
            if len(arch_seq) == 0:
                continue
            finite = np.isfinite(arch_seq)
            if not finite.any():
                field_curves[f].append(np.full(len(GRID), np.nan))
                continue
            src = np.linspace(0, 1, int(finite.sum()))
            field_curves[f].append(np.interp(GRID, src, arch_seq[finite]))
    sig = {}
    for f in fields:
        stack = np.stack(field_curves[f], axis=0)
        mean = np.nanmean(stack, axis=0)
        sig[f] = shape_signature(mean)

    # 2. Compute concentration index
    triples = np.array([compute_concentration(ind) for ind in pool])
    bot_med, mid_med, top_med = np.median(triples, axis=0)

    # 3. Within-arch attention-vs-MLP correlation by depth
    attn_curves = []
    mlp_curves = []
    for ind in pool:
        a, m = per_arch_layer_flops(ind)
        if len(a) < 4:
            continue
        # Resample onto the grid
        src = np.linspace(0, 1, len(a))
        attn_curves.append(np.interp(GRID, src, a))
        mlp_curves.append(np.interp(GRID, src, m))
    attn_stack = np.stack(attn_curves, axis=0)
    mlp_stack = np.stack(mlp_curves, axis=0)
    # Correlation across grid points (per arch), then mean across pool
    per_arch_rho = []
    for av, mv in zip(attn_stack, mlp_stack):
        if np.std(av) < 1e-6 or np.std(mv) < 1e-6:
            continue
        per_arch_rho.append(float(np.corrcoef(av, mv)[0, 1]))
    rho_mean = float(np.mean(per_arch_rho)) if per_arch_rho else float("nan")

    return label, sig, (bot_med, mid_med, top_med), rho_mean, pretty, fields


def main():
    print("Loading and analyzing top-100 archs per substrate...\n")
    results = []
    for label, ckpt in SUBSTRATES:
        if not Path(ckpt).exists():
            print(f"[skip] {label}: missing {ckpt}")
            continue
        results.append(analyze(label, ckpt))

    # ── Report 1: shape signatures ──
    pretty = results[0][4]
    fields = results[0][5]
    print("=" * 100)
    print("SIGNATURE 1 — Per-field shape (slope = late−early; curv = middle − ½(early+late); range)")
    print("=" * 100)
    for f in fields:
        print(f"\n  Field: {pretty[f]}")
        print(f"    {'substrate':<22} {'slope':>10} {'curvature':>11} {'range':>10}     interpretation")
        for label, sig, _, _, _, _ in results:
            sl, cu, rg = sig[f]
            interp = ""
            if abs(sl) < 0.1 * rg:
                interp = "flat"
            elif sl > 0:
                interp = "rises with depth"
            else:
                interp = "falls with depth"
            if abs(cu) > 0.2 * rg:
                interp += "; middle-" + ("heavy" if cu > 0 else "light")
            print(f"    {label:<22} {sl:>+10.2f} {cu:>+11.2f} {rg:>10.2f}     {interp}")

    # ── Report 2: compute concentration ──
    print("\n" + "=" * 100)
    print("SIGNATURE 2 — Compute-concentration index (% of arch's active-layer FLOPs by depth third)")
    print("=" * 100)
    print(f"\n  {'substrate':<22} {'bottom-3rd':>12} {'middle-3rd':>12} {'top-3rd':>10}     interpretation")
    for label, _, conc, _, _, _ in results:
        b, m, t = conc
        if b > max(m, t) + 0.04:
            interp = "front-loaded compute"
        elif t > max(b, m) + 0.04:
            interp = "back-loaded compute"
        elif m > max(b, t) + 0.04:
            interp = "middle-heavy compute"
        else:
            interp = "uniform along depth"
        print(f"  {label:<22} {b*100:>11.1f}% {m*100:>11.1f}% {t*100:>9.1f}%     {interp}")

    # ── Report 3: attn-vs-MLP within-arch correlation ──
    print("\n" + "=" * 100)
    print("SIGNATURE 3 — Within-arch attention-vs-MLP FLOP correlation across depth")
    print("=" * 100)
    print(f"\n  {'substrate':<22} {'mean ρ':>10}     interpretation")
    for label, _, _, rho, _, _ in results:
        if rho < -0.2:
            interp = "trade-off (attention and MLP compete for capacity by layer)"
        elif rho > 0.5:
            interp = "co-scale (attention and MLP grow together)"
        else:
            interp = "near-independent (per-layer assignment is decoupled)"
        print(f"  {label:<22} {rho:>+10.3f}     {interp}")


if __name__ == "__main__":
    main()
