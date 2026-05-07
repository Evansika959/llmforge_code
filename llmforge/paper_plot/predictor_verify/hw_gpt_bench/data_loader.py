"""HW-GPT-Bench (Sukthanker et al., NeurIPS D&B 2024) dataset adapter.

Loads `stats.pkl` from the HW-GPT-Bench release (10K architecture-perplexity
pairs per scale s/m/l) and converts each architecture into ForgeFormer's
ArchBatchRaw format with FIELD_COUNT=5 (mlp_ratio, n_head, is_active,
embed_dim, bias).

Per-token feature vector layout matches the legacy V1 pipeline:
    [ mlp_ratio, n_head, is_active, embed_dim, bias ]
with embed_dim and bias broadcast across all max_layers positions.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

HWGPT_REPO = Path("$HOME/hw-gpt-bench")
SCALE_TO_MAX_LAYERS = {"s": 12, "m": 24, "l": 36}
FIELD_COUNT = 5  # mlp_ratio, n_head, is_active, embed_dim, bias

# HW-GPT-Bench search-space cardinality (verbatim from hw-gpt-bench/lib/utils.py).
# Used by the paper-baseline MLP to one-hot-encode each field.
HWGPT_SEARCH_SPACES = {
    "s": dict(embed_dim=[192, 384, 768], n_layer=[10, 11, 12],
              mlp_ratio=[2, 3, 4], n_head=[4, 8, 12], bias=["True", "False"]),
    "m": dict(embed_dim=[256, 512, 1024], n_layer=[22, 23, 24],
              mlp_ratio=[2, 3, 4], n_head=[8, 12, 16], bias=["True", "False"]),
    "l": dict(embed_dim=[320, 640, 1280], n_layer=[34, 35, 36],
              mlp_ratio=[2, 3, 4], n_head=[8, 16, 20], bias=["True", "False"]),
}


@dataclass
class HWGPTBatch:
    x_raw: torch.FloatTensor       # [N, max_layers, FIELD_COUNT]
    padding_mask: torch.BoolTensor  # [N, max_layers] True = padding
    val_loss: torch.FloatTensor     # [N] (perplexity, used as the regression target)


def parse_arch_str(arch_str: str, scale: str) -> dict:
    """Parse a HW-GPT-Bench architecture string of form
    `gpt-<scale>-<n_layer>-<embed_dim>-<mlp×L>-<head×L>-<bias>`."""
    parts = arch_str.split("-")
    assert parts[0] == "gpt"
    assert parts[1] == scale, f"scale mismatch: arch is {parts[1]}, expected {scale}"
    n_layer = int(parts[2])
    embed_dim = int(parts[3])
    mlp_ratios = [int(x) for x in parts[4:4 + n_layer]]
    n_heads = [int(x) for x in parts[4 + n_layer:4 + 2 * n_layer]]
    bias = 1 if parts[-1] == "True" else 0
    return dict(n_layer=n_layer, embed_dim=embed_dim,
                mlp_ratios=mlp_ratios, n_heads=n_heads, bias=bias)


def load_hwgpt_archs(scale: str, repo: Path = HWGPT_REPO):
    pkl_path = repo / f"data_collection/gpt_datasets/gpt_{scale}/stats.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    archs, ppls = [], []
    for arch_str, metrics in data.items():
        ppl = metrics.get("perplexity")
        if ppl is None or not np.isfinite(float(ppl)):
            continue
        archs.append(parse_arch_str(arch_str, scale))
        ppls.append(float(ppl))
    return archs, np.asarray(ppls, dtype=np.float32)


def build_batch(archs: List[dict], scale: str) -> HWGPTBatch:
    max_layers = SCALE_TO_MAX_LAYERS[scale]
    n = len(archs)
    x_raw = np.zeros((n, max_layers, FIELD_COUNT), dtype=np.float32)
    pad = np.ones((n, max_layers), dtype=bool)  # True = padding
    for i, a in enumerate(archs):
        L = a["n_layer"]
        for j in range(L):
            x_raw[i, j, 0] = a["mlp_ratios"][j]
            x_raw[i, j, 1] = a["n_heads"][j]
            x_raw[i, j, 2] = 1.0  # is_active
            x_raw[i, j, 3] = a["embed_dim"]  # broadcast global
            x_raw[i, j, 4] = a["bias"]       # broadcast global
        pad[i, :L] = False
    return HWGPTBatch(
        x_raw=torch.from_numpy(x_raw),
        padding_mask=torch.from_numpy(pad),
        val_loss=torch.zeros(n, dtype=torch.float32),  # filled by caller
    )


def normalize_per_field(x_raw: torch.Tensor, padding_mask: torch.Tensor):
    """Min-max normalize each field over *active* (non-padding) positions only.

    Returns (x_norm, mins, span) where x_norm has zeroed padding positions.
    """
    pad_np = padding_mask.numpy()
    x_np = x_raw.numpy()
    mask_flat = (~pad_np).reshape(-1)
    x_flat = x_np.reshape(-1, FIELD_COUNT)
    mins = x_flat[mask_flat].min(axis=0)
    maxs = x_flat[mask_flat].max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0).astype(np.float32)
    x_norm = (x_np - mins) / span
    x_norm = np.where(pad_np[..., None], 0.0, x_norm).astype(np.float32)
    return torch.from_numpy(x_norm), mins.astype(np.float32), span


def one_hot_features(archs: List[dict], scale: str) -> np.ndarray:
    """One-hot encoding matching the HW-GPT-Bench paper's `Net` input.

    Layout per architecture (concatenated):
        embed_dim (one-hot, 3 dims)
      ⊕ n_layer   (one-hot, 3 dims)
      ⊕ mlp_ratio per layer × max_layers (one-hot, 3 × max_layers dims)
      ⊕ n_head per layer × max_layers (one-hot, 3 × max_layers dims)
      ⊕ bias (one-hot, 2 dims)
    Layers beyond the architecture's actual depth are left as zeros (no hot).

    For gpt_l (max_layers=36): 3+3+108+108+2 = 224 dims.
    """
    cd = HWGPT_SEARCH_SPACES[scale]
    max_layers = max(cd["n_layer"])
    rows = []
    for a in archs:
        e = np.zeros(len(cd["embed_dim"]), dtype=np.float32)
        l = np.zeros(len(cd["n_layer"]), dtype=np.float32)
        m = np.zeros((max_layers, len(cd["mlp_ratio"])), dtype=np.float32)
        h = np.zeros((max_layers, len(cd["n_head"])), dtype=np.float32)
        b = np.zeros(len(cd["bias"]), dtype=np.float32)
        e[cd["embed_dim"].index(a["embed_dim"])] = 1.0
        l[cd["n_layer"].index(a["n_layer"])] = 1.0
        for j in range(a["n_layer"]):
            m[j][cd["mlp_ratio"].index(a["mlp_ratios"][j])] = 1.0
            h[j][cd["n_head"].index(a["n_heads"][j])] = 1.0
        bias_str = "True" if a["bias"] == 1 else "False"
        b[cd["bias"].index(bias_str)] = 1.0
        rows.append(np.concatenate([e, l, m.reshape(-1), h.reshape(-1), b]))
    return np.stack(rows).astype(np.float32)


def flatten_for_baselines(archs: List[dict], scale: str) -> np.ndarray:
    """Flat per-architecture feature vector for RF / MLP baselines.

    Layout: [embed_dim, bias, n_layer, mlp_ratios×max_layers (pad -1),
             n_heads×max_layers (pad -1)]
    """
    max_layers = SCALE_TO_MAX_LAYERS[scale]
    rows = []
    for a in archs:
        L = a["n_layer"]
        v = [a["embed_dim"], a["bias"], L]
        v += a["mlp_ratios"] + [-1] * (max_layers - L)
        v += a["n_heads"] + [-1] * (max_layers - L)
        rows.append(v)
    return np.asarray(rows, dtype=np.float32)


def load_hwgpt_dataset(scale: str = "l", repo: Path = HWGPT_REPO
                       ) -> Tuple[HWGPTBatch, np.ndarray, np.ndarray, dict]:
    """Convenience: load + tensorize. Returns (norm_batch, X_flat, ppls, info)."""
    archs, ppls = load_hwgpt_archs(scale, repo)
    batch = build_batch(archs, scale)
    batch.val_loss = torch.from_numpy(ppls)
    x_norm, mins, span = normalize_per_field(batch.x_raw, batch.padding_mask)
    norm_batch = HWGPTBatch(x_raw=x_norm, padding_mask=batch.padding_mask, val_loss=batch.val_loss)
    X_flat = flatten_for_baselines(archs, scale)
    info = dict(scale=scale, n=len(archs), max_layers=SCALE_TO_MAX_LAYERS[scale],
                ppl_min=float(ppls.min()), ppl_max=float(ppls.max()),
                norm_mins=mins, norm_span=span)
    return norm_batch, X_flat, ppls, info
