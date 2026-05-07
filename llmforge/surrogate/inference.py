from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (
    ALL_FIELDS,
    GLOBAL_FIELDS,
    PER_LAYER_FIELDS,
    NormStats,
    _detect_layer_count,
    _encode_attention_variant,
    _resolve_column,
    norm_stats_from_dict,
)
from .model import ArchTransformerRanker

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search_space import Individual


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, padding_mask: torch.Tensor):
        self.x = x
        self.padding_mask = padding_mask

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.padding_mask[idx]


def load_surrogate(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ArchTransformerRanker, NormStats, int]:
    """Load a trained surrogate model from checkpoint.

    Model hyperparameters (d_model, nhead, num_layers, dropout) are loaded
    from a .json config file alongside the .pt checkpoint (same stem).

    Returns:
        model: Loaded ArchTransformerRanker in eval mode
        norm_stats: Normalization statistics from training
        max_layers: Maximum layer count the model supports
    """
    import json as _json

    # Load config JSON alongside checkpoint
    ckpt_path = Path(checkpoint_path)
    config_path = ckpt_path.with_suffix(".json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}. "
            f"A .json sidecar with model hyperparameters is required alongside the .pt checkpoint."
        )
    with open(config_path, "r") as f:
        cfg = _json.load(f)
    mcfg = cfg.get("model", {})
    d_model = mcfg["d_model"]
    nhead = mcfg["nhead"]
    num_layers = mcfg["num_layers"]
    dropout = mcfg.get("dropout", 0.1)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "norm_stats" not in ckpt:
        raise ValueError("Checkpoint is missing norm_stats; cannot normalize inference data")

    norm_stats = norm_stats_from_dict(ckpt["norm_stats"])
    max_layers = ckpt.get("max_layers", 40)

    model = ArchTransformerRanker(
        max_layers=max_layers,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, norm_stats, max_layers


def build_arch_tensor(df: pd.DataFrame, max_layers: int = 40) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """Build architecture tensor and padding mask from DataFrame.

    Active layers are compacted into sequential positions (no holes).
    Masked/inactive layers are skipped entirely. Padding is only at the end.

    Returns:
        x_raw: [B, max_layers, FIELD_COUNT] tensor
        padding_mask: [B, max_layers] bool tensor, True = padding
    """
    from .data import _parse_layer_mask

    n = len(df)
    if n == 0:
        return (
            torch.empty((0, max_layers, len(ALL_FIELDS)), dtype=torch.float32),
            torch.ones((0, max_layers), dtype=torch.bool),
        )

    csv_layers = _detect_layer_count(df)
    has_layer_mask = "global_layer_mask" in df.columns

    global_vals: Dict[str, np.ndarray] = {}
    for gf in GLOBAL_FIELDS:
        col = f"global_{gf}"
        if col not in df.columns:
            raise KeyError(f"Missing global column '{col}' in CSV")
        global_vals[gf] = df[col].astype(float).to_numpy()

    NUMERIC_PER_LAYER = ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]

    # Resolve columns once
    layer_field_cols = {}
    layer_attn_cols = {}
    for li in range(csv_layers):
        layer_field_cols[li] = {}
        for field in NUMERIC_PER_LAYER:
            layer_field_cols[li][field] = _resolve_column(df, li, field)
        try:
            layer_attn_cols[li] = _resolve_column(df, li, "attention_variant")
        except KeyError:
            pass

    x_raw = torch.zeros(n, max_layers, len(ALL_FIELDS), dtype=torch.float32)
    padding_mask = torch.ones(n, max_layers, dtype=torch.bool)

    for row_i in range(n):
        # Parse layer mask
        if has_layer_mask:
            mask = _parse_layer_mask(df["global_layer_mask"].iloc[row_i])
            if mask is None:
                mask = [True] * csv_layers
        else:
            mask = [True] * csv_layers

        # Compact active layers sequentially
        out_pos = 0
        for li in range(csv_layers):
            active = mask[li] if li < len(mask) else False
            if not active:
                continue
            if out_pos >= max_layers:
                break

            for fi, field in enumerate(NUMERIC_PER_LAYER):
                x_raw[row_i, out_pos, fi] = float(df[layer_field_cols[li][field]].iloc[row_i])

            x_raw[row_i, out_pos, 5] = 1.0  # is_active

            if li in layer_attn_cols:
                x_raw[row_i, out_pos, 6] = _encode_attention_variant(df[layer_attn_cols[li]].iloc[row_i])
            else:
                x_raw[row_i, out_pos, 6] = 1.0  # default infinite

            for gi, gf in enumerate(GLOBAL_FIELDS):
                x_raw[row_i, out_pos, len(PER_LAYER_FIELDS) + gi] = global_vals[gf][row_i]

            padding_mask[row_i, out_pos] = False
            out_pos += 1

    return x_raw, padding_mask


def _individuals_to_df(individuals: List[Individual], max_layers: int = 40) -> pd.DataFrame:
    """Convert Individuals to DataFrame with compacted active layers.

    Active layers are packed sequentially (no holes from masked layers).
    Padding columns are filled with zeros at the end.
    """
    rows: List[Dict[str, float]] = []
    for ind in individuals:
        g = ind.get("globals", {})
        layers = ind.get("layers", [])
        mask = g.get("layer_mask", [True] * len(layers))

        row: Dict[str, float] = {}
        for gf in GLOBAL_FIELDS:
            row[f"global_{gf}"] = float(g.get(gf, 0.0))

        # Compact: only emit active layers, sequentially
        out_pos = 0
        for layer_idx in range(len(layers)):
            active = bool(mask[layer_idx]) if layer_idx < len(mask) else True
            if not active:
                continue
            if out_pos >= max_layers:
                break

            li = layers[layer_idx]
            for field in ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]:
                row[f"l{out_pos}_{field}"] = float(li.get(field, 0.0))
            row[f"l{out_pos}_is_active"] = 1.0
            attn_var = li.get("attention_variant", g.get("attention_variant", "infinite"))
            row[f"l{out_pos}_attention_variant"] = _encode_attention_variant(attn_var)
            out_pos += 1

        # Fill remaining positions as padding
        for pad_pos in range(out_pos, max_layers):
            for field in ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]:
                row[f"l{pad_pos}_{field}"] = 0.0
            row[f"l{pad_pos}_is_active"] = 0.0
            row[f"l{pad_pos}_attention_variant"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def _build_tensor_from_individuals_df(df: pd.DataFrame, max_layers: int) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """Build tensor from _individuals_to_df output (uses l{i}_ prefix columns)."""
    n = len(df)
    x_raw = torch.zeros(n, max_layers, len(ALL_FIELDS), dtype=torch.float32)
    padding_mask = torch.ones(n, max_layers, dtype=torch.bool)

    NUMERIC_FIELDS = ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]

    global_vals = {}
    for gf in GLOBAL_FIELDS:
        col = f"global_{gf}"
        global_vals[gf] = df[col].astype(float).to_numpy()

    for li in range(max_layers):
        is_active_col = f"l{li}_is_active"
        if is_active_col not in df.columns:
            break

        # Numeric fields
        for fi, field in enumerate(NUMERIC_FIELDS):
            col = f"l{li}_{field}"
            x_raw[:, li, fi] = torch.tensor(df[col].astype(float).to_numpy(), dtype=torch.float32)

        # is_active
        is_active = df[is_active_col].astype(float).to_numpy()
        x_raw[:, li, 5] = torch.tensor(is_active, dtype=torch.float32)

        # Mark non-padding positions
        padding_mask[:, li] = torch.tensor(is_active < 0.5, dtype=torch.bool)

        # attention_variant
        attn_col = f"l{li}_attention_variant"
        if attn_col in df.columns:
            x_raw[:, li, 6] = torch.tensor(df[attn_col].astype(float).to_numpy(), dtype=torch.float32)

        # Global fields
        for gi, gf in enumerate(GLOBAL_FIELDS):
            x_raw[:, li, len(PER_LAYER_FIELDS) + gi] = torch.tensor(global_vals[gf], dtype=torch.float32)

    return x_raw, padding_mask


def normalize_x(x_raw: torch.FloatTensor, norm: NormStats) -> torch.FloatTensor:
    if x_raw.numel() == 0:
        return x_raw
    xs = []
    for fi, field in enumerate(ALL_FIELDS):
        fs = norm.stats[field]
        rng = max(fs.vmax - fs.vmin, 1e-9)
        vals = (x_raw[:, :, fi] - fs.vmin) / rng
        vals = torch.clamp(vals, 0.0, 1.0)
        xs.append(vals.unsqueeze(-1))
    return torch.cat(xs, dim=-1)


def surrogate_eval(
    individuals: List[Individual],
    model: ArchTransformerRanker,
    norm: NormStats,
    device: torch.device,
    max_layers: int = 40,
    batch_size: int = 256,
) -> List[float]:
    """Predict validation losses for a list of Individuals."""
    if not individuals:
        return []

    df = _individuals_to_df(individuals, max_layers=max_layers)
    x_raw, padding_mask = _build_tensor_from_individuals_df(df, max_layers)
    x_norm = normalize_x(x_raw, norm)

    ds = TensorDataset(x_norm, padding_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            score = model(x, padding_mask=mask)
            preds.append(score.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).astype(float).tolist()


def run_inference_on_chunk(
    model: ArchTransformerRanker,
    df: pd.DataFrame,
    norm: NormStats,
    device: torch.device,
    batch_size: int,
    max_layers: int = 40,
) -> np.ndarray:
    x_raw, padding_mask = build_arch_tensor(df, max_layers=max_layers)
    x_norm = normalize_x(x_raw, norm)
    if x_norm.numel() == 0:
        return np.array([])

    ds = TensorDataset(x_norm, padding_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for x, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            score = model(x, padding_mask=mask)
            preds.append(score.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def estimate_params_chunk(df: pd.DataFrame, max_layers: int = 40) -> np.ndarray:
    """Estimate parameter count per row using the same logic as search_space.estimate_params."""
    if len(df) == 0:
        return np.array([], dtype=np.int64)

    vocab_size = 50257
    if "global_n_embd" not in df.columns:
        raise KeyError("Missing global column 'global_n_embd' in CSV")
    d = df["global_n_embd"].astype(float).to_numpy()

    csv_layers = _detect_layer_count(df)

    total = vocab_size * d.astype(np.float64)

    for layer in range(min(csv_layers, max_layers)):
        h = df[_resolve_column(df, layer, "n_head")].astype(float).to_numpy()
        m = df[_resolve_column(df, layer, "mlp_size")].astype(float).to_numpy()
        qk = df[_resolve_column(df, layer, "n_qk_head_dim")].astype(float).to_numpy()
        v = df[_resolve_column(df, layer, "n_v_head_dim")].astype(float).to_numpy()
        try:
            nkg_col = _resolve_column(df, layer, "n_kv_group")
            n_kv_group = df[nkg_col].astype(float).to_numpy()
        except KeyError:
            n_kv_group = h

        # Determine attention variant per row
        try:
            attn_col = _resolve_column(df, layer, "attention_variant")
            attn_vals = df[attn_col].values
        except KeyError:
            attn_vals = ["infinite"] * len(df)

        n_cproj = 1.0
        use_concat_heads = True

        attn_cost = np.zeros_like(d, dtype=np.float64)

        for row_i in range(len(df)):
            av = str(attn_vals[row_i]).strip().lower() if not isinstance(attn_vals[row_i], (int, float)) else "infinite"
            if av == "identity":
                continue  # zero cost
            elif av == "infinite":
                q_p = d[row_i] * (h[row_i] * qk[row_i])
                k_p = d[row_i] * (n_kv_group[row_i] * qk[row_i])
                v_p = d[row_i] * (n_kv_group[row_i] * v[row_i])
                if use_concat_heads:
                    o_p = (h[row_i] * v[row_i]) * d[row_i]
                else:
                    o_p = n_cproj * (v[row_i] * d[row_i])
                attn_cost[row_i] = q_p + k_p + v_p + o_p
            elif av in {"causal", "mha"}:
                qkv_p = d[row_i] * (h[row_i] * (qk[row_i] + qk[row_i] + v[row_i]))
                if use_concat_heads:
                    out_p = (h[row_i] * v[row_i]) * d[row_i]
                else:
                    out_p = n_cproj * (v[row_i] * d[row_i])
                attn_cost[row_i] = qkv_p + out_p

        mlp_params = 2 * d * m
        total = total + attn_cost + mlp_params

    return total.astype(np.int64)
