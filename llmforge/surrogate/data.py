import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Attention variant encoding: categorical → float
ATTN_VARIANT_MAP = {
    "identity": 0.0,
    "infinite": 1.0,
    "causal": 2.0,
    "mha": 3.0,
}

PER_LAYER_FIELDS = [
    "n_head",
    "n_kv_group",
    "n_qk_head_dim",
    "n_v_head_dim",
    "mlp_size",
    "is_active",          # 1.0 = real layer, 0.0 = padding
    "attention_variant",  # float-encoded via ATTN_VARIANT_MAP
]
# Global numeric fields to be broadcast across layers so the model can see them
GLOBAL_FIELDS = [
    "n_embd",
    "block_size",
]
ALL_FIELDS = PER_LAYER_FIELDS + GLOBAL_FIELDS
FIELD_COUNT = len(ALL_FIELDS)

EPS = 1e-9


@dataclass
class FieldStats:
    vmin: float
    vmax: float


@dataclass
class NormStats:
    stats: Dict[str, FieldStats]


@dataclass
class ArchBatch:
    x: torch.FloatTensor       # [B, L, FIELD_COUNT] normalized floats
    padding_mask: torch.BoolTensor  # [B, L] True = padding (ignore)
    val_loss: torch.FloatTensor     # [B]


@dataclass
class ArchBatchRaw:
    x_raw: torch.FloatTensor       # [B, L, FIELD_COUNT] raw numeric values
    padding_mask: torch.BoolTensor  # [B, L] True = padding (ignore)
    val_loss: torch.FloatTensor     # [B]


def _canonical_columns(layer: int, field: str) -> List[str]:
    # Try common patterns: l0_n_head, layer0_n_head, l0.n_head, etc.
    bases = [f"l{layer}", f"layer{layer}"]
    cols = []
    for b in bases:
        cols.append(f"{b}_{field}")
        cols.append(f"{b}.{field}")
    return cols


def _resolve_column(df: pd.DataFrame, layer: int, field: str) -> str:
    candidates = _canonical_columns(layer, field)
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in lower_cols:
            return lower_cols[lc]
    # fallback: any column containing both layer tag and field
    for col in df.columns:
        lc = col.lower()
        if f"{layer}" in lc and field in lc:
            return col
    raise KeyError(f"Could not find column for layer {layer}, field {field}")


def _detect_layer_count(df: pd.DataFrame) -> int:
    """Auto-detect the number of layers in a CSV by counting layer{i}_n_head columns."""
    count = 0
    while True:
        try:
            _resolve_column(df, count, "n_head")
            count += 1
        except KeyError:
            break
    if count == 0:
        raise ValueError("Could not detect any layer columns (layer0_n_head / l0_n_head) in CSV")
    return count


def _parse_layer_mask(mask_str) -> List[bool]:
    """Parse global_layer_mask from CSV (e.g., '[True, True, False]')."""
    if isinstance(mask_str, list):
        return [bool(v) for v in mask_str]
    if isinstance(mask_str, str):
        try:
            parsed = ast.literal_eval(mask_str)
            if isinstance(parsed, list):
                return [bool(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    # fallback: all active
    return None


def _encode_attention_variant(val) -> float:
    """Encode attention_variant string to float."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    return ATTN_VARIANT_MAP.get(s, 1.0)  # default to infinite


def load_raw_arch_dataset(csv_path: str, max_layers: int = 40) -> ArchBatchRaw:
    df = pd.read_csv(csv_path)
    # Drop rows with non-finite targets to avoid inf/NaN losses during training/validation
    valid_mask = np.isfinite(df["val_loss"].values)
    dropped = int((~valid_mask).sum())
    if dropped:
        print(f"[data] Dropping {dropped} rows with non-finite val_loss from {csv_path}")
    df = df[valid_mask].reset_index(drop=True)

    csv_layers = _detect_layer_count(df)

    # Check for layer_mask column
    has_layer_mask = "global_layer_mask" in df.columns

    # Check for attention_variant columns per layer
    has_attn_variant = {}
    for li in range(csv_layers):
        try:
            _resolve_column(df, li, "attention_variant")
            has_attn_variant[li] = True
        except KeyError:
            has_attn_variant[li] = False

    n = len(df)

    # Preload global values
    global_vals = {}
    for gf in GLOBAL_FIELDS:
        col = f"global_{gf}"
        if col not in df.columns:
            raise KeyError(f"Missing global column '{col}' in CSV")
        global_vals[gf] = df[col].astype(float).values

    # Parse layer masks for all rows
    row_masks: List[List[bool]] = []
    if has_layer_mask:
        for i in range(n):
            mask = _parse_layer_mask(df["global_layer_mask"].iloc[i])
            if mask is None:
                mask = [True] * csv_layers
            row_masks.append(mask)
    else:
        row_masks = [[True] * csv_layers] * n

    # Resolve columns once per layer
    NUMERIC_PER_LAYER = ["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]
    layer_field_cols = {}
    layer_attn_cols = {}
    for li in range(csv_layers):
        layer_field_cols[li] = {}
        for field in NUMERIC_PER_LAYER:
            layer_field_cols[li][field] = _resolve_column(df, li, field)
        if has_attn_variant.get(li, False):
            layer_attn_cols[li] = _resolve_column(df, li, "attention_variant")

    # Compact active layers: extract only active layers per row, pack sequentially,
    # pad at the end. This ensures that architectures with the same active layers
    # in the same order get identical representations regardless of their slot positions.
    x_raw = torch.zeros(n, max_layers, FIELD_COUNT, dtype=torch.float32)
    padding_mask = torch.ones(n, max_layers, dtype=torch.bool)  # True = padding

    for row_i in range(n):
        out_pos = 0  # sequential position for compacted active layers
        for li in range(csv_layers):
            active = row_masks[row_i][li] if li < len(row_masks[row_i]) else False
            if not active:
                continue  # skip masked layers entirely
            if out_pos >= max_layers:
                break

            # Numeric fields (indices 0-4)
            for fi, field in enumerate(NUMERIC_PER_LAYER):
                x_raw[row_i, out_pos, fi] = float(df[layer_field_cols[li][field]].iloc[row_i])

            # is_active (index 5) — always 1.0 for compacted layers
            x_raw[row_i, out_pos, 5] = 1.0

            # attention_variant (index 6)
            if li in layer_attn_cols:
                x_raw[row_i, out_pos, 6] = _encode_attention_variant(df[layer_attn_cols[li]].iloc[row_i])
            else:
                x_raw[row_i, out_pos, 6] = 1.0  # default to infinite

            # Global fields (indices 7, 8)
            for gi, gf in enumerate(GLOBAL_FIELDS):
                x_raw[row_i, out_pos, len(PER_LAYER_FIELDS) + gi] = global_vals[gf][row_i]

            padding_mask[row_i, out_pos] = False
            out_pos += 1

    val_loss = torch.tensor(df["val_loss"].values, dtype=torch.float32)
    return ArchBatchRaw(x_raw=x_raw, padding_mask=padding_mask, val_loss=val_loss)


def compute_norm_stats(batch_raw: ArchBatchRaw) -> NormStats:
    stats: Dict[str, FieldStats] = {}
    # Compute stats only over non-padding positions
    mask = ~batch_raw.padding_mask  # [B, L] True = real
    for fi, field in enumerate(ALL_FIELDS):
        field_vals = batch_raw.x_raw[:, :, fi]  # [B, L]
        real_vals = field_vals[mask]  # flatten to real values only
        if real_vals.numel() == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(real_vals.min().item())
            vmax = float(real_vals.max().item())
        stats[field] = FieldStats(vmin=vmin, vmax=vmax)
    return NormStats(stats=stats)


def normalize_batch(batch_raw: ArchBatchRaw, norm: NormStats) -> ArchBatch:
    xs = []
    for fi, field in enumerate(ALL_FIELDS):
        fs = norm.stats[field]
        rng = max(fs.vmax - fs.vmin, EPS)
        vals = (batch_raw.x_raw[:, :, fi] - fs.vmin) / rng
        vals = torch.clamp(vals, 0.0, 1.0)
        xs.append(vals.unsqueeze(-1))
    x_norm = torch.cat(xs, dim=-1)
    return ArchBatch(x=x_norm, padding_mask=batch_raw.padding_mask, val_loss=batch_raw.val_loss)


class PairDataset(Dataset):
    def __init__(
        self,
        arch_batch: ArchBatch,
        num_pairs: int,
        tie_eps: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.x = arch_batch.x
        self.padding_mask = arch_batch.padding_mask
        self.y = arch_batch.val_loss
        self.num_pairs = num_pairs
        self.tie_eps = tie_eps
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int):
        a, b = self.rng.integers(0, len(self.x), size=2)
        xa, xb = self.x[a], self.x[b]
        ma, mb = self.padding_mask[a], self.padding_mask[b]
        la, lb = self.y[a].item(), self.y[b].item()
        label = 1.0 if la < lb else 0.0
        weight = 0.0 if abs(la - lb) < self.tie_eps else 1.0
        return xa, ma, xb, mb, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)


class ArchDataset(Dataset):
    def __init__(self, arch_batch: ArchBatch):
        self.x = arch_batch.x
        self.padding_mask = arch_batch.padding_mask
        self.y = arch_batch.val_loss

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.padding_mask[idx], self.y[idx]


def norm_stats_to_dict(norm: NormStats) -> Dict[str, Dict[str, float]]:
    return {k: {"vmin": v.vmin, "vmax": v.vmax} for k, v in norm.stats.items()}


def norm_stats_from_dict(d: Dict[str, Dict[str, float]]) -> NormStats:
    return NormStats(stats={k: FieldStats(vmin=v["vmin"], vmax=v["vmax"]) for k, v in d.items()})
