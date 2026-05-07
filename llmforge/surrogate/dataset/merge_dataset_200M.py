"""Merge the 200M-style training CSVs into surrogate/dataset/dataset_200M.csv.

Sources:
  csv/dataset_infi_search_100M.csv       (20-layer; padded out to 40)
  csv/dataset_infi_search_200M.csv       (40-layer; same schema as target)
  csv/dataset_infi_search_200M_full.csv  (40-layer)
  csv/dataset_no_identity.csv            (40-layer)

Conventions:
  - Schema = #idx, 4 globals, 40 × 7 per-layer fields, 5 tail (val_loss,
    params, mem_bytes, flops, kv_cache_size).
  - Padded layers inherit values from the last active layer; their roles
    are advertised via `global_layer_mask` so downstream tensorisation
    still treats them as padding.
  - Dedup is per-row on every column except #idx — i.e. exact duplicates
    only. Different val_loss measurements of the same arch are kept (they
    are independent samples, not duplicates).
  - #idx is renumbered 1..N at the end so downstream lookup-by-idx works.

Run:
  python surrogate/dataset/merge_dataset_200M.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # nsga_search/
SRC_DIR = ROOT / "csv"
OUT_PATH = ROOT / "surrogate" / "dataset" / "dataset_200M.csv"

SOURCES = [
    SRC_DIR / "dataset_infi_search_100M.csv",
    SRC_DIR / "dataset_infi_search_200M.csv",
    SRC_DIR / "dataset_infi_search_200M_full.csv",
    SRC_DIR / "dataset_no_identity.csv",
    # 56 real-trained labels harvested from the 26_finetune_200m_paramsloss
    # NSGA run (2026-04-26). 8 lost to host_6 idx-7 bug; not included.
    ROOT / "ckpts/26_finetune_200m_paramsloss/real_trained_dataset.csv",
    # 33 real-trained labels harvested from the infi_search_100M_arch run
    # (2026-04-09). Run was degraded — most offspring inf'd, but post-elim
    # survivors have valid labels. Filtered: drop val_loss>4 and n_active<=2
    # (degenerate small archs).
    ROOT / "ckpts/infi_search_100M_arch/harvested_real_labels.filtered.csv",
]

LAYER_FIELDS = [
    "n_head", "n_kv_group", "mlp_size",
    "n_qk_head_dim", "n_v_head_dim", "n_cproj", "attention_variant",
]
TARGET_LAYERS = 40
GLOBAL_COLS = ["#idx", "global_n_embd", "global_block_size",
               "global_use_concat_heads", "global_layer_mask"]
TAIL_COLS = ["val_loss", "params", "mem_bytes", "flops", "kv_cache_size"]


def target_columns() -> list[str]:
    cols = list(GLOBAL_COLS)
    for L in range(TARGET_LAYERS):
        for f in LAYER_FIELDS:
            cols.append(f"layer{L}_{f}")
    cols += TAIL_COLS
    return cols


def expand_to_40_layers(df: pd.DataFrame, src_layers: int) -> pd.DataFrame:
    """Pad a 20-layer dataframe out to 40 layers.

    Each padded slot copies values from the last active layer (so cells are
    in-spec), and `global_layer_mask` is extended with `False` for every
    new slot so the surrogate's tokeniser ignores them.
    """
    if src_layers >= TARGET_LAYERS:
        return df

    out = df.copy()

    # Extend layer_mask with False for new slots
    def _ext_mask(s):
        m = ast.literal_eval(s) if isinstance(s, str) else list(s)
        return str(m + [False] * (TARGET_LAYERS - len(m)))
    out["global_layer_mask"] = out["global_layer_mask"].map(_ext_mask)

    # Fill new layer columns by copying the last existing layer block.
    # Build the new columns in one shot via pd.concat to avoid the
    # high-fragmentation warning from many .insert() calls.
    last = src_layers - 1
    new_cols = {
        f"layer{L}_{f}": out[f"layer{last}_{f}"]
        for L in range(src_layers, TARGET_LAYERS) for f in LAYER_FIELDS
    }
    out = pd.concat([out, pd.DataFrame(new_cols)], axis=1)
    return out


def detect_layer_cap(df: pd.DataFrame) -> int:
    layer_idx = set()
    for c in df.columns:
        if c.startswith("layer"):
            try:
                layer_idx.add(int(c.split("_", 1)[0][len("layer"):]))
            except ValueError:
                pass
    return max(layer_idx) + 1 if layer_idx else 0


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    src_layers = detect_layer_cap(df)
    if src_layers != TARGET_LAYERS:
        df = expand_to_40_layers(df, src_layers)
    cols = target_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns after expansion: {missing[:5]}...")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        # Surface but don't drop silently
        print(f"  [warn] {path.name}: dropping {len(extra)} extra columns "
              f"(first few: {extra[:3]})")
    return df[cols]


def main() -> int:
    print(f"merging into: {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    for src in SOURCES:
        if not src.exists():
            raise FileNotFoundError(src)
        df = load_and_normalize(src)
        before = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "#idx"])
        print(f"  loaded {src.name:42s}  rows {before:5d}  ({before - len(df)} intra-file dups dropped)")
        pieces.append(df)

    full = pd.concat(pieces, ignore_index=True)
    before = len(full)
    full = full.drop_duplicates(subset=[c for c in full.columns if c != "#idx"])
    print(f"\n  cross-file dedup: {before:5d} → {len(full):5d}  ({before - len(full)} cross-file dups dropped)")

    # Renumber #idx 1..N
    full = full.reset_index(drop=True)
    full["#idx"] = range(1, len(full) + 1)

    # Atomic write
    tmp = OUT_PATH.with_suffix(".csv.tmp")
    full.to_csv(tmp, index=False)
    tmp.replace(OUT_PATH)
    print(f"\nwrote {len(full)} rows × {len(full.columns)} cols → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
