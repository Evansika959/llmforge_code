"""Seeded population init around a single reference arch.

Library form of seed_from_known_arch.py: the dispatcher calls
build_seeded_population(...) when --seed_arch is provided.

Seed 0 is the unperturbed reference (repaired against the target space).
Seeds 1..N-1 are mild per-layer jitter copies (also repaired). Hard error
if any field of the seed lies outside the target search-space spec.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List

import yaml

from search_space import HeteroSearchSpace, Individual


# ── Reference-arch loader (matches seed_from_known_arch.py YAML schema) ────

def load_reference_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        ref = yaml.safe_load(f)
    if not isinstance(ref, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    if "n_embd" not in ref:
        raise ValueError(f"{path}: missing 'n_embd'")
    layers = ref.get("layers")
    if not layers:
        tmpl = ref.get("layer_template")
        n_layers = ref.get("n_layers")
        if tmpl is None or n_layers is None:
            raise ValueError(
                f"{path}: provide 'layers' (list) or 'layer_template' + 'n_layers'")
        layers = [dict(tmpl) for _ in range(int(n_layers))]
    else:
        layers = [dict(li) for li in layers]
    return {
        "name": ref.get("name", os.path.basename(path)),
        "n_embd": int(ref["n_embd"]),
        "block_size": int(ref.get("block_size", 512)),
        "use_concat_heads": bool(ref.get("use_concat_heads", True)),
        "layers": layers,
    }


def load_reference(path: str) -> Dict[str, Any]:
    """Auto-detect YAML vs JSON by extension; both forms map to the same
    normalized dict the rest of this module consumes."""
    if path.endswith((".yaml", ".yml")):
        return load_reference_yaml(path)
    if path.endswith(".json"):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        # Accept either the same schema or a Population-style individual.
        if "layers" in data and "n_embd" in data:
            return {
                "name": data.get("name", os.path.basename(path)),
                "n_embd": int(data["n_embd"]),
                "block_size": int(data.get("block_size", 512)),
                "use_concat_heads": bool(data.get("use_concat_heads", True)),
                "layers": [dict(li) for li in data["layers"]],
            }
        if "globals" in data and "layers" in data:
            g = data["globals"]
            mask = g.get("layer_mask", [True] * len(data["layers"]))
            active = [li for li, m in zip(data["layers"], mask) if m]
            return {
                "name": data.get("name", os.path.basename(path)),
                "n_embd": int(g["n_embd"]),
                "block_size": int(g.get("block_size", 512)),
                "use_concat_heads": bool(g.get("use_concat_heads", True)),
                "layers": [dict(li) for li in active],
            }
        raise ValueError(f"{path}: unrecognized JSON schema for seed arch")
    raise ValueError(f"{path}: --seed_arch must be .yaml/.yml/.json")


def reference_to_individual(ref: Dict[str, Any], L_max: int) -> Dict[str, Any]:
    """Pack the reference into the IHA Individual layout (active prefix + padding)."""
    base_layers = ref["layers"]
    n_active = min(len(base_layers), L_max)
    if len(base_layers) > L_max:
        raise ValueError(
            f"[seed_arch] reference has {len(base_layers)} layers but L_max={L_max}; "
            f"raise --max_layers or trim the seed.")
    layers = []
    for i in range(L_max):
        src = base_layers[i] if i < n_active else base_layers[-1]
        layers.append(dict(src))
    mask = [i < n_active for i in range(L_max)]
    globals_ = {
        "n_embd": ref["n_embd"],
        "block_size": ref["block_size"],
        "use_concat_heads": ref["use_concat_heads"],
        "layer_mask": mask,
    }
    return {"globals": globals_, "layers": layers}


# ── Jitter ─────────────────────────────────────────────────────────────────

def jitter_individual(base: Dict[str, Any], layer_spec: Dict[str, Any], *,
                      p_mlp: float = 0.15, p_head: float = 0.10,
                      p_kv: float = 0.10, p_identity: float = 0.03,
                      p_qk_dim: float = 0.05, p_v_dim: float = 0.05) -> Dict[str, Any]:
    """Per-active-layer perturbations. Step sizes match the layer_spec so
    values land on-grid; downstream repair() handles divisibility / clamps."""
    out = copy.deepcopy(base)
    mlp_step = int(layer_spec["mlp_size"].get("step", 256))
    qk_step = int(layer_spec["n_qk_head_dim"].get("step", 32))
    v_step = int(layer_spec.get("n_v_head_dim",
                                 layer_spec["n_qk_head_dim"]).get("step", qk_step))
    mask = out["globals"].get("layer_mask", [True] * len(out["layers"]))
    for i, li in enumerate(out["layers"]):
        if not mask[i]:
            continue
        if random.random() < p_mlp:
            li["mlp_size"] += random.choice([-1, 1]) * mlp_step
        if random.random() < p_head:
            li["n_head"] = max(1, li["n_head"] + random.choice([-1, 1]))
        if random.random() < p_kv:
            li["n_kv_group"] = max(1, li["n_kv_group"] + random.choice([-1, 1]))
        if random.random() < p_identity:
            li["attention_variant"] = "identity"
        if random.random() < p_qk_dim:
            li["n_qk_head_dim"] += random.choice([-1, 1]) * qk_step
        if random.random() < p_v_dim:
            li["n_v_head_dim"] += random.choice([-1, 1]) * v_step
    return out


# ── Validation ─────────────────────────────────────────────────────────────

def _check_field(label: str, value: Any, spec: Dict[str, Any]) -> List[str]:
    """Compare a single seed field against its search-space spec entry.

    Schema (matches search_space_def/*.yaml):
      type: int | float | cat
      int / float: low, high, step (continuous range; ‑step grid)
      cat:         choices (explicit enum)
    """
    errs: List[str] = []
    t = spec.get("type")
    if t == "cat":
        choices = spec.get("choices", [])
        if value not in choices:
            errs.append(f"{label}={value!r} not in choices {choices}")
    elif t in ("int", "float"):
        lo = spec.get("low")
        hi = spec.get("high")
        step = spec.get("step")
        if lo is not None and value < lo:
            errs.append(f"{label}={value} < low {lo}")
        if hi is not None and value > hi:
            errs.append(f"{label}={value} > high {hi}")
        if (t == "int" and step and lo is not None
                and step > 1 and (int(value) - int(lo)) % int(step) != 0):
            errs.append(f"{label}={value} not on int grid (low={lo}, step={step})")
    return errs


def _check_global(ref: Dict[str, Any], global_spec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for key, spec in global_spec.items():
        if key not in ref:
            continue
        errs += _check_field(f"globals.{key}", ref[key], spec)
    return errs


def _check_layers(ref_layers: List[Dict[str, Any]],
                  layer_spec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for i, li in enumerate(ref_layers):
        for key, spec in layer_spec.items():
            if key not in li:
                continue
            errs += _check_field(f"layer[{i}].{key}", li[key], spec)
    return errs


def validate_seed_in_space(ref: Dict[str, Any], global_spec: Dict[str, Any],
                            layer_spec: Dict[str, Any], L_min: int, L_max: int) -> None:
    """Hard-error if any seed field is outside the search-space spec, or
    if the active layer count is outside [L_min, L_max]."""
    errs: List[str] = []
    n = len(ref["layers"])
    if n < L_min or n > L_max:
        errs.append(f"layer count {n} outside [L_min={L_min}, L_max={L_max}]")
    errs += _check_global(ref, global_spec)
    errs += _check_layers(ref["layers"], layer_spec)
    if errs:
        raise ValueError(
            "[seed_arch] reference arch is outside the search space. "
            "Either expand the search space or fix the seed:\n  - "
            + "\n  - ".join(errs)
        )


# ── Public entry ───────────────────────────────────────────────────────────

def build_seeded_population(seed_path: str, *, n: int,
                             search_space: HeteroSearchSpace,
                             global_spec: Dict[str, Any],
                             layer_spec: Dict[str, Any],
                             L_max: int, L_min: int,
                             p_mlp: float = 0.15, p_head: float = 0.10,
                             p_kv: float = 0.10, p_identity: float = 0.03,
                             p_qk_dim: float = 0.05,
                             p_v_dim: float = 0.05) -> List[Individual]:
    """Return a list of `n` individuals: seed 0 unperturbed, 1..n-1 jittered.
    Each is repaired against the target search space."""
    ref = load_reference(seed_path)
    validate_seed_in_space(ref, global_spec, layer_spec, L_min, L_max)
    base = reference_to_individual(ref, L_max=L_max)

    out: List[Individual] = []
    out.append(Individual(**search_space.repair(copy.deepcopy(base))))
    for _ in range(max(0, n - 1)):
        jittered = jitter_individual(
            base, layer_spec=layer_spec,
            p_mlp=p_mlp, p_head=p_head, p_kv=p_kv,
            p_identity=p_identity, p_qk_dim=p_qk_dim, p_v_dim=p_v_dim)
        out.append(Individual(**search_space.repair(jittered)))
    return out
