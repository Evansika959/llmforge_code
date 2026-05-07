#!/usr/bin/env python3
"""Re-evaluate WinoGrande (lm-eval-harness convention) and SciQ (with support passage).

WinoGrande fix: per-choice context (`before + option_i`) and shared target (`after`).
                Removes option-frequency contamination from the score.
SciQ fix: include the `support` passage in the context (open-book scoring).
"""
import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from typing import List, Tuple

import torch

# Locate the llmforge_train package root (containing model.py). Override via $LLMFORGE_REPO_ROOT,
# else walk upward from this file until we find model.py.
REPO_ROOT = os.environ.get("LLMFORGE_REPO_ROOT")
if not REPO_ROOT:
    _here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(_here, "model.py")):
            REPO_ROOT = _here
            break
        _parent = os.path.dirname(_here)
        if _parent == _here:
            break
        _here = _parent
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks.evaluate_custom_models import (  # noqa: E402
    _load_checkpoint, _load_tokenizer, _get_block_size,
)
from benchmarks.evaluate_huggingface_models import (  # noqa: E402
    _get_benchmark_dataset, _load_dataset_with_retry,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--config_path", default=None)
    p.add_argument("--benchmark", choices=["winogrande", "sciq"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--output_json", default=None)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--datasets_cache_dir", default=None)
    p.add_argument("--modules_cache_dir", default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--weights_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print_examples", action="store_true", default=False)
    p.add_argument("--init_from", default="resume")
    p.add_argument("--meta_path", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--tokenizer_model", default=None)
    p.add_argument("--bos_token_id", type=int, default=None)
    p.add_argument("--eos_token_id", type=int, default=None)
    return p.parse_args()


def _per_choice_loglikelihood(
    model, encode, ctx_text: str, target_text: str,
    block_size: int, length_norm: bool, device, ctx_autocast,
) -> float:
    ctx_tokens = encode(ctx_text)
    tgt_tokens = encode(target_text)
    if len(tgt_tokens) == 0:
        return -math.inf
    max_ctx_len = max(0, block_size - len(tgt_tokens))
    if len(ctx_tokens) > max_ctx_len:
        ctx_tokens = ctx_tokens[-max_ctx_len:]
    full = ctx_tokens + tgt_tokens
    if len(full) < 2:
        return -math.inf
    input_ids = torch.tensor(full[:-1], device=device).unsqueeze(0)
    target_ids = torch.tensor(full[1:], device=device).unsqueeze(0)
    target_start = max(len(ctx_tokens) - 1, 0)
    with ctx_autocast:
        logits, _ = model(input_ids, target_ids)
    logprobs = torch.log_softmax(logits, dim=-1)
    target_slice = target_ids[:, target_start:]
    lp = logprobs[:, target_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
    return (lp.mean() if length_norm else lp.sum()).item()


def _extract_pairs(benchmark: str, example: dict) -> Tuple[List[Tuple[str, str]], int]:
    if benchmark == "winogrande":
        sentence = example["sentence"].strip()
        option1 = example["option1"]
        option2 = example["option2"]
        if "_" in sentence:
            before, after = sentence.split("_", 1)
        else:
            before, after = sentence, ""
        # lm-eval-harness convention: ctx = before+option_i, target = after
        pairs = [(before + option1, after), (before + option2, after)]
        ans = example.get("answer")
        label = int(ans) - 1 if ans else None
        return pairs, label

    if benchmark == "sciq":
        question = example.get("question", "").strip()
        support = (example.get("support", "") or "").strip()
        if support:
            ctx = f"{support}\nQuestion: {question}\nAnswer:"
        else:
            ctx = f"Question: {question}\nAnswer:"
        endings = [
            " " + (example.get("correct_answer", "") or "").strip(),
            " " + (example.get("distractor1", "") or "").strip(),
            " " + (example.get("distractor2", "") or "").strip(),
            " " + (example.get("distractor3", "") or "").strip(),
        ]
        return [(ctx, e) for e in endings], 0

    raise ValueError(benchmark)


def main():
    args = parse_args()
    if args.datasets_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.datasets_cache_dir
    if args.modules_cache_dir:
        os.environ["HF_MODULES_CACHE"] = args.modules_cache_dir
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, ckpt_cfg = _load_checkpoint(args)
    encode, _decode = _load_tokenizer(args, ckpt_cfg)
    model.eval()
    model.to(args.device)

    block_size = int(getattr(model.config, "block_size",
                             _get_block_size(model, args.block_size)))
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx_autocast = (nullcontext() if device_type == "cpu"
                    else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    ds_name, ds_cfg = _get_benchmark_dataset(args.benchmark)
    ds = _load_dataset_with_retry(ds_name, ds_cfg, args.split,
                                  args.datasets_cache_dir, args.modules_cache_dir)
    if args.max_examples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_examples))

    correct = 0; total = 0; skipped = 0
    with torch.inference_mode():
        for ex in ds:
            pairs, label = _extract_pairs(args.benchmark, ex)
            if label is None:
                skipped += 1
                continue
            scores = [
                _per_choice_loglikelihood(model, encode, ctx, tgt,
                                          block_size, args.length_norm,
                                          args.device, ctx_autocast)
                for ctx, tgt in pairs
            ]
            pred = max(range(len(scores)), key=lambda k: scores[k])
            if pred == label:
                correct += 1
            total += 1

    acc = correct / total if total else float("nan")
    out = [{
        "split": args.split, "total": total, "correct": correct, "accuracy": acc,
        "skipped": skipped, "block_size": block_size, "length_norm": bool(args.length_norm),
        "benchmark": args.benchmark,
        "scoring_convention": "lm-eval-harness compatible (per-choice ctx for WG; support+question for SciQ)",
    }]
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
    print(f"{args.benchmark} acc = {acc:.4f}  ({correct}/{total}, skipped={skipped})")


if __name__ == "__main__":
    main()
