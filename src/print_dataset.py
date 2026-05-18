"""Dump the first rows of the training split exactly as training would build it."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoTokenizer

from .dataset import get_dataset

_NUM_PREVIEW = 5
_DECODE_MAX_CHARS = 500


def _format_tensor_line(name: str, t: torch.Tensor, tokenizer: Any) -> str:
    bits = [f"shape={tuple(t.shape)}", f"dtype={t.dtype}"]
    if name == "input_ids":
        text = tokenizer.decode(t.tolist(), skip_special_tokens=False)
        if len(text) > _DECODE_MAX_CHARS:
            text = text[: _DECODE_MAX_CHARS] + "…"
        bits.append(f"decoded={text!r}")
    else:
        flat = t.detach().cpu().flatten()
        n = min(32, flat.numel())
        bits.append(f"first_{n}={flat[:n].tolist()!r}")
    return "; ".join(bits)


def _print_heading_and_item(heading: str, item: dict[str, Any], tokenizer: Any) -> None:
    print(heading)
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {_format_tensor_line(key, value, tokenizer)}")
        else:
            print(f"  {key}: {value!r}")


def run_print_dataset(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mock = getattr(args, "mock", False)
    dataset = get_dataset(
        str(args.train_tsv),
        str(args.test_tsv),
        tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
        mock=mock,
    )
    n = min(_NUM_PREVIEW, len(dataset))
    print(f"Training dataset size: {len(dataset)} (showing first {n} elements)\n")
    for i in range(n):
        sample = dataset[i]
        if isinstance(sample, tuple) and len(sample) == 2:
            pos, neg = sample
            print(f"Pair index {i} (positive + negative)\n")
            _print_heading_and_item(f"Pair {i + 1} — positive", pos, tokenizer)
            print()
            _print_heading_and_item(f"Pair {i + 1} — negative", neg, tokenizer)
        else:
            assert isinstance(sample, dict)
            _print_heading_and_item(f"Element {i + 1}", sample, tokenizer)
        if i + 1 < n:
            print()


