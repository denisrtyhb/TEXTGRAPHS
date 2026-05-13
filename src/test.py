from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .dataset import LoaderBundle, get_loaders
from .device import resolve_device
from .model import BertSimpleClassifier, build_classifier

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def predict_logits(model: BertSimpleClassifier, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(inputs=input_ids, attention_mask=attention_mask).squeeze(1)
            scores.extend(logits.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def run_test_predictions(
    model: BertSimpleClassifier,
    test_loader: DataLoader,
    test_frame: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    logits = predict_logits(model, test_loader, device)
    predictions = (logits >= 0.0).astype(np.int32)
    out = test_frame[["sample_id"]].copy()
    out["prediction"] = predictions
    return out


def _default_paths(project_root: Path) -> tuple[Path, Path]:
    data = project_root / "data" / "tsv"
    return data / "train.tsv", data / "test.tsv"


def build_test_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    train_default, test_default = _default_paths(project_root)
    parser = argparse.ArgumentParser(
        description="Run baseline on test TSV and write predictions.",
    )
    parser.add_argument("--train-tsv", type=Path, default=train_default)
    parser.add_argument("--test-tsv", type=Path, default=test_default)
    parser.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "predictions_test.tsv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--context-key", default="linearized_graph")
    parser.add_argument("--truncation", default="only_second")
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto (cuda/mps/cpu), cpu, cuda, cuda:N, mps, …",
    )
    return parser


def run_testing(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = resolve_device(args.device)
    logging.info("Using device %s (requested %r)", device, args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders: LoaderBundle = get_loaders(
        str(args.train_tsv),
        str(args.test_tsv),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        context_key=args.context_key,
        tokenizer_truncation=args.truncation,
        graph_only=args.graph_only,
        device=device,
    )

    model = build_classifier(args.model_name, dropout=args.dropout).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    submission = run_test_predictions(model, loaders.test, loaders.test_frame, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, sep="\t", index=False)
    logging.info("Wrote %d rows to %s", len(submission), args.output)


def main(argv: list[str] | None = None) -> None:
    args = build_test_parser().parse_args(argv)
    run_testing(args)


if __name__ == "__main__":
    main()
