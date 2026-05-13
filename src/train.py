from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset import LoaderBundle, get_loaders
from .device import resolve_device
from .model import BertSimpleClassifier, build_classifier

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_epoch(
    model: BertSimpleClassifier,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for batch in tqdm(data_loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs=input_ids, attention_mask=attention_mask).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    return epoch_loss / max(num_batches, 1)


def val_epoch(
    model: BertSimpleClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    true_labels: list[float] = []
    pred_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            true_labels.extend(labels.cpu().numpy().tolist())
            labels_dev = labels.to(device)
            logits = model(inputs=input_ids, attention_mask=attention_mask).squeeze(1)
            batch_pred = (logits.detach().cpu().numpy() >= 0.0).astype(int).tolist()
            pred_labels.extend(batch_pred)
            loss = criterion(logits, labels_dev)
            epoch_loss += loss.item()

    num_batches = max(len(data_loader), 1)
    val_f1 = f1_score(true_labels, pred_labels)
    return epoch_loss / num_batches, val_f1


def train(
    model: BertSimpleClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    best_f1 = 0.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = val_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - start
        logging.info(
            "Epoch %d/%d | %.1fs | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch + 1,
            n_epochs,
            elapsed,
            train_loss,
            val_loss,
            val_f1,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                "Saved new best checkpoint (f1=%.4f) -> %s",
                val_f1,
                checkpoint_path,
            )


def _default_paths(project_root: Path) -> tuple[Path, Path]:
    data = project_root / "data" / "tsv"
    return data / "train.tsv", data / "test.tsv"


def build_train_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    train_default, test_default = _default_paths(project_root)
    parser = argparse.ArgumentParser(
        description="Train baseline classifier (bert_baselines notebook).",
    )
    parser.add_argument("--train-tsv", type=Path, default=train_default)
    parser.add_argument("--test-tsv", type=Path, default=test_default)
    parser.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "weights" / "best-val-baseline.pt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--context-key",
        default="linearized_graph",
        help="DataFrame column paired with question (e.g. answerEntity or linearized_graph).",
    )
    parser.add_argument(
        "--truncation",
        default="only_second",
        help="Tokenizer truncation strategy (use only_first for text-only baseline).",
    )
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto (cuda/mps/cpu), cpu, cuda, cuda:N, mps, …",
    )
    return parser


def run_training(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = resolve_device(args.device)
    logging.info("Using device %s (requested %r)", device, args.device)

    torch.manual_seed(args.seed)
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    train(
        model,
        loaders.train,
        loaders.val,
        optimizer,
        criterion,
        n_epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        device=device,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
