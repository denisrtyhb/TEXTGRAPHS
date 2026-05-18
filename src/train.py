from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from .dataset import ALLOWED_DATASET_IDS, LoaderBundle, get_loaders
from .device import resolve_device
from .model import BertSimpleClassifier, build_classifier

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


MOCK_TRAIN_MAX_BATCHES = 10
MOCK_VAL_MAX_BATCHES = 1


def train_epoch(
    model: BertSimpleClassifier,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> float:
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    progress = tqdm(data_loader, desc="train", leave=False, dynamic_ncols=True)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs=input_ids, attention_mask=attention_mask).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        num_batches += 1
        epoch_loss += batch_loss
        avg_loss = epoch_loss / num_batches
        progress.set_postfix(avg_loss=f"{avg_loss:.4f}", last=f"{batch_loss:.4f}")
        if max_batches is not None and num_batches >= max_batches:
            break
    mean_loss = epoch_loss / max(num_batches, 1)
    print(f"train epoch mean loss: {mean_loss:.4f} ({num_batches} batches)")
    return mean_loss


def val_epoch(
    model: BertSimpleClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    epoch_loss = 0.0
    true_labels: list[float] = []
    pred_labels: list[int] = []

    n_batches = 0
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
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    if not true_labels:
        return 0.0, 0.0
    val_f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return epoch_loss / max(n_batches, 1), val_f1


def train(
    model: BertSimpleClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    checkpoint_path: Path,
    device: torch.device,
    *,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> None:
    best_f1 = 0.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, max_batches=max_train_batches
        )
        val_loss, val_f1 = val_epoch(
            model, val_loader, criterion, device, max_batches=max_val_batches
        )
        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{n_epochs} | {elapsed:.1f}s | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint (f1={val_f1:.4f}) -> {checkpoint_path}")


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
        "--dataset",
        choices=list(ALLOWED_DATASET_IDS),
        default=ALLOWED_DATASET_IDS[0],
        help="Built-in corpus / tokenization preset (see src.dataset.get_loaders).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto (cuda/mps/cpu), cpu, cuda, cuda:N, mps, …",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Smoke mode: tiny data subset, 10 train batches, 1 val batch, 1 epoch.",
    )
    return parser


def run_training(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Using device {device} (requested {args.device!r})")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mock = getattr(args, "mock", False)
    if mock:
        print(
            f"mock: training with at most {MOCK_TRAIN_MAX_BATCHES} train batches and "
            f"{MOCK_VAL_MAX_BATCHES} val batch(es), 1 epoch"
        )

    loaders: LoaderBundle = get_loaders(
        str(args.train_tsv),
        str(args.test_tsv),
        tokenizer,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        mock=mock,
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
        n_epochs=1 if mock else args.epochs,
        checkpoint_path=args.checkpoint,
        device=device,
        max_train_batches=MOCK_TRAIN_MAX_BATCHES if mock else None,
        max_val_batches=MOCK_VAL_MAX_BATCHES if mock else None,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
