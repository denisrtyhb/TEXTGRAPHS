from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .device import resolve_device


def linearize_graph(graph_dict: dict[str, Any], sep_token: str) -> str:
    """Turn graph JSON dict into a linearized string (baseline notebook, cell 48)."""
    nodes = sorted(graph_dict["nodes"], key=lambda d: d["id"])
    for n_id, node_dict in enumerate(nodes):
        if n_id != node_dict["id"]:
            raise ValueError("Node ids must be contiguous starting at 0")

    src_node_id2links: dict[int, list] = {}
    for link_dict in graph_dict["links"]:
        link_src = link_dict["source"]
        src_node_id2links.setdefault(link_src, []).append(link_dict)

    graph_s = ""
    for n_id, node_dict in enumerate(nodes):
        links = src_node_id2links.get(n_id, [])
        start_label = node_dict["label"]
        if node_dict["type"] == "ANSWER_CANDIDATE_ENTITY":
            start_label = f"{sep_token} {start_label} {sep_token}"
        for link_dict in links:
            target = nodes[link_dict["target"]]
            target_label = target["label"]
            if target["type"] == "ANSWER_CANDIDATE_ENTITY":
                target_label = f"{sep_token} {target_label} {sep_token}"
            link_s = f" {start_label}, {link_dict['label']}, {target_label} "
            graph_s += link_s
    return graph_s


def parse_graph_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return eval(value)  # noqa: S307 — matches notebook; graphs are trusted local TSV data
    raise TypeError(f"Unsupported graph column type: {type(value)}")


class QuestionAnswerDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        context_key: str = "answerEntity",
        tokenizer_truncation: str = "only_first",
        graph_only: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.graph_only = graph_only
        self.tokenizer_truncation = tokenizer_truncation

        self.questions = frame["question"].values
        self.contexts = frame[context_key].values
        self.labels = torch.tensor(frame["label"].values, dtype=torch.float32)

        if graph_only:
            self.tokenized_input = [
                tokenizer(
                    str(y),
                    max_length=max_length,
                    padding="max_length",
                    truncation=tokenizer_truncation,
                    return_tensors="pt",
                )
                for y in self.contexts
            ]
        else:
            self.tokenized_input = [
                tokenizer(
                    str(x),
                    str(y),
                    max_length=max_length,
                    padding="max_length",
                    truncation=tokenizer_truncation,
                    return_tensors="pt",
                )
                for x, y in zip(self.questions, self.contexts)
            ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenized_input[idx]
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": self.labels[idx],
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def train_val_split_by_question(
    train_dev_df: pd.DataFrame, train_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = list(train_dev_df["question"].unique())
    rng = random.Random(seed)
    rng.shuffle(questions)
    num_train = int(len(questions) * train_ratio)
    train_q = set(questions[:num_train])
    dev_q = set(questions[num_train:])
    train_df = train_dev_df[train_dev_df["question"].isin(train_q)].copy()
    dev_df = train_dev_df[train_dev_df["question"].isin(dev_q)].copy()
    return train_df, dev_df


@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    test_frame: pd.DataFrame


def get_loaders(
    train_tsv_path: str,
    test_tsv_path: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    max_length: int = 128,
    train_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = 0,
    context_key: str = "linearized_graph",
    tokenizer_truncation: str = "only_second",
    graph_only: bool = False,
    device: str | torch.device | None = None,
) -> LoaderBundle:
    """
    Load TSVs, question-level train/dev split, linearize graphs, and build DataLoaders.

    Mirrors ``baselines/bert_baselines.ipynb`` (paths: ``data/tsv/train.tsv``, ``test.tsv``).

    ``device``: pass the same training/inference device to enable ``pin_memory`` on CUDA.
    """
    if device is None:
        pin_memory = False
    elif isinstance(device, torch.device):
        pin_memory = device.type == "cuda"
    else:
        pin_memory = resolve_device(device).type == "cuda"

    train_dev_df = pd.read_csv(train_tsv_path, sep="\t")
    test_df = pd.read_csv(test_tsv_path, sep="\t").copy()

    train_df, dev_df = train_val_split_by_question(train_dev_df, train_ratio, seed)

    train_df["label"] = train_df["correct"].astype(np.float32)
    dev_df["label"] = dev_df["correct"].astype(np.float32)
    test_df["label"] = np.zeros(test_df.shape[0], dtype=np.float32)

    train_df["graph"] = train_df["graph"].apply(parse_graph_cell)
    dev_df["graph"] = dev_df["graph"].apply(parse_graph_cell)
    test_df["graph"] = test_df["graph"].apply(parse_graph_cell)

    sep = tokenizer.sep_token or "[SEP]"
    train_df["linearized_graph"] = train_df["graph"].apply(lambda g: linearize_graph(g, sep))
    dev_df["linearized_graph"] = dev_df["graph"].apply(lambda g: linearize_graph(g, sep))
    test_df["linearized_graph"] = test_df["graph"].apply(lambda g: linearize_graph(g, sep))

    train_dataset = QuestionAnswerDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=context_key,
        tokenizer_truncation=tokenizer_truncation,
        graph_only=graph_only,
    )
    dev_dataset = QuestionAnswerDataset(
        dev_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=context_key,
        tokenizer_truncation=tokenizer_truncation,
        graph_only=graph_only,
    )
    test_dataset = QuestionAnswerDataset(
        test_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=context_key,
        tokenizer_truncation=tokenizer_truncation,
        graph_only=graph_only,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
    )
    return LoaderBundle(train=train_loader, val=val_loader, test=test_loader, test_frame=test_df)
