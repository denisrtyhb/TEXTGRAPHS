from __future__ import annotations

"""Data loading: ``get_loaders`` / ``get_dataset``. Presets ``linearized_graph`` (default) vs
``nlp_enjoyers_dataset`` (Eq: prefix + ``; ``-separated edges). ``ALLOWED_DATASET_IDS``: valid preset ids."""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .device import resolve_device


def linearize_graph(graph_dict: dict[str, Any], sep_token: str) -> str:
    """Turn graph JSON dict into a linearized string (baseline notebook), edges concatenated loosely."""
    return _linearize_graph_from_edges(graph_dict, sep_token, join_edges=lambda parts: "".join(f" {p} " for p in parts))


def linearize_graph_semicolon_edges(graph_dict: dict[str, Any], sep_token: str) -> str:
    """Linearized graph with each edge triple separated by ``"; "`` (Kurdiukov et al., 2024)."""
    return _linearize_graph_from_edges(graph_dict, sep_token, join_edges=lambda parts: "; ".join(parts))


def _linearize_graph_from_edges(
    graph_dict: dict[str, Any],
    sep_token: str,
    join_edges,
) -> str:
    """Walk the graph once per edge triple; ``join_edges`` controls how triples are combined."""
    nodes = sorted(graph_dict["nodes"], key=lambda d: d["id"])
    for n_id, node_dict in enumerate(nodes):
        if n_id != node_dict["id"]:
            raise ValueError("Node ids must be contiguous starting at 0")

    src_node_id2links: dict[int, list] = {}
    for link_dict in graph_dict["links"]:
        link_src = link_dict["source"]
        src_node_id2links.setdefault(link_src, []).append(link_dict)

    edges: list[str] = []
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
            triple = f"{start_label.strip()}, {link_dict['label']}, {target_label.strip()}".strip()
            edges.append(triple)
    return join_edges(edges)


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
        question_prefix: str | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.graph_only = graph_only
        self.tokenizer_truncation = tokenizer_truncation

        if question_prefix:
            qs = np.array([f"{question_prefix}{x}" for x in frame["question"].values])
        else:
            qs = frame["question"].values

        self.questions = qs
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


def _subset_df_to_first_n_questions(df: pd.DataFrame, max_questions: int) -> pd.DataFrame:
    if max_questions <= 0 or len(df) == 0:
        return df
    questions = list(pd.unique(df["question"]))[:max_questions]
    return df[df["question"].isin(questions)].copy()


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


class _DatasetPreset(str, Enum):
    LINEARIZED_GRAPH = "linearized_graph"
    NLP_ENJOYERS_DATASET = "nlp_enjoyers_dataset"


_PRESETS: dict[_DatasetPreset, dict[str, Any]] = {
    _DatasetPreset.LINEARIZED_GRAPH: {
        "context_key": "linearized_graph",
        "tokenizer_truncation": "only_second",
        "graph_only": False,
        "train_positives_only": False,
        "linearize_style": "default",
        "question_prefix": None,
    },
    _DatasetPreset.NLP_ENJOYERS_DATASET: {
        "context_key": "linearized_graph",
        "tokenizer_truncation": "only_second",
        "graph_only": False,
        "train_positives_only": False,
        "linearize_style": "semicolon",
        "question_prefix": "Eq: ",
    },
}

ALLOWED_DATASET_IDS: tuple[str, ...] = tuple(p.value for p in _DatasetPreset)


def _parse_dataset_id(value: str) -> _DatasetPreset:
    try:
        return _DatasetPreset(value)
    except ValueError as e:
        allowed = ", ".join(sorted(d.value for d in _DatasetPreset))
        raise ValueError(f"unknown dataset {value!r}; choose one of: {allowed}") from e


def _preset_for(id_: _DatasetPreset) -> dict[str, Any]:
    return dict(_PRESETS[id_])


def _prepare_split_dataframes(
    train_tsv_path: str,
    test_tsv_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset: str,
    train_ratio: float,
    seed: int,
    mock: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preset = dict(_preset_for(_parse_dataset_id(dataset)))
    train_positives_only = bool(preset.pop("train_positives_only"))
    linearize_style = str(preset.pop("linearize_style", "default"))

    train_dev_df = pd.read_csv(train_tsv_path, sep="\t")
    test_df = pd.read_csv(test_tsv_path, sep="\t").copy()

    train_df, dev_df = train_val_split_by_question(train_dev_df, train_ratio, seed)

    if mock:
        train_df = _subset_df_to_first_n_questions(train_df, 12)
        dev_df = _subset_df_to_first_n_questions(dev_df, 3)
        test_df = _subset_df_to_first_n_questions(test_df, 10)
        print(
            f"mock: subset to {train_df['question'].nunique()} train questions, "
            f"{dev_df['question'].nunique()} dev questions, "
            f"{test_df['question'].nunique()} test questions "
            f"({len(test_df)} test rows)."
        )

    train_df["label"] = train_df["correct"].astype(np.float32)
    dev_df["label"] = dev_df["correct"].astype(np.float32)
    test_df["label"] = np.zeros(test_df.shape[0], dtype=np.float32)

    if train_positives_only:
        n_train_before = len(train_df)
        train_df = train_df.loc[train_df["label"] > 0.5].copy()
        print(
            f"train-positives-only: kept {len(train_df)} / {n_train_before} train rows "
            "(correct=True only; val/test unchanged)."
        )
        if len(train_df) == 0:
            raise ValueError("train-positives-only removed all training rows")

    train_df["graph"] = train_df["graph"].apply(parse_graph_cell)
    dev_df["graph"] = dev_df["graph"].apply(parse_graph_cell)
    test_df["graph"] = test_df["graph"].apply(parse_graph_cell)

    sep = tokenizer.sep_token or "[SEP]"
    if linearize_style == "semicolon":
        lin = lambda g: linearize_graph_semicolon_edges(g, sep)
    else:
        lin = lambda g: linearize_graph(g, sep)

    train_df["linearized_graph"] = train_df["graph"].apply(lin)
    dev_df["linearized_graph"] = dev_df["graph"].apply(lin)
    test_df["linearized_graph"] = test_df["graph"].apply(lin)

    return train_df, dev_df, test_df, preset


def get_dataset(
    train_tsv_path: str,
    test_tsv_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset: str = ALLOWED_DATASET_IDS[0],
    max_length: int = 128,
    train_ratio: float = 0.9,
    seed: int = 42,
    mock: bool = False,
) -> QuestionAnswerDataset:
    """
    Same train-split preprocessing/tokenization pipeline as ``get_loaders``, but returns only the
    **training** :class:`QuestionAnswerDataset` (no DataLoaders).
    """
    train_df, _dev_df, _test_df, preset = _prepare_split_dataframes(
        train_tsv_path,
        test_tsv_path,
        tokenizer,
        dataset=dataset,
        train_ratio=train_ratio,
        seed=seed,
        mock=mock,
    )
    return QuestionAnswerDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=preset["context_key"],
        tokenizer_truncation=preset["tokenizer_truncation"],
        graph_only=preset["graph_only"],
        question_prefix=preset.get("question_prefix"),
    )


def get_loaders(
    train_tsv_path: str,
    test_tsv_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    dataset: str = ALLOWED_DATASET_IDS[0],
    batch_size: int = 32,
    max_length: int = 128,
    train_ratio: float = 0.9,
    seed: int = 42,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    mock: bool = False,
) -> LoaderBundle:
    """
    Load TSVs, apply the named ``dataset`` preset (column/truncation options), linearize graphs, and
    build train / val / test :class:`~torch.utils.data.DataLoader` instances.
    """
    train_df, dev_df, test_df, preset = _prepare_split_dataframes(
        train_tsv_path,
        test_tsv_path,
        tokenizer,
        dataset=dataset,
        train_ratio=train_ratio,
        seed=seed,
        mock=mock,
    )

    if device is None:
        pin_memory = False
    elif isinstance(device, torch.device):
        pin_memory = device.type == "cuda"
    else:
        pin_memory = resolve_device(device).type == "cuda"

    train_dataset = QuestionAnswerDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=preset["context_key"],
        tokenizer_truncation=preset["tokenizer_truncation"],
        graph_only=preset["graph_only"],
        question_prefix=preset.get("question_prefix"),
    )
    dev_dataset = QuestionAnswerDataset(
        dev_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=preset["context_key"],
        tokenizer_truncation=preset["tokenizer_truncation"],
        graph_only=preset["graph_only"],
        question_prefix=preset.get("question_prefix"),
    )
    test_dataset = QuestionAnswerDataset(
        test_df,
        tokenizer=tokenizer,
        max_length=max_length,
        context_key=preset["context_key"],
        tokenizer_truncation=preset["tokenizer_truncation"],
        graph_only=preset["graph_only"],
        question_prefix=preset.get("question_prefix"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=(not mock),
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
