"""Unified CLI: ``python -m src.main <mode> [--config ...] [overrides...]``.

Modes: ``train``, ``print_dataset``, ``test`` / ``eval`` (inference), ``evaluate`` (metrics).

The ``checkpoint`` field is used in both train and test/eval: in **train** it is the path
where the best validation weights are **saved**; in **test** / **eval** it is the path
to **load** weights from for inference.

Resolution order for each setting: command line > JSON config > built-in default.
Unknown keys in the JSON file are ignored (with a short notice).

``mock``: for **train** and **test** / **eval**, fast smoke runs (subset data, short loops).

Run as ``python -m src.main`` from the repository root so package imports resolve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from .dataset import ALLOWED_DATASET_IDS
from .evaluate import run_evaluation
from .print_dataset import run_print_dataset
from .test import run_testing
from .train import run_training


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_train_test_tsv() -> tuple[Path, Path]:
    data = _project_root() / "data" / "tsv"
    return data / "train.tsv", data / "test.tsv"


def train_defaults() -> dict[str, Any]:
    root = _project_root()
    train_tsv, test_tsv = _default_train_test_tsv()
    return {
        "train_tsv": train_tsv,
        "test_tsv": test_tsv,
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "checkpoint": root / "weights" / "best-val-baseline.pt",
        "epochs": 5,
        "batch_size": 32,
        "lr": 3e-5,
        "dropout": 0.2,
        "max_length": 128,
        "train_ratio": 0.9,
        "seed": 42,
        "num_workers": 0,
        "dataset": ALLOWED_DATASET_IDS[0],
        "device": "auto",
        "mock": False,
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
        "freeze_encoder_layers": 0,
    }


def print_dataset_defaults() -> dict[str, Any]:
    """Same parameters as training (data loading + tokenization)."""
    return train_defaults()


def inference_defaults() -> dict[str, Any]:
    """Defaults for ``test`` and ``eval`` (load checkpoint, write predictions)."""
    train_tsv, test_tsv = _default_train_test_tsv()
    return {
        "train_tsv": train_tsv,
        "test_tsv": test_tsv,
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "checkpoint": None,
        "output": None,
        "batch_size": 32,
        "max_length": 128,
        "dropout": 0.2,
        "train_ratio": 0.9,
        "seed": 42,
        "num_workers": 0,
        "dataset": ALLOWED_DATASET_IDS[0],
        "device": "auto",
        "one_yes_per_question": False,
        "mock": False,
    }


def evaluate_defaults() -> dict[str, Any]:
    """Defaults for ``evaluate`` (metrics on prediction TSV vs gold)."""
    return {
        "predictions_path": None,
        "gold_path": None,
        "strict_sample_id": False,
    }


def _normalize_config_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k).replace("-", "_"): v for k, v in raw.items()}


def _filter_config(mode: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    allowed = _allowed_keys(mode)
    used: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in cfg.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key in allowed:
            used[key] = value
        else:
            ignored.append(key)
    return used, ignored


def _allowed_keys(mode: str) -> frozenset[str]:
    if mode == "train":
        return frozenset(train_defaults().keys())
    if mode == "print_dataset":
        return frozenset(print_dataset_defaults().keys())
    if mode in ("test", "eval"):
        return frozenset(inference_defaults().keys())
    if mode == "evaluate":
        return frozenset(evaluate_defaults().keys())
    raise ValueError(f"unknown mode: {mode}")


def _coerce_config_for_mode(mode: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply JSON-friendly coercions (paths); only for keys relevant to ``mode``."""
    out = dict(cfg)
    if mode == "train":
        path_keys = ("train_tsv", "test_tsv", "checkpoint")
    elif mode == "print_dataset":
        path_keys = ("train_tsv", "test_tsv", "checkpoint")
    elif mode in ("test", "eval"):
        path_keys = ("train_tsv", "test_tsv", "checkpoint", "output")
    else:
        path_keys = ()
    for key in path_keys:
        if key in out and out[key] is not None and isinstance(out[key], str):
            out[key] = Path(out[key])
    return out


def _merge(
    defaults: dict[str, Any],
    file_cfg: dict[str, Any] | None,
    cli: dict[str, Any],
) -> dict[str, Any]:
    merged = {**defaults}
    if file_cfg:
        merged.update(file_cfg)
    merged.update(cli)
    return merged


def _checkpoint_is_missing(ckpt: Any) -> bool:
    if ckpt is None:
        return True
    if isinstance(ckpt, str) and not ckpt.strip():
        return True
    if isinstance(ckpt, Path) and not str(ckpt).strip():
        return True
    return False


def _build_train_cli_parser() -> argparse.ArgumentParser:
    pr = argparse.ArgumentParser(add_help=False)
    s = argparse.SUPPRESS
    pr.add_argument("--train-tsv", dest="train_tsv", type=Path, default=s)
    pr.add_argument("--test-tsv", dest="test_tsv", type=Path, default=s)
    pr.add_argument("--model-name", dest="model_name", default=s)
    pr.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        default=s,
        help="Where to save the best validation checkpoint.",
    )
    pr.add_argument("--epochs", dest="epochs", type=int, default=s)
    pr.add_argument("--batch-size", dest="batch_size", type=int, default=s)
    pr.add_argument("--lr", dest="lr", type=float, default=s)
    pr.add_argument("--dropout", dest="dropout", type=float, default=s)
    pr.add_argument("--max-length", dest="max_length", type=int, default=s)
    pr.add_argument("--train-ratio", dest="train_ratio", type=float, default=s)
    pr.add_argument("--seed", dest="seed", type=int, default=s)
    pr.add_argument("--num-workers", dest="num_workers", type=int, default=s)
    pr.add_argument(
        "--dataset",
        dest="dataset",
        choices=list(ALLOWED_DATASET_IDS),
        default=s,
    )
    pr.add_argument("--device", dest="device", default=s)
    pr.add_argument("--mock", dest="mock", action="store_true", default=s)
    pr.add_argument("--weight-decay", dest="weight_decay", type=float, default=s)
    pr.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=s)
    pr.add_argument(
        "--freeze-encoder-layers",
        dest="freeze_encoder_layers",
        type=int,
        default=s,
    )
    return pr


def _build_inference_cli_parser() -> argparse.ArgumentParser:
    pr = argparse.ArgumentParser(add_help=False)
    s = argparse.SUPPRESS
    pr.add_argument("--train-tsv", dest="train_tsv", type=Path, default=s)
    pr.add_argument("--test-tsv", dest="test_tsv", type=Path, default=s)
    pr.add_argument("--model-name", dest="model_name", default=s)
    pr.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        default=s,
        help="Path to checkpoint file to load for inference.",
    )
    pr.add_argument("--output", dest="output", type=Path, default=s)
    pr.add_argument("--batch-size", dest="batch_size", type=int, default=s)
    pr.add_argument("--max-length", dest="max_length", type=int, default=s)
    pr.add_argument("--dropout", dest="dropout", type=float, default=s)
    pr.add_argument("--train-ratio", dest="train_ratio", type=float, default=s)
    pr.add_argument("--seed", dest="seed", type=int, default=s)
    pr.add_argument("--num-workers", dest="num_workers", type=int, default=s)
    pr.add_argument(
        "--dataset",
        dest="dataset",
        choices=list(ALLOWED_DATASET_IDS),
        default=s,
    )
    pr.add_argument("--device", dest="device", default=s)
    pr.add_argument("--one-yes-per-question", dest="one_yes_per_question", action="store_true", default=s)
    pr.add_argument("--mock", dest="mock", action="store_true", default=s)
    return pr


def _build_evaluate_cli_parser() -> argparse.ArgumentParser:
    pr = argparse.ArgumentParser(add_help=False)
    s = argparse.SUPPRESS
    pr.add_argument("--predictions-path", dest="predictions_path", default=s)
    pr.add_argument("--gold-path", dest="gold_path", default=s)
    pr.add_argument("--strict-sample-id", dest="strict_sample_id", action="store_true", default=s)
    return pr


def _cli_overrides(parser: argparse.ArgumentParser, remainder: list[str]) -> dict[str, Any]:
    if not remainder:
        return {}
    ns = parser.parse_args(remainder)
    return dict(vars(ns))


def _load_json_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object, got {type(raw).__name__}")
    return _normalize_config_keys(cast(dict[str, Any], raw))


def _display_key_for_setting(mode: str, key: str) -> str:
    if key == "checkpoint":
        if mode == "train":
            return "checkpoint (save path — best weights are written here)"
        if mode == "print_dataset":
            return "checkpoint (unused — only data/tokenization are printed)"
        if mode in ("test", "eval"):
            return "checkpoint (load path — weights read from here for inference)"
    return key


def _print_resolved_settings(mode: str, config_path: Path | None, resolved: dict[str, Any]) -> None:
    printable: dict[str, Any] = dict(resolved)
    for k, v in list(printable.items()):
        if isinstance(v, Path):
            printable[k] = str(v)
    print("Resolved run configuration")
    print(f"  mode: {mode}")
    print(f"  config_file: {config_path if config_path is not None else '(none)'}")
    for key in sorted(printable.keys()):
        label = _display_key_for_setting(mode, key)
        print(f"  {label}: {printable[key]!r}")


def _build_root_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.main",
        description="TextGraphs baseline — train, print_dataset, inference (test/eval), or evaluate metrics; "
        "optional JSON config plus CLI overrides.",
    )
    p.add_argument(
        "mode",
        choices=("train", "print_dataset", "test", "eval", "evaluate"),
        help="train: fit checkpoint; print_dataset: show first training samples; "
        "test/eval: predict; evaluate: score predictions vs gold TSV.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file with run parameters (overrides defaults; CLI overrides file). "
        "Unknown keys are ignored.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    root = _build_root_parser()
    root_args, remainder = root.parse_known_args(argv)

    mode = root_args.mode

    raw_file_cfg: dict[str, Any] | None = None
    if root_args.config is not None:
        raw_file_cfg = _load_json_config(root_args.config)

    filtered_cfg: dict[str, Any] | None = None
    ignored_keys: list[str] = []
    if raw_file_cfg is not None:
        filtered, ignored_keys = _filter_config(mode, raw_file_cfg)
        filtered_cfg = _coerce_config_for_mode(mode, filtered)
        if ignored_keys:
            print(
                f"Ignored {len(ignored_keys)} unknown config key(s) for mode {mode!r}: "
                f"{', '.join(sorted(ignored_keys))}"
            )

    if mode == "train":
        defaults = train_defaults()
        cli_parser = _build_train_cli_parser()
    elif mode == "print_dataset":
        defaults = print_dataset_defaults()
        cli_parser = _build_train_cli_parser()
    elif mode in ("test", "eval"):
        defaults = inference_defaults()
        cli_parser = _build_inference_cli_parser()
    else:
        defaults = evaluate_defaults()
        cli_parser = _build_evaluate_cli_parser()

    cli_dict = _cli_overrides(cli_parser, remainder)
    merged = _merge(defaults, filtered_cfg, cli_dict)

    if mode in ("train", "print_dataset", "test", "eval"):
        ds_raw = str(merged.get("dataset", ALLOWED_DATASET_IDS[0]))
        if ds_raw not in ALLOWED_DATASET_IDS:
            allowed = ", ".join(sorted(ALLOWED_DATASET_IDS))
            print(
                f"error: unknown dataset {ds_raw!r}; allowed: {allowed}",
                file=sys.stderr,
            )
            sys.exit(2)

    if mode == "train" and _checkpoint_is_missing(merged.get("checkpoint")):
        print(
            "error: checkpoint is required for train (path where the best checkpoint is saved); "
            "set in config JSON or pass --checkpoint",
            file=sys.stderr,
        )
        sys.exit(2)
    if mode in ("test", "eval") and _checkpoint_is_missing(merged.get("checkpoint")):
        print(
            "error: checkpoint is required for test/eval (path to weights to load); "
            "set in config JSON or pass --checkpoint",
            file=sys.stderr,
        )
        sys.exit(2)

    if mode == "evaluate":
        if not merged.get("predictions_path"):
            print(
                "error: predictions_path is required (set in config JSON or pass --predictions-path)",
                file=sys.stderr,
            )
            sys.exit(2)
        if not merged.get("gold_path"):
            print(
                "error: gold_path is required (set in config JSON or pass --gold-path)",
                file=sys.stderr,
            )
            sys.exit(2)

    _print_resolved_settings(mode, root_args.config, merged)
    ns = argparse.Namespace(**merged)

    if mode == "train":
        run_training(ns)
    elif mode == "print_dataset":
        run_print_dataset(ns)
    elif mode in ("test", "eval"):
        run_testing(ns)
    else:
        run_evaluation(ns)


if __name__ == "__main__":
    main()
