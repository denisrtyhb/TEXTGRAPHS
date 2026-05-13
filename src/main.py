"""Unified CLI: ``python -m src.main train ...`` or ``python -m src.main test ...``."""

from __future__ import annotations

import argparse
import sys

from test import build_test_parser, run_testing
from train import build_train_parser, run_training


def main() -> None:
    root = argparse.ArgumentParser(
        prog="python -m src.main",
        description="TextGraphs baseline (see baselines/bert_baselines.ipynb).",
    )
    root.add_argument(
        "command",
        nargs="?",
        choices=("train", "test"),
        help="train: fit and save checkpoint; test: load checkpoint and write predictions TSV.",
    )
    root_args, remainder = root.parse_known_args()

    if root_args.command is None:
        root.print_help()
        sys.exit(1)

    if root_args.command == "train":
        args = build_train_parser().parse_args(remainder)
        run_training(args)
    else:
        args = build_test_parser().parse_args(remainder)
        run_testing(args)


if __name__ == "__main__":
    main()
