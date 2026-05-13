"""Evaluation with alignment diagnostics (sample_id merge, row counts)."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions vs gold TSV with alignment diagnostics.",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="TSV with 'prediction' (0/1) and ideally 'sample_id'.",
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        required=True,
        help="TSV with 'correct' (False/True) and ideally 'sample_id'.",
    )
    parser.add_argument(
        "--strict-sample-id",
        action="store_true",
        help="Exit with error if gold and predictions sample_id sets differ.",
    )
    return parser.parse_args()


def _print_duplicate_sample_ids(name: str, df: pd.DataFrame) -> None:
    if "sample_id" not in df.columns:
        return
    dup_mask = df["sample_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        return
    n_unique_dup_ids = df.loc[dup_mask, "sample_id"].nunique()
    print(
        f"  Duplicates: {n_dup_rows} rows share repeated sample_id values "
        f"({n_unique_dup_ids} distinct repeated ids) in {name}."
    )
    print(f"  Example repeated sample_id values: {df.loc[dup_mask, 'sample_id'].head(8).tolist()}")


def _diagnose_alignment(
    predictions_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    print("=== Alignment diagnostics ===\n")
    print(f"Predictions file: {len(predictions_df)} rows, columns: {list(predictions_df.columns)}")
    print(f"Gold file:          {len(gold_df)} rows, columns: {list(gold_df.columns)}\n")

    pred_has_sid = "sample_id" in predictions_df.columns
    gold_has_sid = "sample_id" in gold_df.columns

    if not pred_has_sid or not gold_has_sid:
        print(
            "sample_id: "
            f"predictions={'yes' if pred_has_sid else 'MISSING'}, "
            f"gold={'yes' if gold_has_sid else 'MISSING'}."
        )
        if not pred_has_sid or not gold_has_sid:
            print(
                "\nWithout sample_id on both sides, rows are paired by position (row i with row i).\n"
                "If files are different lengths or row order does not match, metrics will be wrong.\n"
            )
        return None

    gold_ids = set(gold_df["sample_id"].astype(int))
    pred_ids = set(predictions_df["sample_id"].astype(int))

    only_gold = gold_ids - pred_ids
    only_pred = pred_ids - gold_ids
    common = gold_ids & pred_ids

    print(f"Distinct sample_id in gold:        {len(gold_ids)}")
    print(f"Distinct sample_id in predictions: {len(pred_ids)}")
    print(f"In both:                          {len(common)}")
    print(f"Only in gold (missing in pred):   {len(only_gold)}")
    print(f"Only in predictions (extra):      {len(only_pred)}")

    if only_gold:
        sample = sorted(only_gold)[:15]
        print(f"\n  Examples of sample_id only in gold: {sample}{' ...' if len(only_gold) > 15 else ''}")
    if only_pred:
        sample = sorted(only_pred)[:15]
        print(f"\n  Examples of sample_id only in predictions: {sample}{' ...' if len(only_pred) > 15 else ''}")

    print()
    _print_duplicate_sample_ids("gold", gold_df)
    _print_duplicate_sample_ids("predictions", predictions_df)
    print()

    merged = gold_df[["sample_id", "correct"]].merge(
        predictions_df[["sample_id", "prediction"]],
        on="sample_id",
        how="inner",
    )

    if len(merged) != len(common):
        print(
            f"WARNING: inner merge yields {len(merged)} rows but |gold ∩ pred ids| = {len(common)} "
            "(likely duplicate sample_id on one or both sides).\n"
        )

    merged["label"] = merged["correct"].astype(np.float32)
    return merged


def _print_paired_label_counts(true_labels: np.ndarray, pred_labels: np.ndarray) -> None:
    t = true_labels.astype(np.int64)
    p = pred_labels.astype(np.int64)
    print("=== Label counts (paired rows) ===\n")
    print(f"  Gold:        0 → {int(np.sum(t == 0))},  1 → {int(np.sum(t == 1))}")
    print(f"  Predictions: 0 → {int(np.sum(p == 0))},  1 → {int(np.sum(p == 1))}\n")


def run_evaluation(args: argparse.Namespace) -> None:
    predictions_df = pd.read_csv(args.predictions_path, sep="\t")
    test_df = pd.read_csv(args.gold_path, sep="\t")

    if "prediction" not in predictions_df.columns:
        raise RuntimeError("prediction column is not found in submission file")

    predictions_unique_values = {int(x) for x in predictions_df["prediction"].unique()}
    if not predictions_unique_values.issubset({0, 1}):
        raise RuntimeError(
            f"prediction column must contain only 0 and 1; got unique values: {sorted(predictions_unique_values)}"
        )

    if "correct" not in test_df.columns:
        raise RuntimeError("gold file must contain a 'correct' column")

    aligned = _diagnose_alignment(predictions_df, test_df)

    if aligned is not None:
        if len(aligned) != len(test_df) or len(aligned) != len(predictions_df):
            gold_ids = set(test_df["sample_id"].astype(int))
            pred_ids = set(predictions_df["sample_id"].astype(int))
            if gold_ids != pred_ids:
                msg = (
                    f"Gold and predictions sample_id sets are not identical: "
                    f"gold-only {len(gold_ids - pred_ids)}, pred-only {len(pred_ids - gold_ids)}."
                )
                print(f"ISSUE: {msg}\n")
                if args.strict_sample_id:
                    sys.exit(1)
                print(
                    "Evaluating on the inner join only (intersection of sample_id). "
                    "Scores ignore rows missing on either side.\n"
                )

        pred_labels = aligned["prediction"].astype(np.int32).values
        true_labels = aligned["label"].astype(np.int32).values
    else:
        if len(predictions_df) != len(test_df):
            print(
                f"ISSUE: Row count mismatch — predictions {len(predictions_df)} vs gold {len(test_df)}.\n"
                "Cannot pair rows without sample_id; add sample_id to both TSVs and re-run.\n"
            )
            sys.exit(1)
        print("Pairing by row order (both files have the same length, no sample_id merge).\n")
        test_df = test_df.copy()
        test_df["label"] = test_df["correct"].astype(np.float32)
        pred_labels = predictions_df["prediction"].astype(np.int32).values
        true_labels = test_df["label"].astype(np.int32).values

    if pred_labels.shape[0] != true_labels.shape[0]:
        raise RuntimeError(
            f"Internal error: aligned length mismatch pred={pred_labels.shape[0]} gold={true_labels.shape[0]}"
        )

    _print_paired_label_counts(true_labels, pred_labels)

    print(f"=== Metrics ({len(true_labels)} paired rows) ===\n")

    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    print("Test\n")
    print(f"\tPublic precision: {p}\n")
    print(f"\tPublic recall: {r}\n")
    print(f"\tPublic F1: {f1}\n")
    print(f"\tPublic accuracy: {acc}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
