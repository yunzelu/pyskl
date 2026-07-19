"""Tune E4 Viterbi smoothing strength on validation subjects."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import (
        LABELS,
        apply_viterbi,
        default_scores_path,
        default_tuning_path,
        manual_transition_matrix,
        parse_lambda_grid,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_transition_matrix,
    )
except ImportError:
    from common import (
        LABELS,
        apply_viterbi,
        default_scores_path,
        default_tuning_path,
        manual_transition_matrix,
        parse_lambda_grid,
        protocol_metadata,
        read_score_csv,
        write_json,
        write_transition_matrix,
    )

from thesis.e2.evaluate import metrics_for_rows  # noqa: E402

DEFAULT_LAMBDA_GRID = "0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3"


def rows_by_fold(rows) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.fold].append(row)
    return dict(sorted(grouped.items()))


def metric_record(scope: str, fold: str, lam: float, rows) -> dict[str, Any]:
    metrics = metrics_for_rows(rows)
    objective = metrics["macro_f1_percent"] + metrics["F1@50"]
    return {
        "scope": scope,
        "fold": fold,
        "lambda": lam,
        "objective_macro_f1_plus_f1_at_50": objective,
        **metrics,
    }


def tune_lambda(rows, lambdas: list[float]) -> tuple[float, list[dict[str, Any]]]:
    if any(row.score_type != "prob" for row in rows):
        raise ValueError("Viterbi tuning expects probability rows")

    transition_matrix = manual_transition_matrix()
    records: list[dict[str, Any]] = []
    overall_records: list[dict[str, Any]] = []

    for lam in lambdas:
        decoded = apply_viterbi(rows, transition_matrix, lam)
        overall = metric_record("overall", "", lam, decoded)
        records.append(overall)
        overall_records.append(overall)

        for fold, fold_rows in rows_by_fold(decoded).items():
            records.append(metric_record("fold", fold, lam, fold_rows))

    selected = sorted(
        overall_records,
        key=lambda row: (
            -float(row["objective_macro_f1_plus_f1_at_50"]),
            -float(row["macro_f1_percent"]),
            -float(row["F1@50"]),
            float(row["lambda"]),
        ),
    )[0]
    return float(selected["lambda"]), records


def write_tuning_csv(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "scope",
        "fold",
        "lambda",
        "objective_macro_f1_plus_f1_at_50",
        "total_windows",
        "evaluated_windows",
        "skipped_windows",
        "center_time_accuracy_percent",
        "macro_f1_percent",
        "transition_class_f1_percent",
        "Edit",
        "F1@10",
        "F1@25",
        "F1@50",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in record.items()
                }
            )


def build_tuning_report(
    val_scores: Path,
    output: Path,
    lambdas: list[float],
    score_kind: str,
    overwrite: bool,
) -> dict[str, Any]:
    rows = read_score_csv(val_scores)
    selected_lambda, records = tune_lambda(rows, lambdas)
    csv_path = output.with_suffix(".csv")
    matrix_outputs = write_transition_matrix(output.parent, overwrite=overwrite)
    write_tuning_csv(csv_path, records, overwrite=overwrite)

    selected_record = next(
        record
        for record in records
        if record["scope"] == "overall" and float(record["lambda"]) == selected_lambda
    )
    result = {
        "experiment": "E4",
        "stage": "viterbi_lambda_tuning",
        "score_kind": score_kind,
        "validation_scores": str(val_scores),
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "lambda_grid": lambdas,
        "selection_objective": "macro_f1_percent + F1@50 on validation subjects",
        "selected_lambda": selected_lambda,
        "selected_overall_metrics": selected_record,
        "tuning_csv": str(csv_path),
        "transition_matrix": matrix_outputs,
        "grid": records,
    }
    write_json(output, result, overwrite=overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune E4 Viterbi lambda on validation scores.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--score-kind", choices=["raw", "calibrated"], default="calibrated")
    parser.add_argument("--val-scores", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lambda-grid", default=DEFAULT_LAMBDA_GRID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_scores = args.val_scores or default_scores_path(args.stream, "val", args.score_kind)
    output = args.output or default_tuning_path(args.stream, args.score_kind)
    result = build_tuning_report(
        val_scores=val_scores,
        output=output,
        lambdas=parse_lambda_grid(args.lambda_grid),
        score_kind=args.score_kind,
        overwrite=args.overwrite,
    )
    metrics = result["selected_overall_metrics"]
    print(f"[DONE] wrote Viterbi tuning reports to {output.parent}")
    print(
        f"Selected lambda={result['selected_lambda']:.6g}: "
        f"macro-F1={metrics['macro_f1_percent']:.4f}, "
        f"F1@50={metrics['F1@50']:.4f}, "
        f"objective={metrics['objective_macro_f1_plus_f1_at_50']:.4f}"
    )


if __name__ == "__main__":
    main()
