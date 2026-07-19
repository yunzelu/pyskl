"""Compare E4 raw, calibrated, and Viterbi-refined test metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_EVAL_DIR,
        LABELS,
        default_calibrated_path,
        default_raw_path,
        default_viterbi_path,
        protocol_metadata,
        read_score_csv,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_EVAL_DIR,
        LABELS,
        default_calibrated_path,
        default_raw_path,
        default_viterbi_path,
        protocol_metadata,
        read_score_csv,
        write_json,
    )

from thesis.e2.evaluate import metrics_for_rows  # noqa: E402


def condition_record(condition: str, path: Path) -> dict[str, Any]:
    rows = read_score_csv(path)
    return {
        "condition": condition,
        "scores": str(path),
        "source": rows[0].source if rows else "",
        "score_type": rows[0].score_type if rows else "",
        **metrics_for_rows(rows),
    }


def write_summary_csv(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "condition",
        "scores",
        "source",
        "score_type",
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


def build_comparison(
    raw_scores: Path,
    calibrated_scores: Path,
    raw_viterbi_scores: Path,
    calibrated_viterbi_scores: Path,
    output_dir: Path,
    name: str,
    overwrite: bool,
) -> dict[str, Any]:
    records = [
        condition_record("raw", raw_scores),
        condition_record("calibrated", calibrated_scores),
        condition_record("raw_viterbi", raw_viterbi_scores),
        condition_record("calibrated_viterbi", calibrated_viterbi_scores),
    ]

    csv_path = output_dir / f"{name}_summary.csv"
    write_summary_csv(csv_path, records, overwrite)
    result = {
        "experiment": "E4",
        "stage": "comparison_evaluation",
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "summary_csv": str(csv_path),
        "summary": records,
        "note": "Temperature scaling does not change argmax metrics unless logits are otherwise transformed.",
    }
    write_json(output_dir / f"{name}_summary.json", result, overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E4 raw/calibrated/Viterbi test metrics.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--raw-scores", type=Path)
    parser.add_argument("--calibrated-scores", type=Path)
    parser.add_argument("--raw-viterbi-scores", type=Path)
    parser.add_argument("--calibrated-viterbi-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--name", default="e4_comparison")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comparison(
        raw_scores=args.raw_scores or default_raw_path(args.stream, "test"),
        calibrated_scores=args.calibrated_scores or default_calibrated_path(args.stream, "test"),
        raw_viterbi_scores=args.raw_viterbi_scores or default_viterbi_path(args.stream, "raw"),
        calibrated_viterbi_scores=(
            args.calibrated_viterbi_scores or default_viterbi_path(args.stream, "calibrated")
        ),
        output_dir=args.output_dir,
        name=args.name,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote E4 comparison to {result['summary_csv']}")
    for record in result["summary"]:
        print(
            f"{record['condition']}: "
            f"acc={record['center_time_accuracy_percent']:.4f}, "
            f"macro-F1={record['macro_f1_percent']:.4f}, "
            f"F1@50={record['F1@50']:.4f}"
        )


if __name__ == "__main__":
    main()
