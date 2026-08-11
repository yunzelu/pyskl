"""Summarize metrics across recursive best_eval.json files.

Example:
    python thesis/s1/summarize_best_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ("top1_acc", "mean_class_accuracy", "macro_f1")
DEFAULT_ROOTS = (
    Path("work_dirs/posec3d/911"),
    Path("work_dirs/stgcn++/911"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and standard deviation for metrics stored in "
            "recursive best_eval.json files."
        )
    )
    parser.add_argument(
        "--root",
        "--roots",
        dest="roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Root directory/directories to search recursively.",
    )
    parser.add_argument(
        "--filename",
        default="best_eval.json",
        help="Evaluation JSON filename to collect.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric keys to summarize.",
    )
    parser.add_argument(
        "--population",
        action="store_true",
        help="Use population standard deviation instead of sample standard deviation.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write the summary as CSV.",
    )
    return parser.parse_args()


def read_metric_values(
    root: Path,
    filename: str,
    metrics: list[str],
) -> tuple[list[Path], dict[str, list[float]], list[str]]:
    paths = sorted(root.rglob(filename))
    values = {metric: [] for metric in metrics}
    warnings: list[str] = []

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipping {path}: {exc}")
            continue

        for metric in metrics:
            raw_value = data.get(metric)
            if raw_value is None:
                warnings.append(f"{path} is missing {metric}")
                continue

            try:
                values[metric].append(float(raw_value))
            except (TypeError, ValueError):
                warnings.append(f"{path} has non-numeric {metric}: {raw_value!r}")

    return paths, values, warnings


def summarize(
    values_by_metric: dict[str, list[float]],
    population: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for metric, values in values_by_metric.items():
        if not values:
            rows.append(
                {
                    "metric": metric,
                    "n": 0,
                    "mean": None,
                    "sd": None,
                    "min": None,
                    "max": None,
                }
            )
            continue

        sd = 0.0
        if len(values) > 1:
            sd = statistics.pstdev(values) if population else statistics.stdev(values)

        rows.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "sd": sd,
                "min": min(values),
                "max": max(values),
            }
        )

    return rows


def root_label(root: Path) -> str:
    parts = root.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(root)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_mean_sd(row: dict[str, Any]) -> str:
    if row["mean"] is None or row["sd"] is None:
        return ""
    return f"{row['mean']:.6f} (+/- {row['sd']:.6f})"


def print_table(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    headers = ["root", "files", *metrics]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["root"]),
                str(row["file_count"]),
                *(str(row.get(metric, "")) for metric in metrics),
            ]
        )

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table_rows))
        for idx in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def write_csv(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["root", "file_count", *metrics]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    for root in args.roots:
        paths, values_by_metric, warnings = read_metric_values(
            root,
            args.filename,
            args.metrics,
        )
        summary_rows = summarize(values_by_metric, population=args.population)
        all_warnings.extend(warnings)

        row: dict[str, Any] = {
            "root": root_label(root),
            "file_count": len(paths),
        }
        for summary_row in summary_rows:
            row[summary_row["metric"]] = fmt_mean_sd(summary_row)
        all_rows.append(row)

    sd_kind = "population SD" if args.population else "sample SD"
    print(f"Format: mean (+/- sd), using {sd_kind}")
    print_table(all_rows, metrics=args.metrics)

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"- {warning}")

    if args.csv_out is not None:
        write_csv(args.csv_out, all_rows, metrics=args.metrics)
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
