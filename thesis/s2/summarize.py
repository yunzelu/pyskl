"""Build the required S2 A/B/C ablation summary table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_EVAL_DIR,
        METHODS,
        default_metrics_path,
        default_summary_table_path,
        protocol_metadata,
        read_json,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_EVAL_DIR,
        METHODS,
        default_metrics_path,
        default_summary_table_path,
        protocol_metadata,
        read_json,
        write_json,
    )


SUMMARY_METRICS = [
    "center_acc",
    "center_macro_f1",
    "state_macro_f1",
    "transition_macro_f1",
    "edit",
    "f1_10",
    "f1_25",
    "f1_50",
]

SUMMARY_COLUMNS = [
    "method",
    *SUMMARY_METRICS,
    "fold_count",
    "fold_sd_ddof",
    *[f"{metric}_fold_mean" for metric in SUMMARY_METRICS],
    *[f"{metric}_fold_sd" for metric in SUMMARY_METRICS],
]

MARKDOWN_METRICS = [
    ("center_acc", "Center Acc"),
    ("center_macro_f1", "Macro F1"),
    ("state_macro_f1", "State F1"),
    ("transition_macro_f1", "Transition F1"),
    ("edit", "Edit"),
    ("f1_10", "F1@10"),
    ("f1_25", "F1@25"),
    ("f1_50", "F1@50"),
]


def fold_metric_stats(data: dict[str, Any], path: Path) -> dict[str, Any]:
    summary = data.get("summary")
    if not isinstance(summary, list):
        raise ValueError(f"{path} does not contain summary rows")
    fold_rows = [row for row in summary if isinstance(row, dict) and row.get("scope") == "fold"]
    if not fold_rows:
        raise ValueError(f"{path} does not contain fold summary rows")

    sd_ddof = 1 if len(fold_rows) > 1 else 0
    stats: dict[str, Any] = {
        "fold_count": len(fold_rows),
        "folds": [str(row.get("fold") or "") for row in fold_rows],
        "sd_ddof": sd_ddof,
        "mean": {},
        "sd": {},
    }
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in fold_rows]
        mean = sum(values) / len(values)
        denominator = len(values) - sd_ddof
        sd = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / denominator)
            if denominator > 0
            else 0.0
        )
        stats["mean"][metric] = mean
        stats["sd"][metric] = sd
    return stats


def load_metrics(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = read_json(path)
    overall = data.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"{path} does not contain an overall metrics object")
    return overall, fold_metric_stats(data, path)


def format_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: f"{value:.6f}" if isinstance((value := row[key]), float) else value
        for key in SUMMARY_COLUMNS
    }


def format_mean_sd(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_fold_mean']:.2f} +/- {row[f'{metric}_fold_sd']:.2f}"


def write_markdown_summary(path: Path, stream: str, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["Method", "n", *[label for _, label in MARKDOWN_METRICS]]
    aligns = ["---", "---:", *["---:" for _ in MARKDOWN_METRICS]]
    lines = [
        f"# S2 {stream} Ablation Summary",
        "",
        "Values are fold mean +/- SD. SD uses sample standard deviation (`ddof=1`) when more than one fold is present.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        values = [
            str(row["method"]),
            str(row["fold_count"]),
            *[format_mean_sd(row, metric) for metric, _ in MARKDOWN_METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    stream: str,
    output: Path,
    metrics_paths: dict[str, Path],
    overwrite: bool,
    markdown_output: Path | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        path = metrics_paths[method]
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for Method {method}: {path}")
        overall, fold_stats = load_metrics(path)
        rows.append(
            {
                "method": method,
                "center_acc": float(overall["center_acc"]),
                "center_macro_f1": float(overall["center_macro_f1"]),
                "state_macro_f1": float(overall["state_macro_f1"]),
                "transition_macro_f1": float(overall["transition_macro_f1"]),
                "edit": float(overall["edit"]),
                "f1_10": float(overall["f1_10"]),
                "f1_25": float(overall["f1_25"]),
                "f1_50": float(overall["f1_50"]),
                "fold_count": int(fold_stats["fold_count"]),
                "fold_sd_ddof": int(fold_stats["sd_ddof"]),
                **{
                    f"{metric}_fold_mean": float(fold_stats["mean"][metric])
                    for metric in SUMMARY_METRICS
                },
                **{
                    f"{metric}_fold_sd": float(fold_stats["sd"][metric])
                    for metric in SUMMARY_METRICS
                },
            }
        )

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_csv_row(row))

    markdown_path = markdown_output or output.with_suffix(".md")
    write_markdown_summary(markdown_path, stream, rows, overwrite)

    write_json(
        output.with_name(f"{output.stem}.json"),
        {
            "experiment": "S2",
            "stage": "ablation_summary",
            "stream": stream,
            "protocol": protocol_metadata(),
            "summary_csv": str(output),
            "summary_markdown": str(markdown_path),
            "fold_statistic": "mean +/- sample SD across fold rows",
            "rows": rows,
            "interpretation": {
                "A_vs_B": "effect of matching training and continuous inference format",
                "B_vs_C": "effect of representing mixed-window temporal content with soft targets",
            },
        },
        overwrite=overwrite,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write S2 A/B/C ablation summary.")
    parser.add_argument("--stream", choices=["joint", "limb", "fusion"], default="joint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-A", type=Path)
    parser.add_argument("--metrics-B", type=Path)
    parser.add_argument("--metrics-C", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_paths = {
        "A": args.metrics_A or default_metrics_path("A", args.stream),
        "B": args.metrics_B or default_metrics_path("B", args.stream),
        "C": args.metrics_C or default_metrics_path("C", args.stream),
    }
    output = args.output or default_summary_table_path(args.stream)
    markdown_output = args.markdown_output or output.with_suffix(".md")
    rows = build_summary(args.stream, output, metrics_paths, args.overwrite, markdown_output)
    print(f"[DONE] wrote S2 ablation summary to {output}")
    print(f"[DONE] wrote S2 ablation Markdown table to {markdown_output}")
    for row in rows:
        print(
            f"{row['method']}: macro-F1={format_mean_sd(row, 'center_macro_f1')}, "
            f"F1@50={format_mean_sd(row, 'f1_50')}"
        )


if __name__ == "__main__":
    main()
