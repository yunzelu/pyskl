"""Build the required S2 A/B/C ablation summary table."""

from __future__ import annotations

import argparse
import csv
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


SUMMARY_COLUMNS = [
    "method",
    "center_acc",
    "center_macro_f1",
    "state_macro_f1",
    "transition_macro_f1",
    "edit",
    "f1_10",
    "f1_25",
    "f1_50",
]


def load_overall(path: Path) -> dict[str, Any]:
    data = read_json(path)
    overall = data.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"{path} does not contain an overall metrics object")
    return overall


def build_summary(stream: str, output: Path, metrics_paths: dict[str, Path], overwrite: bool) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        path = metrics_paths[method]
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for Method {method}: {path}")
        overall = load_overall(path)
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
            }
        )

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    write_json(
        output.with_name(f"{output.stem}.json"),
        {
            "experiment": "S2",
            "stage": "ablation_summary",
            "stream": stream,
            "protocol": protocol_metadata(),
            "summary_csv": str(output),
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
    rows = build_summary(args.stream, output, metrics_paths, args.overwrite)
    print(f"[DONE] wrote S2 ablation summary to {output}")
    for row in rows:
        print(
            f"{row['method']}: acc={row['center_acc']:.4f}, "
            f"macro-F1={row['center_macro_f1']:.4f}, F1@50={row['f1_50']:.4f}"
        )


if __name__ == "__main__":
    main()
