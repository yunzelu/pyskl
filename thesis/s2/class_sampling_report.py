"""Report S2 epoch-wise class sampling probabilities and expected draws."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .class_balance import train_label_counts_by_fold
    from .common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        S2FoldSpec,
        discover_s2_folds,
    )
except ImportError:
    from class_balance import train_label_counts_by_fold
    from common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        S2FoldSpec,
        discover_s2_folds,
    )


CLASS_SAMPLE_STRATEGIES = ("sqrt", "power", "none")


def class_sampling_probs(
    counts: list[int],
    strategy: str,
    power: float,
) -> list[float]:
    if len(counts) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} class counts, got {len(counts)}")
    if strategy not in CLASS_SAMPLE_STRATEGIES:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
    if power < 0:
        raise ValueError("power must be non-negative")
    if strategy == "sqrt" and abs(power - 0.5) > 1e-12:
        raise ValueError("strategy='sqrt' requires power=0.5")

    if strategy == "none":
        weights = [float(count) for count in counts]
    else:
        weights = [float(count) ** power if count > 0 else 0.0 for count in counts]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Cannot compute sampling probabilities without positive class counts")
    return [weight / total for weight in weights]


def rows_for_fold(
    fold: S2FoldSpec,
    counts: list[int],
    strategy: str,
    power: float,
    epoch_size: int,
) -> list[dict[str, Any]]:
    raw_total = sum(counts)
    probs = class_sampling_probs(counts, strategy, power)
    rows: list[dict[str, Any]] = []
    for class_id, (label, count, prob) in enumerate(zip(LABELS, counts, probs)):
        expected = prob * epoch_size
        rows.append(
            {
                "fold": fold.fold,
                "class_id": class_id,
                "label": label,
                "raw_count": count,
                "raw_percent": 0.0 if raw_total <= 0 else 100.0 * count / raw_total,
                "sample_strategy": strategy,
                "sample_power": power,
                "epoch_size": epoch_size,
                "class_sample_probability": prob,
                "expected_epoch_samples": expected,
                "expected_epoch_percent": 100.0 * prob,
                "expected_per_materialized_window": 0.0 if count <= 0 else expected / count,
            }
        )
    return rows


def build_report_rows(
    folds: list[S2FoldSpec],
    counts_by_fold: dict[str, list[int]],
    strategy: str,
    power: float,
    epoch_size: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        counts = counts_by_fold[fold.fold]
        fold_epoch_size = int(epoch_size) if epoch_size is not None else int(sum(counts))
        rows.extend(rows_for_fold(fold, counts, strategy, power, fold_epoch_size))
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "class_id",
        "label",
        "raw_count",
        "raw_percent",
        "sample_strategy",
        "sample_power",
        "epoch_size",
        "class_sample_probability",
        "expected_epoch_samples",
        "expected_epoch_percent",
        "expected_per_materialized_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report S2 class sampling probabilities.")
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_S2_PKL)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: all.")
    parser.add_argument("--strategy", choices=CLASS_SAMPLE_STRATEGIES, default="sqrt")
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=None,
        help="Global samples drawn per epoch. Default: each fold's train-window count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "eval" / "class_sampling_report.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.power < 0:
        raise ValueError("--power must be non-negative")
    if args.strategy == "sqrt" and abs(args.power - 0.5) > 1e-12:
        raise ValueError("--strategy sqrt requires --power 0.5")
    if args.epoch_size is not None and args.epoch_size <= 0:
        raise ValueError("--epoch-size must be positive")

    folds = discover_s2_folds()
    if args.folds:
        requested = {item.lower().replace("fold_", "") for item in args.folds}
        folds = [fold for fold in folds if fold.fold in requested]
        missing = sorted(requested - {fold.fold for fold in folds})
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")

    counts_by_fold = train_label_counts_by_fold(args.ann_file, folds)
    rows = build_report_rows(
        folds=folds,
        counts_by_fold=counts_by_fold,
        strategy=args.strategy,
        power=args.power,
        epoch_size=args.epoch_size,
    )
    write_report(args.output, rows, args.overwrite)

    print(f"[DONE] wrote class sampling report to {args.output}")
    for fold in folds:
        fold_rows = [row for row in rows if row["fold"] == fold.fold]
        counts = Counter({row["label"]: row["raw_count"] for row in fold_rows})
        print(f"fold={fold.fold} train_total={sum(counts.values())}")
        for row in fold_rows:
            print(
                "  "
                f"{row['label']}: n={row['raw_count']} "
                f"p={row['class_sample_probability']:.6f} "
                f"expected={row['expected_epoch_samples']:.1f}"
            )


if __name__ == "__main__":
    main()
