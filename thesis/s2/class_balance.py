"""Report and compute S2 train-split class balancing multipliers."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        S2FoldSpec,
        discover_s2_folds,
    )
except ImportError:
    from common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_S2_PKL,
        LABELS,
        S2FoldSpec,
        discover_s2_folds,
    )


STAGE1_CLASS_PROB = (2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0)
CLASS_PROB_STRATEGIES = (
    "train_inverse_mean",
    "train_sqrt_inverse_mean",
    "stage1",
    "none",
)


def load_s2_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict")
    if "annotations" not in data or "split" not in data:
        raise ValueError(f"{path} must contain annotations and split")
    return data


def train_label_counts_by_fold(
    ann_file: Path,
    folds: list[S2FoldSpec],
) -> dict[str, list[int]]:
    data = load_s2_pkl(ann_file)
    annotations = data["annotations"]
    split = data["split"]
    by_frame_dir = {item["frame_dir"]: item for item in annotations}

    counts_by_fold: dict[str, list[int]] = {}
    for fold in folds:
        split_key = fold.split_key("train")
        if split_key not in split:
            raise KeyError(f"{ann_file} does not contain split {split_key!r}")
        counts = Counter()
        for frame_dir in split[split_key]:
            item = by_frame_dir[frame_dir]
            counts[int(item["label"])] += 1
        counts_by_fold[fold.fold] = [int(counts.get(index, 0)) for index in range(len(LABELS))]
    return counts_by_fold


def compute_class_prob(
    counts: list[int],
    strategy: str,
    cap: float,
) -> list[float]:
    if len(counts) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} class counts, got {len(counts)}")
    if strategy not in CLASS_PROB_STRATEGIES:
        raise ValueError(f"Unknown class-prob strategy: {strategy}")
    if cap < 1:
        raise ValueError("cap must be >= 1")

    if strategy == "none":
        return [1.0 for _ in counts]
    if strategy == "stage1":
        return list(STAGE1_CLASS_PROB)

    positive = [count for count in counts if count > 0]
    if not positive:
        raise ValueError("Cannot compute class_prob without positive class counts")
    target = sum(counts) / len(counts)

    values = []
    for count in counts:
        if count <= 0:
            values.append(1.0)
            continue
        ratio = target / count
        if strategy == "train_sqrt_inverse_mean":
            ratio = math.sqrt(ratio)
        values.append(max(1.0, min(float(cap), float(ratio))))
    return values


def format_prob(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def effective_counts(counts: list[int], class_prob: list[float]) -> list[float]:
    return [float(count) * float(multiplier) for count, multiplier in zip(counts, class_prob)]


def report_rows(
    folds: list[S2FoldSpec],
    counts_by_fold: dict[str, list[int]],
    strategies: tuple[str, ...],
    cap: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        counts = counts_by_fold[fold.fold]
        raw_total = sum(counts)
        for strategy in strategies:
            class_prob = compute_class_prob(counts, strategy, cap)
            effective = effective_counts(counts, class_prob)
            effective_total = sum(effective)
            for class_id, label in enumerate(LABELS):
                rows.append(
                    {
                        "fold": fold.fold,
                        "strategy": strategy,
                        "cap": cap,
                        "class_id": class_id,
                        "label": label,
                        "raw_count": counts[class_id],
                        "raw_percent": 0.0 if raw_total <= 0 else 100.0 * counts[class_id] / raw_total,
                        "class_prob": class_prob[class_id],
                        "effective_count": effective[class_id],
                        "effective_percent": (
                            0.0
                            if effective_total <= 0
                            else 100.0 * effective[class_id] / effective_total
                        ),
                    }
                )
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "strategy",
        "cap",
        "class_id",
        "label",
        "raw_count",
        "raw_percent",
        "class_prob",
        "effective_count",
        "effective_percent",
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
    parser = argparse.ArgumentParser(description="Report S2 train-split class balancing.")
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_S2_PKL)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: all.")
    parser.add_argument("--cap", type=float, default=4.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=CLASS_PROB_STRATEGIES,
        default=["stage1", "train_inverse_mean", "train_sqrt_inverse_mean", "none"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "eval" / "class_balance_report.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cap < 1:
        raise ValueError("--cap must be >= 1")

    folds = discover_s2_folds()
    if args.folds:
        requested = {item.lower().replace("fold_", "") for item in args.folds}
        folds = [fold for fold in folds if fold.fold in requested]
        missing = sorted(requested - {fold.fold for fold in folds})
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")

    counts_by_fold = train_label_counts_by_fold(args.ann_file, folds)
    rows = report_rows(folds, counts_by_fold, tuple(args.strategies), args.cap)
    write_report(args.output, rows, args.overwrite)

    print(f"[DONE] wrote class balance report to {args.output}")
    for fold in folds:
        counts = counts_by_fold[fold.fold]
        print(f"fold={fold.fold} train_total={sum(counts)}")
        for strategy in args.strategies:
            print(f"  {strategy}: {format_prob(compute_class_prob(counts, strategy, args.cap))}")


if __name__ == "__main__":
    main()
