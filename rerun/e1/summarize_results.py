"""Summarize E1 single-stream and joint/bone fusion results."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_continuous_segmental import (
    evaluate_continuous_segmental,
    parse_background_labels,
)


LABELS = [
    "lie-stationary",
    "sit-stationary",
    "walk",
    "fall",
    "transition-lie-to-sit",
    "transition-lie-to-stand",
    "transition-sit-to-lie",
    "transition-sit-to-stand",
    "transition-stand-to-sit",
]

FOLDS = ["a", "b", "c"]
STREAMS = ["joint", "bone"]


@dataclass(frozen=True)
class Condition:
    key: str
    result_dir: str
    pkl_protocol_dir: str
    pkl_stem: str


CONDITIONS = [
    Condition(
        key="a1",
        result_dir="a1_activity_aligned",
        pkl_protocol_dir="activity_aligned",
        pkl_stem="radarv4_yolo26xpose_activity_aligned_fold_{fold}.pkl",
    ),
    Condition(
        key="a2",
        result_dir="a2_activity_checkpoint_on_continuous",
        pkl_protocol_dir="continuous_window_w60_s12",
        pkl_stem="radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl",
    ),
    Condition(
        key="b",
        result_dir="b_continuous_window",
        pkl_protocol_dir="continuous_window_w60_s12",
        pkl_stem="radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl",
    ),
]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pkl_path(data_root: Path, condition: Condition, fold: str) -> Path:
    return (
        data_root
        / condition.pkl_protocol_dir
        / condition.pkl_stem.format(fold=fold)
    )


def split_annotations(pkl_file: Path, split_name: str) -> list[dict[str, Any]]:
    data = load_pickle(pkl_file)
    annotations = data["annotations"]
    identifier = "filename" if "filename" in annotations[0] else "frame_dir"
    split_ids = set(data["split"][split_name])
    return [item for item in annotations if item[identifier] in split_ids]


def load_labels_and_ids(pkl_file: Path, split_name: str) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    annotations = split_annotations(pkl_file, split_name)
    labels = np.array([int(item["label"]) for item in annotations], dtype=np.int64)
    ids = [str(item.get("frame_dir", item.get("filename"))) for item in annotations]
    return labels, ids, annotations


def load_scores(path: Path) -> np.ndarray:
    scores = np.asarray(load_pickle(path), dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score array in {path}, got shape {scores.shape}")
    if scores.shape[1] != len(LABELS):
        raise ValueError(
            f"Expected {len(LABELS)} classes in {path}, got {scores.shape[1]}"
        )
    return scores


def scores_to_probabilities(scores: np.ndarray) -> tuple[np.ndarray, str]:
    row_sums = scores.sum(axis=1)
    looks_like_prob = (
        np.all(scores >= -1e-6)
        and np.all(scores <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    )
    if looks_like_prob:
        return scores, "already_probabilities"

    stable = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(stable)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True), "softmax_applied"


def metric_dict(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    top1 = float(np.mean(predictions == labels))
    num_classes = scores.shape[1]
    confusion = np.bincount(
        num_classes * labels + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).astype(np.float64)
    tp = np.diag(confusion)
    pred_count = confusion.sum(axis=0)
    true_count = confusion.sum(axis=1)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count != 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)
    return {"top1_acc": top1, "macro_f1": float(np.mean(f1))}


def result_paths(work_root: Path, fold: str, stream: str, condition: Condition) -> tuple[Path, Path]:
    base = work_root / f"fold_{fold}" / stream / condition.result_dir
    return base / "best_pred.pkl", base / "best_eval.json"


def add_single_stream_rows(
    rows: list[dict[str, Any]],
    work_root: Path,
    data_root: Path,
    fold: str,
    condition: Condition,
) -> dict[str, np.ndarray]:
    labels, ids, _ = load_labels_and_ids(pkl_path(data_root, condition, fold), "test")
    score_by_stream: dict[str, np.ndarray] = {}
    for stream in STREAMS:
        pred_path, eval_path = result_paths(work_root, fold, stream, condition)
        scores = load_scores(pred_path)
        probs, score_format = scores_to_probabilities(scores)
        if len(probs) != len(labels):
            raise ValueError(
                f"{pred_path} has {len(probs)} predictions, expected {len(labels)}"
            )
        metrics = metric_dict(probs, labels)
        eval_metrics = load_json(eval_path) if eval_path.exists() else {}
        for key, value in metrics.items():
            if key in eval_metrics and abs(float(eval_metrics[key]) - value) > 1e-6:
                raise ValueError(
                    f"{eval_path} {key}={eval_metrics[key]} does not match "
                    f"recomputed {value}"
                )
        rows.append(
            {
                "condition": condition.key,
                "fold": fold,
                "stream": stream,
                "num_samples": len(labels),
                "score_format": score_format,
                **metrics,
            }
        )
        score_by_stream[stream] = probs
    return score_by_stream


def add_fusion_row(
    rows: list[dict[str, Any]],
    work_root: Path,
    data_root: Path,
    fold: str,
    condition: Condition,
    score_by_stream: dict[str, np.ndarray],
    write_predictions: bool,
) -> None:
    labels, ids, _ = load_labels_and_ids(pkl_path(data_root, condition, fold), "test")
    joint = score_by_stream["joint"]
    bone = score_by_stream["bone"]
    if joint.shape != bone.shape:
        raise ValueError(f"Joint/bone shape mismatch for {condition.key} fold {fold}")
    fusion_scores = 0.5 * (joint + bone)
    metrics = metric_dict(fusion_scores, labels)

    fusion_dir = work_root / f"fold_{fold}" / "fusion" / condition.result_dir
    if write_predictions:
        fusion_dir.mkdir(parents=True, exist_ok=True)
        with (fusion_dir / "best_pred.pkl").open("wb") as f:
            pickle.dump(
                fusion_scores.astype(np.float32).tolist(),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        write_json(
            fusion_dir / "best_eval.json",
            {
                **metrics,
                "fusion_rule": "0.5 * (joint_probability + bone_probability)",
                "num_samples": len(labels),
                "source_order": "PYSKL PoseDataset test order",
            },
        )

    rows.append(
        {
            "condition": condition.key,
            "fold": fold,
            "stream": "fusion",
            "num_samples": len(labels),
            "score_format": "mean_joint_bone_probabilities",
            **metrics,
        }
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["stream"]))].append(row)

    summary: list[dict[str, Any]] = []
    for condition in [item.key for item in CONDITIONS]:
        for stream in [*STREAMS, "fusion"]:
            fold_rows = grouped[(condition, stream)]
            if len(fold_rows) != len(FOLDS):
                raise ValueError(
                    f"Expected {len(FOLDS)} folds for {condition}/{stream}, "
                    f"found {len(fold_rows)}"
                )
            top1 = np.array([float(row["top1_acc"]) for row in fold_rows])
            macro_f1 = np.array([float(row["macro_f1"]) for row in fold_rows])
            summary.append(
                {
                    "condition": condition,
                    "stream": stream,
                    "folds": len(fold_rows),
                    "top1_acc_mean": float(np.mean(top1)),
                    "top1_acc_sd": float(np.std(top1, ddof=1)),
                    "macro_f1_mean": float(np.mean(macro_f1)),
                    "macro_f1_sd": float(np.std(macro_f1, ddof=1)),
                }
            )
    return summary


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.4f} +- {sd:.4f}"


def format_segmental_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.2f} +- {sd:.2f}"


def markdown_report(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    temporal: dict[str, Any],
    segmental: dict[str, Any] | None,
) -> str:
    lines = [
        "# E1 Result Summary",
        "",
        "Fusion uses `0.5 * (joint_probability + bone_probability)`. The saved",
        "`best_pred.pkl` files already contain softmax probabilities from PYSKL, so",
        "fusion is applied directly to those probabilities.",
        "",
        "## Mean +- SD Across Folds",
        "",
        "| Condition | Stream | Top-1 Acc | Macro F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {condition} | {stream} | {top1} | {f1} |".format(
                condition=str(row["condition"]).upper(),
                stream=row["stream"],
                top1=format_mean_sd(row["top1_acc_mean"], row["top1_acc_sd"]),
                f1=format_mean_sd(row["macro_f1_mean"], row["macro_f1_sd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| Condition | Fold | Stream | N | Top-1 Acc | Macro F1 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {condition} | {fold} | {stream} | {n} | {top1:.4f} | {f1:.4f} |".format(
                condition=str(row["condition"]).upper(),
                fold=str(row["fold"]).upper(),
                stream=row["stream"],
                n=int(row["num_samples"]),
                top1=float(row["top1_acc"]),
                f1=float(row["macro_f1"]),
            )
        )

    if segmental is not None:
        lines.extend(
            [
                "",
                "## Continuous Segmental Metrics",
                "",
                "Segmental metrics are computed only for continuous-window evaluation",
                "conditions A2 and B. Each recording is evaluated independently;",
                "segmental F1 pools TP, FP, and FN counts across recordings within",
                "a fold, while Edit is normalized per recording and then averaged.",
                "",
                "| Condition | Stream | Edit | F1@10 | F1@25 | F1@50 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in segmental["mean_sd"]:
            lines.append(
                "| {condition} | {stream} | {edit} | {f10} | {f25} | {f50} |".format(
                    condition=str(row["condition"]).upper(),
                    stream=row["stream"],
                    edit=format_segmental_mean_sd(row["edit_mean"], row["edit_sd"]),
                    f10=format_segmental_mean_sd(row["f1_10_mean"], row["f1_10_sd"]),
                    f25=format_segmental_mean_sd(row["f1_25_mean"], row["f1_25_sd"]),
                    f50=format_segmental_mean_sd(row["f1_50_mean"], row["f1_50_sd"]),
                )
            )

        lines.extend(
            [
                "",
                "| Condition | Fold | Stream | Recordings | Windows | Edit | F1@10 | F1@25 | F1@50 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in segmental["fold_metrics"]:
            lines.append(
                "| {condition} | {fold} | {stream} | {recordings} | {windows} | "
                "{edit:.2f} | {f10:.2f} | {f25:.2f} | {f50:.2f} |".format(
                    condition=str(row["condition"]).upper(),
                    fold=str(row["fold"]).upper(),
                    stream=row["stream"],
                    recordings=int(row["num_recordings"]),
                    windows=int(row["num_windows"]),
                    edit=float(row["edit"]),
                    f10=float(row["f1_10"]),
                    f25=float(row["f1_25"]),
                    f50=float(row["f1_50"]),
                )
            )

    lines.extend(
        [
            "",
            "## Continuous Segmental Metric Readiness",
            "",
            "Continuous-window pkls support per-recording temporal evaluation.",
            "Every test sample has `session_name`, `window_row_start`,",
            "`center_source_frame`, `center_timestamp_sec`, `label`, and",
            "`frame_dir`. For segmental Edit/F1@k, group by `session_name` and",
            "sort each group by `window_row_start` or `center_source_frame`.",
            "Do not concatenate different `session_name` groups.",
            "",
            "| Fold | Test Sequences | Test Windows | Missing Required Fields | Ordering Issues |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in FOLDS:
        item = temporal[fold]
        lines.append(
            f"| {fold.upper()} | {item['num_sequences']} | {item['num_windows']} | "
            f"{item['missing_required_fields']} | {item['ordering_issues']} |"
        )
    return "\n".join(lines) + "\n"


def temporal_support_report(data_root: Path) -> dict[str, Any]:
    required = [
        "frame_dir",
        "session_name",
        "window_row_start",
        "center_source_frame",
        "center_timestamp_sec",
        "label",
    ]
    condition = CONDITIONS[1]
    report: dict[str, Any] = {}
    for fold in FOLDS:
        annotations = split_annotations(pkl_path(data_root, condition, fold), "test")
        missing = 0
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in annotations:
            if any(key not in item for key in required):
                missing += 1
            groups[str(item.get("session_name", ""))].append(item)

        ordering_issues = 0
        sequence_rows = []
        for session_name, items in sorted(groups.items()):
            row_starts = [int(item["window_row_start"]) for item in items]
            centers = [int(item["center_source_frame"]) for item in items]
            natural_is_ordered = row_starts == sorted(row_starts)
            sorted_items = sorted(
                items,
                key=lambda item: (
                    int(item["window_row_start"]),
                    int(item["center_source_frame"]),
                    str(item["frame_dir"]),
                ),
            )
            sorted_row_starts = [int(item["window_row_start"]) for item in sorted_items]
            sorted_centers = [int(item["center_source_frame"]) for item in sorted_items]
            row_deltas = np.diff(sorted_row_starts) if len(sorted_row_starts) > 1 else np.array([])
            center_deltas = np.diff(sorted_centers) if len(sorted_centers) > 1 else np.array([])
            has_duplicate_order = len(set(sorted_row_starts)) != len(sorted_row_starts)
            has_nonmonotonic_center = bool(np.any(center_deltas <= 0)) if len(center_deltas) else False
            if (not natural_is_ordered) or has_duplicate_order or has_nonmonotonic_center:
                ordering_issues += 1
            labels = [int(item["label"]) for item in sorted_items]
            sequence_rows.append(
                {
                    "session_name": session_name,
                    "num_windows": len(items),
                    "natural_order_by_window_row_start": natural_is_ordered,
                    "duplicate_window_row_start": has_duplicate_order,
                    "strictly_increasing_center_source_frame": not has_nonmonotonic_center,
                    "min_window_row_start_delta": None if len(row_deltas) == 0 else int(row_deltas.min()),
                    "max_window_row_start_delta": None if len(row_deltas) == 0 else int(row_deltas.max()),
                    "min_center_source_frame_delta": None if len(center_deltas) == 0 else int(center_deltas.min()),
                    "max_center_source_frame_delta": None if len(center_deltas) == 0 else int(center_deltas.max()),
                    "labels_present": dict(sorted(Counter(labels).items())),
                }
            )

        report[fold] = {
            "num_windows": len(annotations),
            "num_sequences": len(groups),
            "missing_required_fields": missing,
            "ordering_issues": ordering_issues,
            "sequences": sequence_rows,
        }
    return report


def summarize(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for fold in FOLDS:
            score_by_stream = add_single_stream_rows(
                rows=rows,
                work_root=args.work_root,
                data_root=args.data_root,
                fold=fold,
                condition=condition,
            )
            add_fusion_row(
                rows=rows,
                work_root=args.work_root,
                data_root=args.data_root,
                fold=fold,
                condition=condition,
                score_by_stream=score_by_stream,
                write_predictions=not args.no_write_fusion_predictions,
            )

    summary = aggregate_rows(rows)
    temporal = temporal_support_report(args.data_root)
    segmental = None
    if not args.skip_continuous_segmental:
        segmental = evaluate_continuous_segmental(
            work_root=args.work_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
            background=parse_background_labels(args.segmental_background_labels),
            axis=args.segmental_overlap_axis,
        )

    write_csv(args.output_dir / "e1_fold_metrics.csv", rows)
    write_csv(args.output_dir / "e1_mean_sd.csv", summary)
    write_json(
        args.output_dir / "e1_summary.json",
        {
            "fold_metrics": rows,
            "mean_sd": summary,
            "continuous_segmental_metric_readiness": temporal,
            "continuous_segmental_metrics": segmental,
        },
    )
    (args.output_dir / "e1_summary.md").write_text(
        markdown_report(rows, summary, temporal, segmental),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("work_dirs/rerun/e1"),
        help="E1 work directory containing fold/stream result subdirectories.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/radar_v4/rerun/yolo26xpose/pyskl"),
        help="Root containing activity_aligned and continuous_window_w60_s12 pkls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rerun/e1/reports"),
        help="Directory for CSV, JSON, and Markdown reports.",
    )
    parser.add_argument(
        "--no-write-fusion-predictions",
        action="store_true",
        help="Do not write fusion best_pred.pkl/best_eval.json under work_root.",
    )
    parser.add_argument(
        "--skip-continuous-segmental",
        action="store_true",
        help="Skip continuous-window segmental Edit/F1 metrics in the E1 summary.",
    )
    parser.add_argument(
        "--segmental-background-labels",
        nargs="*",
        default=[],
        help="Optional background labels by id or label name for segmental metrics.",
    )
    parser.add_argument(
        "--segmental-overlap-axis",
        choices=["center_source_frame", "sequence_index"],
        default="sequence_index",
        help="Axis used for continuous segment IoU intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args)
    print(f"[DONE] wrote E1 reports under {args.output_dir}")


if __name__ == "__main__":
    main()
