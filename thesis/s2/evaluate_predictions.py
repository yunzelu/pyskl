"""Evaluate S2 center-window prediction CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

FOLD_STAT_METRICS = [
    "center_acc",
    "center_macro_f1",
    "center_weighted_f1",
    "state_macro_f1",
    "transition_macro_f1",
    "edit",
    "f1_10",
    "f1_25",
    "f1_50",
]

try:
    from .common import (
        DEFAULT_EVAL_DIR,
        LABELS,
        LABEL_TO_ID,
        STRIDE,
        FPS,
        default_confusion_path,
        default_metrics_path,
        default_per_class_path,
        default_prediction_path,
        label_to_group,
        logit_column_name,
        percent,
        prob_column_name,
        protocol_metadata,
        safe_float,
        safe_int,
        write_json,
        write_rows_csv,
    )
except ImportError:
    from common import (
        DEFAULT_EVAL_DIR,
        LABELS,
        LABEL_TO_ID,
        STRIDE,
        FPS,
        default_confusion_path,
        default_metrics_path,
        default_per_class_path,
        default_prediction_path,
        label_to_group,
        logit_column_name,
        percent,
        prob_column_name,
        protocol_metadata,
        safe_float,
        safe_int,
        write_json,
        write_rows_csv,
    )


@dataclass(frozen=True)
class EvalRow:
    method: str
    stream: str
    eta: str
    fold: str
    subject_id: str
    recording_id: str
    start_frame: int
    end_frame: int
    center_frame: int
    center_timestamp: float
    gt_label: str
    gt_group: str
    pred_label: str
    pred_id: int
    confidence: float
    logits: np.ndarray
    probabilities: np.ndarray

    @property
    def valid_gt(self) -> bool:
        return self.gt_label in LABEL_TO_ID

    @property
    def gt_id(self) -> int:
        return LABEL_TO_ID[self.gt_label]

    @property
    def correct(self) -> bool:
        return self.valid_gt and self.gt_label == self.pred_label


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int


def read_predictions(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        required = [
            "method",
            "stream",
            "fold",
            "subject_id",
            "recording_id",
            "start_frame",
            "end_frame",
            "center_frame",
            "center_timestamp",
            "ground_truth_center_label",
            "ground_truth_group",
        ]
        missing = [name for name in required if name not in reader.fieldnames]
        missing += [name for name in [logit_column_name(label) for label in LABELS] if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing[:10]}")

        for item in reader:
            logits = np.asarray(
                [safe_float(item.get(logit_column_name(label))) for label in LABELS],
                dtype=np.float32,
            )
            if all(prob_column_name(label) in item for label in LABELS):
                probabilities = np.asarray(
                    [safe_float(item.get(prob_column_name(label))) for label in LABELS],
                    dtype=np.float32,
                )
            else:
                shifted = logits - np.max(logits)
                exp_values = np.exp(shifted)
                probabilities = (exp_values / np.sum(exp_values)).astype(np.float32)

            pred_id = int(np.argmax(logits))
            pred_label = LABELS[pred_id]
            rows.append(
                EvalRow(
                    method=str(item.get("method") or ""),
                    stream=str(item.get("stream") or ""),
                    eta=str(item.get("eta") or ""),
                    fold=str(item.get("fold") or ""),
                    subject_id=str(item.get("subject_id") or ""),
                    recording_id=str(item.get("recording_id") or ""),
                    start_frame=safe_int(item.get("start_frame"), -1),
                    end_frame=safe_int(item.get("end_frame"), -1),
                    center_frame=safe_int(item.get("center_frame"), -1),
                    center_timestamp=safe_float(item.get("center_timestamp")),
                    gt_label=str(item.get("ground_truth_center_label") or ""),
                    gt_group=str(item.get("ground_truth_group") or ""),
                    pred_label=pred_label,
                    pred_id=pred_id,
                    confidence=float(probabilities[pred_id]),
                    logits=logits,
                    probabilities=probabilities,
                )
            )

    if not rows:
        raise ValueError(f"{path} has no prediction rows")
    return rows


def f1_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def confusion_matrix(rows: list[EvalRow]) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for row in rows:
        if row.valid_gt:
            matrix[row.gt_id, row.pred_id] += 1
    return matrix


def per_class_rows(rows: list[EvalRow]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.valid_gt]
    output = []
    for class_id, label in enumerate(LABELS):
        tp = sum(row.gt_label == label and row.pred_id == class_id for row in valid)
        fp = sum(row.gt_label != label and row.pred_id == class_id for row in valid)
        fn = sum(row.gt_label == label and row.pred_id != class_id for row in valid)
        precision, recall, f1 = f1_counts(tp, fp, fn)
        support = sum(row.gt_label == label for row in valid)
        predicted = sum(row.pred_id == class_id for row in valid)
        output.append(
            {
                "class_id": class_id,
                "label": label,
                "group": group_for_label(label),
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_percent": percent(precision),
                "recall_percent": percent(recall),
                "f1_percent": percent(f1),
            }
        )
    return output


def group_for_label(label: str) -> str:
    group = label_to_group(label)
    if group in {"stationary", "walking"}:
        return "state"
    if group == "transition":
        return "transition"
    return group


def macro_f1_percent(class_rows: list[dict[str, Any]], group: str | None = None) -> float:
    active = []
    for row in class_rows:
        if group is not None and row["group"] != group:
            continue
        if int(row["support"]) > 0 or int(row["predicted"]) > 0:
            active.append(float(row["f1_percent"]))
    return sum(active) / len(active) if active else 0.0


def weighted_f1_percent(class_rows: list[dict[str, Any]]) -> float:
    total = sum(int(row["support"]) for row in class_rows)
    if total <= 0:
        return 0.0
    return sum(float(row["f1_percent"]) * int(row["support"]) for row in class_rows) / total


def collapse_labels(labels: list[str]) -> list[str]:
    collapsed = []
    previous = None
    for label in labels:
        if label != previous:
            collapsed.append(label)
            previous = label
    return collapsed


def levenshtein_distance(first: list[str], second: list[str]) -> int:
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, first_label in enumerate(first, start=1):
        current = [i]
        for j, second_label in enumerate(second, start=1):
            cost = 0 if first_label == second_label else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def rows_to_segments(rows: list[EvalRow], which: str) -> list[Segment]:
    segments: list[Segment] = []
    current_label: str | None = None
    current_start: int | None = None

    def close(end_index: int) -> None:
        nonlocal current_label, current_start
        if current_label is not None and current_start is not None:
            segments.append(Segment(current_label, current_start, end_index))
        current_label = None
        current_start = None

    for index, row in enumerate(rows):
        label = None
        if row.valid_gt:
            label = row.gt_label if which == "gt" else row.pred_label
        if label == current_label:
            continue
        close(index - 1)
        if label is not None:
            current_label = label
            current_start = index
    close(len(rows) - 1)
    return segments


def edit_score(gt_segments: list[Segment], pred_segments: list[Segment]) -> float:
    gt_labels = collapse_labels([segment.label for segment in gt_segments])
    pred_labels = collapse_labels([segment.label for segment in pred_segments])
    normalizer = max(len(gt_labels), len(pred_labels))
    if normalizer == 0:
        return 0.0
    return (1.0 - levenshtein_distance(pred_labels, gt_labels) / normalizer) * 100.0


def segment_iou(first: Segment, second: Segment) -> float:
    intersection = max(0, min(first.end, second.end) - max(first.start, second.start) + 1)
    union = (first.end - first.start + 1) + (second.end - second.start + 1) - intersection
    return intersection / union if union > 0 else 0.0


def segment_counts(
    gt_segments: list[Segment],
    pred_segments: list[Segment],
    threshold: float,
) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_segment in pred_segments:
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt_segment in enumerate(gt_segments):
            if gt_index in matched_gt or pred_segment.label != gt_segment.label:
                continue
            iou = segment_iou(pred_segment, gt_segment)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index >= 0 and best_iou >= threshold:
            matched_gt.add(best_gt_index)
            tp += 1
        else:
            fp += 1
    fn = len(gt_segments) - len(matched_gt)
    return tp, fp, fn


def grouped_recordings(rows: list[EvalRow]) -> list[list[EvalRow]]:
    groups: dict[tuple[str, str], list[EvalRow]] = defaultdict(list)
    for row in rows:
        groups[(row.fold, row.recording_id)].append(row)
    return [
        sorted(groups[key], key=lambda item: (item.center_timestamp, item.start_frame))
        for key in sorted(groups)
    ]


def segmental_metrics(rows: list[EvalRow]) -> dict[str, float]:
    edit_scores: list[float] = []
    totals = {0.10: [0, 0, 0], 0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    for recording_rows in grouped_recordings(rows):
        gt_segments = rows_to_segments(recording_rows, "gt")
        pred_segments = rows_to_segments(recording_rows, "pred")
        if not gt_segments and not pred_segments:
            continue
        edit_scores.append(edit_score(gt_segments, pred_segments))
        for threshold in totals:
            counts = segment_counts(gt_segments, pred_segments, threshold)
            for index, value in enumerate(counts):
                totals[threshold][index] += value

    metrics = {"edit": sum(edit_scores) / len(edit_scores) if edit_scores else 0.0}
    for threshold, (tp, fp, fn) in totals.items():
        _, _, f1 = f1_counts(tp, fp, fn)
        metrics[f"f1_{int(threshold * 100)}"] = percent(f1)
    return metrics


def metrics_for_rows(rows: list[EvalRow]) -> dict[str, Any]:
    valid = [row for row in rows if row.valid_gt]
    class_rows = per_class_rows(rows)
    correct = sum(row.correct for row in valid)
    seq = segmental_metrics(rows)
    return {
        "total_windows": len(rows),
        "evaluated_windows": len(valid),
        "skipped_windows": len(rows) - len(valid),
        "center_acc": percent(correct / len(valid)) if valid else 0.0,
        "center_macro_f1": macro_f1_percent(class_rows),
        "center_weighted_f1": weighted_f1_percent(class_rows),
        "state_macro_f1": macro_f1_percent(class_rows, group="state"),
        "transition_macro_f1": macro_f1_percent(class_rows, group="transition"),
        "edit": seq["edit"],
        "f1_10": seq["f1_10"],
        "f1_25": seq["f1_25"],
        "f1_50": seq["f1_50"],
    }


def scoped_rows(rows: list[EvalRow]) -> list[tuple[str, str, str, list[EvalRow]]]:
    scopes: list[tuple[str, str, str, list[EvalRow]]] = [("overall", "", "", rows)]
    by_fold: dict[str, list[EvalRow]] = defaultdict(list)
    by_recording: dict[tuple[str, str], list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_fold[row.fold].append(row)
        by_recording[(row.fold, row.recording_id)].append(row)
    for fold in sorted(by_fold):
        scopes.append(("fold", fold, "", by_fold[fold]))
    for fold, recording_id in sorted(by_recording):
        scopes.append(("recording", fold, recording_id, by_recording[(fold, recording_id)]))
    return scopes


def fold_metric_stats(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    fold_rows = [row for row in scopes if row["scope"] == "fold"]
    sd_ddof = 1 if len(fold_rows) > 1 else 0
    stats: dict[str, Any] = {
        "scope": "fold",
        "fold_count": len(fold_rows),
        "folds": [str(row["fold"]) for row in fold_rows],
        "sd_ddof": sd_ddof,
        "mean": {},
        "sd": {},
    }
    for metric in FOLD_STAT_METRICS:
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=np.float64)
        stats["mean"][metric] = float(np.mean(values)) if values.size else 0.0
        stats["sd"][metric] = float(np.std(values, ddof=sd_ddof)) if values.size else 0.0
    return stats


def write_per_class(path: Path, method: str, rows: list[EvalRow], overwrite: bool) -> None:
    fieldnames = [
        "method",
        "scope",
        "fold",
        "recording_id",
        "class_id",
        "label",
        "group",
        "support",
        "predicted",
        "tp",
        "fp",
        "fn",
        "precision_percent",
        "recall_percent",
        "f1_percent",
    ]
    output = []
    for scope, fold, recording_id, scope_rows in scoped_rows(rows):
        for row in per_class_rows(scope_rows):
            output.append(
                {
                    "method": method,
                    "scope": scope,
                    "fold": fold,
                    "recording_id": recording_id,
                    **{
                        key: f"{value:.6f}" if isinstance(value, float) else value
                        for key, value in row.items()
                    },
                }
            )
    write_rows_csv(path, fieldnames, output, overwrite)


def write_confusion(path: Path, method: str, rows: list[EvalRow], overwrite: bool) -> None:
    matrix = confusion_matrix(rows)
    fieldnames = ["method", "true_label", *[f"pred_{label}" for label in LABELS]]
    output = []
    for row_index, label in enumerate(LABELS):
        output.append(
            {
                "method": method,
                "true_label": label,
                **{
                    f"pred_{pred_label}": int(matrix[row_index, pred_index])
                    for pred_index, pred_label in enumerate(LABELS)
                },
            }
        )
    write_rows_csv(path, fieldnames, output, overwrite)


def evaluate_predictions(
    predictions: Path,
    method: str,
    metrics_path: Path,
    per_class_path: Path,
    confusion_path: Path,
    summary_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    rows = read_predictions(predictions)
    scopes = [
        {
            "scope": scope,
            "fold": fold,
            "recording_id": recording_id,
            **metrics_for_rows(scope_rows),
        }
        for scope, fold, recording_id, scope_rows in scoped_rows(rows)
    ]
    overall = scopes[0]
    fold_stats = fold_metric_stats(scopes)
    result = {
        "experiment": "S2",
        "stage": "evaluation",
        "method": method,
        "predictions": str(predictions),
        "labels": LABELS,
        "protocol": protocol_metadata(),
        "temporal_resolution_seconds": STRIDE / FPS,
        "summary": scopes,
        "overall": overall,
        "fold_summary": fold_stats,
    }
    write_json(metrics_path, result, overwrite)
    write_per_class(per_class_path, method, rows, overwrite)
    write_confusion(confusion_path, method, rows, overwrite)
    summary_fieldnames = [
        "method",
        *SUMMARY_METRICS,
        "fold_count",
        "fold_sd_ddof",
        *[f"{metric}_fold_mean" for metric in SUMMARY_METRICS],
        *[f"{metric}_fold_sd" for metric in SUMMARY_METRICS],
    ]
    summary_row = {
        "method": method,
        "fold_count": fold_stats["fold_count"],
        "fold_sd_ddof": fold_stats["sd_ddof"],
    }
    summary_row.update({metric: f"{overall[metric]:.6f}" for metric in SUMMARY_METRICS})
    summary_row.update(
        {
            f"{metric}_fold_mean": f"{fold_stats['mean'][metric]:.6f}"
            for metric in SUMMARY_METRICS
        }
    )
    summary_row.update(
        {
            f"{metric}_fold_sd": f"{fold_stats['sd'][metric]:.6f}"
            for metric in SUMMARY_METRICS
        }
    )
    write_rows_csv(
        summary_path,
        summary_fieldnames,
        [summary_row],
        overwrite,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an S2 prediction CSV.")
    parser.add_argument("--method", choices=["A", "B", "C"], required=True)
    parser.add_argument("--stream", choices=["joint", "limb", "fusion"], default="joint")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--per-class", type=Path)
    parser.add_argument("--confusion", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = args.predictions or default_prediction_path(args.method, args.stream)
    suffix = f"_{args.stream}" if args.stream != "joint" else ""
    metrics_path = args.metrics or args.output_dir / f"metrics_{args.method}{suffix}.json"
    per_class_path = args.per_class or args.output_dir / f"per_class_{args.method}{suffix}.csv"
    confusion_path = args.confusion or args.output_dir / f"confusion_matrix_{args.method}{suffix}.csv"
    summary_path = args.summary or args.output_dir / f"metrics_{args.method}{suffix}_summary.csv"

    result = evaluate_predictions(
        predictions=predictions,
        method=args.method,
        metrics_path=metrics_path,
        per_class_path=per_class_path,
        confusion_path=confusion_path,
        summary_path=summary_path,
        overwrite=args.overwrite,
    )
    overall = result["overall"]
    print(f"[DONE] wrote metrics to {metrics_path}")
    print(
        f"{args.method}: acc={overall['center_acc']:.4f}, "
        f"macro-F1={overall['center_macro_f1']:.4f}, "
        f"state-F1={overall['state_macro_f1']:.4f}, "
        f"transition-F1={overall['transition_macro_f1']:.4f}, "
        f"F1@50={overall['f1_50']:.4f}"
    )
    fold_summary = result["fold_summary"]
    fold_mean = fold_summary["mean"]
    fold_sd = fold_summary["sd"]
    print(
        f"{args.method} fold mean (n={fold_summary['fold_count']}): "
        f"acc={fold_mean['center_acc']:.4f}, "
        f"macro-F1={fold_mean['center_macro_f1']:.4f}, "
        f"state-F1={fold_mean['state_macro_f1']:.4f}, "
        f"transition-F1={fold_mean['transition_macro_f1']:.4f}, "
        f"F1@50={fold_mean['f1_50']:.4f}"
    )
    print(
        f"{args.method} fold SD (ddof={fold_summary['sd_ddof']}): "
        f"acc={fold_sd['center_acc']:.4f}, "
        f"macro-F1={fold_sd['center_macro_f1']:.4f}, "
        f"state-F1={fold_sd['state_macro_f1']:.4f}, "
        f"transition-F1={fold_sd['transition_macro_f1']:.4f}, "
        f"F1@50={fold_sd['f1_50']:.4f}"
    )


if __name__ == "__main__":
    main()
