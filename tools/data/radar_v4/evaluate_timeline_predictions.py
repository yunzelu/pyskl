"""Evaluate frame-level radar v4 timeline predictions for one session.

Metrics:
    Acc: framewise accuracy on frames with a valid ground-truth label.
    Edit: normalized segmental edit score, where 100 is best.
    F1@10/25/50: segmental F1 at IoU thresholds 10%, 25%, and 50%.

Example:
    python tools/data/radar_v4/evaluate_timeline_predictions.py ^
        --predictions work_dirs/radar_v4_eval/3-han-laysofa_predictions.csv ^
        --origin-session data/radar_v4/origin/3-han-laysofa ^
        --output-json work_dirs/radar_v4_eval/3-han-laysofa_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LABEL_MAP = Path("tools/data/label_map/radarv4.txt")
DROP_LABELS = {
    "DELETE",
    "END",
    "Kneeling-Stationary",
    "Transition-Kneeling-to-Stand",
}
RENAME_LABELS = {
    "Transition-Sit-to-Laybed": "Transition-Sit-to-LayBed",
    "LayBed-Stationary": "Lying-Stationary",
    "LayFloor-Stationary": "Lying-Stationary",
}
UNKNOWN_LABEL = "__UNKNOWN__"


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_label_map(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"{path} has no labels")
    return labels


def clean_label(label: Any, valid_labels: set[str]) -> str | None:
    if label is None:
        return None

    text = str(label).strip()
    if not text or text in DROP_LABELS:
        return None

    text = RENAME_LABELS.get(text, text)
    if text not in valid_labels:
        return None
    return text


def read_prediction_timeline(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        if "frame_index" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            raise ValueError(f"{path} must contain frame_index and prediction columns")

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"{path} has no prediction rows")

    total_frames = max(safe_int(row.get("frame_index")) for row in rows) + 1
    timeline = [UNKNOWN_LABEL] * total_frames
    for row in rows:
        frame_index = safe_int(row.get("frame_index"), default=-1)
        if 0 <= frame_index < total_frames:
            prediction = str(row.get("prediction", "")).strip() or UNKNOWN_LABEL
            timeline[frame_index] = prediction

    return timeline, rows


def read_frame_label_rows(label_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Frame" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise ValueError(f"{label_path} must contain Frame and Label columns")

        for row_index, row in enumerate(reader):
            rows.append(
                {
                    "row_index": row_index,
                    "frame": safe_int(row.get("Frame"), default=-1),
                    "label": row.get("Label"),
                }
            )

    return sorted(
        (row for row in rows if row["frame"] >= 0),
        key=lambda row: (row["frame"], row["row_index"]),
    )


def build_gt_timeline(
    origin_session: Path,
    total_frames: int,
    valid_labels: set[str],
) -> list[str | None]:
    label_path = origin_session / "frame_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    rows = read_frame_label_rows(label_path)
    timeline: list[str | None] = [None] * total_frames

    for index, row in enumerate(rows):
        start = int(row["frame"])
        if start >= total_frames:
            continue

        end = rows[index + 1]["frame"] - 1 if index + 1 < len(rows) else total_frames - 1
        end = min(end, total_frames - 1)
        if end < start:
            continue

        label = clean_label(row["label"], valid_labels)
        if label is None:
            continue

        for frame_index in range(max(0, start), end + 1):
            timeline[frame_index] = label

    return timeline


def normalize_prediction_label(label: str, valid_labels: set[str]) -> str:
    cleaned = clean_label(label, valid_labels)
    return cleaned if cleaned is not None else UNKNOWN_LABEL


def filtered_eval_sequences(
    gt_timeline: list[str | None],
    pred_timeline: list[str],
    valid_labels: set[str],
) -> tuple[list[str], list[str], list[int]]:
    gt_sequence: list[str] = []
    pred_sequence: list[str] = []
    frame_indices: list[int] = []

    for frame_index, gt_label in enumerate(gt_timeline):
        if gt_label is None:
            continue
        gt_sequence.append(gt_label)
        pred_sequence.append(normalize_prediction_label(pred_timeline[frame_index], valid_labels))
        frame_indices.append(frame_index)

    return gt_sequence, pred_sequence, frame_indices


def collapse_labels(labels: list[str]) -> list[str]:
    collapsed: list[str] = []
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
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def edit_score(gt_labels: list[str], pred_labels: list[str]) -> tuple[float, int]:
    gt_collapsed = collapse_labels(gt_labels)
    pred_collapsed = collapse_labels(pred_labels)
    distance = levenshtein_distance(pred_collapsed, gt_collapsed)
    normalizer = max(len(gt_collapsed), len(pred_collapsed))
    if normalizer == 0:
        return 100.0, 0
    return (1.0 - distance / normalizer) * 100.0, distance


def timeline_to_segments(labels: list[str]) -> list[Segment]:
    if not labels:
        return []

    segments: list[Segment] = []
    current_label = labels[0]
    start = 0

    for index, label in enumerate(labels[1:], start=1):
        if label == current_label:
            continue
        segments.append(Segment(label=current_label, start=start, end=index - 1))
        current_label = label
        start = index

    segments.append(Segment(label=current_label, start=start, end=len(labels) - 1))
    return segments


def segment_iou(first: Segment, second: Segment) -> float:
    intersection = max(0, min(first.end, second.end) - max(first.start, second.start) + 1)
    union = (first.end - first.start + 1) + (second.end - second.start + 1) - intersection
    return intersection / union if union > 0 else 0.0


def f1_at_threshold(gt_labels: list[str], pred_labels: list[str], threshold: float) -> float:
    gt_segments = timeline_to_segments(gt_labels)
    pred_segments = timeline_to_segments(pred_labels)

    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0

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
            true_positive += 1
        else:
            false_positive += 1

    false_negative = len(gt_segments) - len(matched_gt)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall) * 100.0


def framewise_accuracy(gt_labels: list[str], pred_labels: list[str]) -> float:
    if not gt_labels:
        return 0.0
    correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
    return correct / len(gt_labels) * 100.0


def evaluate(
    predictions_path: Path,
    origin_session: Path,
    label_map_path: Path,
) -> dict[str, Any]:
    labels = load_label_map(label_map_path)
    valid_labels = set(labels)
    pred_timeline, prediction_rows = read_prediction_timeline(predictions_path)
    gt_timeline = build_gt_timeline(
        origin_session=origin_session,
        total_frames=len(pred_timeline),
        valid_labels=valid_labels,
    )
    gt_eval, pred_eval, frame_indices = filtered_eval_sequences(
        gt_timeline=gt_timeline,
        pred_timeline=pred_timeline,
        valid_labels=valid_labels,
    )

    edit, raw_edit_distance = edit_score(gt_eval, pred_eval)
    metrics = {
        "Acc": framewise_accuracy(gt_eval, pred_eval),
        "Edit": edit,
        "EditDistance": raw_edit_distance,
        "F1@10": f1_at_threshold(gt_eval, pred_eval, 0.10),
        "F1@25": f1_at_threshold(gt_eval, pred_eval, 0.25),
        "F1@50": f1_at_threshold(gt_eval, pred_eval, 0.50),
    }

    evaluated_range = {
        "start_frame": frame_indices[0] if frame_indices else None,
        "end_frame": frame_indices[-1] if frame_indices else None,
    }
    gt_segments = timeline_to_segments(gt_eval)
    pred_segments = timeline_to_segments(pred_eval)

    return {
        "predictions": str(predictions_path),
        "origin_session": str(origin_session),
        "label_map": str(label_map_path),
        "total_prediction_frames": len(pred_timeline),
        "frames_evaluated": len(gt_eval),
        "evaluated_frame_range": evaluated_range,
        "num_gt_segments": len(gt_segments),
        "num_pred_segments": len(pred_segments),
        "metrics_percent": metrics,
        "note": (
            "Metrics are computed only on frames with valid ground-truth labels. "
            "Edit is normalized segmental edit score, where 100 is best."
        ),
        "prediction_rows": len(prediction_rows),
    }


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one radar v4 frame-level prediction timeline against origin frame_labels.csv."
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Frame-level prediction CSV.")
    parser.add_argument(
        "--origin-session",
        type=Path,
        required=True,
        help="Session folder under data/radar_v4/origin, e.g. data/radar_v4/origin/3-han-laysofa.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-json", type=Path, help="Optional metrics JSON output path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        predictions_path=args.predictions,
        origin_session=args.origin_session,
        label_map_path=args.label_map,
    )

    metrics = result["metrics_percent"]
    print(f"[INFO] Frames evaluated: {result['frames_evaluated']}")
    print(f"Acc:   {metrics['Acc']:.4f}")
    print(f"Edit:  {metrics['Edit']:.4f}")
    print(f"F1@10: {metrics['F1@10']:.4f}")
    print(f"F1@25: {metrics['F1@25']:.4f}")
    print(f"F1@50: {metrics['F1@50']:.4f}")

    if args.output_json:
        write_json(args.output_json, result, overwrite=args.overwrite)
        print(f"[DONE] Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
