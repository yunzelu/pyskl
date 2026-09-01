"""Evaluate continuous-window E1 outputs with segmental metrics.

The evaluation treats each recording as an independent temporal sequence.
Predictions are aligned to annotations by the PYSKL test dataset order:
``PoseDataset`` filters the pkl annotation list by split while preserving the
annotation order, and ``best_pred.pkl`` is written in that same order.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
FOLDS = ["a", "b", "c"]
STREAMS = ["joint", "bone"]
THRESHOLDS = [0.10, 0.25, 0.50]


@dataclass(frozen=True)
class ContinuousCondition:
    key: str
    result_dir: str


CONDITIONS = [
    ContinuousCondition("a2", "a2_activity_checkpoint_on_continuous"),
    ContinuousCondition("b", "b_continuous_window"),
]


@dataclass(frozen=True)
class Segment:
    label: int
    start: float
    end: float


def install_numpy_pickle_compat_aliases() -> None:
    """Allow NumPy 2-generated pkls to load under NumPy 1.x."""

    try:
        importlib.import_module("numpy._core.numeric")
        return
    except ModuleNotFoundError:
        pass

    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias, target in aliases.items():
        try:
            sys.modules.setdefault(alias, importlib.import_module(target))
        except ModuleNotFoundError:
            continue


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def continuous_pkl_path(data_root: Path, fold: str) -> Path:
    return (
        data_root
        / "continuous_window_w60_s12"
        / f"radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"
    )


def split_annotations(pkl_file: Path, split_name: str = "test") -> list[dict[str, Any]]:
    data = load_pickle(pkl_file)
    annotations = data["annotations"]
    identifier = "filename" if "filename" in annotations[0] else "frame_dir"
    split_ids = set(data["split"][split_name])
    return [item for item in annotations if item[identifier] in split_ids]


def load_scores(path: Path) -> np.ndarray:
    scores = np.asarray(load_pickle(path), dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score array in {path}, got {scores.shape}")
    if scores.shape[1] != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} classes in {path}, got {scores.shape[1]}")
    return scores


def to_probabilities(scores: np.ndarray) -> tuple[np.ndarray, str]:
    row_sums = scores.sum(axis=1)
    if (
        np.all(scores >= -1e-6)
        and np.all(scores <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    ):
        return scores, "already_probabilities"

    stable = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(stable)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True), "softmax_applied"


def parse_background_labels(values: list[str]) -> set[int]:
    background: set[int] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.lstrip("-").isdigit():
            background.add(int(text))
            continue
        key = text.lower()
        if key not in LABEL_TO_ID:
            raise ValueError(
                f"Unknown background label {value!r}; use an id or one of {LABELS}"
            )
        background.add(LABEL_TO_ID[key])
    return background


def annotation_order_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(item["window_row_start"]),
        int(item["center_source_frame"]),
        str(item["frame_dir"]),
    )


def grouped_sequences(
    annotations: list[dict[str, Any]],
    probabilities: np.ndarray,
) -> dict[str, list[tuple[dict[str, Any], np.ndarray]]]:
    if len(annotations) != len(probabilities):
        raise ValueError(
            f"Annotation/prediction length mismatch: {len(annotations)} vs {len(probabilities)}"
        )
    groups: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    for item, score in zip(annotations, probabilities):
        groups[str(item["session_name"])].append((item, score))
    return {
        session_name: sorted(items, key=lambda pair: annotation_order_key(pair[0]))
        for session_name, items in sorted(groups.items())
    }


def sequence_axis(items: list[dict[str, Any]], axis: str) -> np.ndarray:
    if axis == "sequence_index":
        return np.arange(len(items), dtype=np.int64)
    if axis == "center_source_frame":
        return np.array([int(item["center_source_frame"]) for item in items], dtype=np.int64)
    raise ValueError(f"Unsupported segment axis: {axis}")


def segments_from_labels(
    labels: np.ndarray,
    positions: np.ndarray,
    background: set[int],
    axis: str,
) -> list[Segment]:
    if len(labels) == 0:
        return []

    segments: list[Segment] = []
    run_start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and int(labels[index]) == int(labels[run_start]):
            continue

        label = int(labels[run_start])
        if label not in background:
            if axis == "sequence_index":
                start = float(run_start)
                end = float(index)
            else:
                start = float(positions[run_start])
                end = float(positions[index - 1] + 1)
            if end <= start:
                end = start + 1.0
            segments.append(Segment(label=label, start=start, end=end))
        run_start = index
    return segments


def levenshtein_distance(first: list[int], second: list[int]) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for i, first_item in enumerate(first, start=1):
        current = [i]
        for j, second_item in enumerate(second, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(first_item != second_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_edit_score(gt_segments: list[Segment], pred_segments: list[Segment]) -> float:
    gt_sequence = [segment.label for segment in gt_segments]
    pred_sequence = [segment.label for segment in pred_segments]
    denominator = max(len(gt_sequence), len(pred_sequence))
    if denominator == 0:
        return 100.0
    distance = levenshtein_distance(gt_sequence, pred_sequence)
    return 100.0 * (1.0 - distance / denominator)


def segment_iou(first: Segment, second: Segment) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = max(first.end, second.end) - min(first.start, second.start)
    if union <= 0:
        return 0.0
    return intersection / union


def f1_counts_for_threshold(
    gt_segments: list[Segment],
    pred_segments: list[Segment],
    threshold: float,
) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred_segment in pred_segments:
        best_index = -1
        best_iou = 0.0
        for gt_index, gt_segment in enumerate(gt_segments):
            if gt_index in matched_gt or pred_segment.label != gt_segment.label:
                continue
            iou = segment_iou(pred_segment, gt_segment)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index >= 0 and best_iou >= threshold:
            tp += 1
            matched_gt.add(best_index)
        else:
            fp += 1

    fn = len(gt_segments) - len(matched_gt)
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return 100.0 * (2 * tp / denominator)


def evaluate_recording(
    items_and_scores: list[tuple[dict[str, Any], np.ndarray]],
    background: set[int],
    axis: str,
) -> dict[str, Any]:
    items = [item for item, _ in items_and_scores]
    scores = np.stack([score for _, score in items_and_scores])
    gt_labels = np.array([int(item["label"]) for item in items], dtype=np.int64)
    pred_labels = np.argmax(scores, axis=1).astype(np.int64)
    positions = sequence_axis(items, axis)

    gt_segments = segments_from_labels(gt_labels, positions, background, axis)
    pred_segments = segments_from_labels(pred_labels, positions, background, axis)
    counts: dict[float, tuple[int, int, int]] = {}
    payload: dict[str, Any] = {
        "num_windows": len(items),
        "num_gt_segments": len(gt_segments),
        "num_pred_segments": len(pred_segments),
        "edit": normalized_edit_score(gt_segments, pred_segments),
    }

    for threshold in THRESHOLDS:
        tp, fp, fn = f1_counts_for_threshold(gt_segments, pred_segments, threshold)
        counts[threshold] = (tp, fp, fn)
        suffix = f"{int(threshold * 100):02d}"
        payload[f"tp_{suffix}"] = tp
        payload[f"fp_{suffix}"] = fp
        payload[f"fn_{suffix}"] = fn
        payload[f"f1_{suffix}"] = f1_from_counts(tp, fp, fn)

    return payload


def result_path(work_root: Path, fold: str, stream: str, condition: ContinuousCondition) -> Path:
    return work_root / f"fold_{fold}" / stream / condition.result_dir / "best_pred.pkl"


def evaluate_fold_stream(
    work_root: Path,
    data_root: Path,
    fold: str,
    condition: ContinuousCondition,
    stream: str,
    probabilities: np.ndarray,
    background: set[int],
    axis: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotations = split_annotations(continuous_pkl_path(data_root, fold), "test")
    groups = grouped_sequences(annotations, probabilities)

    record_rows: list[dict[str, Any]] = []
    edit_scores: list[float] = []
    pooled_counts: dict[float, Counter[str]] = {
        threshold: Counter() for threshold in THRESHOLDS
    }

    for session_name, items_and_scores in groups.items():
        record_metrics = evaluate_recording(items_and_scores, background, axis)
        edit_scores.append(float(record_metrics["edit"]))
        for threshold in THRESHOLDS:
            suffix = f"{int(threshold * 100):02d}"
            pooled_counts[threshold]["tp"] += int(record_metrics[f"tp_{suffix}"])
            pooled_counts[threshold]["fp"] += int(record_metrics[f"fp_{suffix}"])
            pooled_counts[threshold]["fn"] += int(record_metrics[f"fn_{suffix}"])
        record_rows.append(
            {
                "condition": condition.key,
                "fold": fold,
                "stream": stream,
                "session_name": session_name,
                "axis": axis,
                **record_metrics,
            }
        )

    fold_row: dict[str, Any] = {
        "condition": condition.key,
        "fold": fold,
        "stream": stream,
        "axis": axis,
        "num_recordings": len(groups),
        "num_windows": sum(int(row["num_windows"]) for row in record_rows),
        "edit": float(np.mean(edit_scores)) if edit_scores else 0.0,
    }
    for threshold in THRESHOLDS:
        suffix = f"{int(threshold * 100):02d}"
        tp = int(pooled_counts[threshold]["tp"])
        fp = int(pooled_counts[threshold]["fp"])
        fn = int(pooled_counts[threshold]["fn"])
        fold_row[f"tp_{suffix}"] = tp
        fold_row[f"fp_{suffix}"] = fp
        fold_row[f"fn_{suffix}"] = fn
        fold_row[f"f1_{suffix}"] = f1_from_counts(tp, fp, fn)

    return fold_row, record_rows


def aggregate_mean_sd(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["condition"]), str(row["stream"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for condition in [item.key for item in CONDITIONS]:
        for stream in [*STREAMS, "fusion"]:
            rows = grouped[(condition, stream)]
            if len(rows) != len(FOLDS):
                raise ValueError(
                    f"Expected {len(FOLDS)} folds for {condition}/{stream}, got {len(rows)}"
                )
            summary: dict[str, Any] = {
                "condition": condition,
                "stream": stream,
                "folds": len(rows),
            }
            for metric in ["edit", "f1_10", "f1_25", "f1_50"]:
                values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_sd"] = float(np.std(values, ddof=1))
            summary_rows.append(summary)
    return summary_rows


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.2f} +- {sd:.2f}"


def markdown_report(fold_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E1 Continuous Segmental Metrics",
        "",
        "Each test recording is evaluated independently. Segmental F1 pools TP,",
        "FP, and FN counts across recordings within the fold. Edit is normalized",
        "per recording and then averaged across recordings in the fold.",
        "",
        "## Mean +- SD Across Folds",
        "",
        "| Condition | Stream | Edit | F1@10 | F1@25 | F1@50 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {stream} | {edit} | {f10} | {f25} | {f50} |".format(
                condition=str(row["condition"]).upper(),
                stream=row["stream"],
                edit=format_mean_sd(row["edit_mean"], row["edit_sd"]),
                f10=format_mean_sd(row["f1_10_mean"], row["f1_10_sd"]),
                f25=format_mean_sd(row["f1_25_mean"], row["f1_25_sd"]),
                f50=format_mean_sd(row["f1_50_mean"], row["f1_50_sd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| Condition | Fold | Stream | Recordings | Windows | Edit | F1@10 | F1@25 | F1@50 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fold_rows:
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
    return "\n".join(lines) + "\n"


def evaluate_continuous_segmental(
    work_root: Path,
    data_root: Path,
    output_dir: Path,
    background: set[int] | None = None,
    axis: str = "sequence_index",
) -> dict[str, Any]:
    background = set() if background is None else set(background)
    fold_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    score_format_rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for fold in FOLDS:
            stream_probs: dict[str, np.ndarray] = {}
            for stream in STREAMS:
                score_path = result_path(work_root, fold, stream, condition)
                scores = load_scores(score_path)
                probabilities, score_format = to_probabilities(scores)
                stream_probs[stream] = probabilities
                score_format_rows.append(
                    {
                        "condition": condition.key,
                        "fold": fold,
                        "stream": stream,
                        "score_path": str(score_path),
                        "score_format": score_format,
                    }
                )
                fold_row, record_rows = evaluate_fold_stream(
                    work_root=work_root,
                    data_root=data_root,
                    fold=fold,
                    condition=condition,
                    stream=stream,
                    probabilities=probabilities,
                    background=background,
                    axis=axis,
                )
                fold_rows.append(fold_row)
                recording_rows.extend(record_rows)

            fusion_probabilities = 0.5 * (stream_probs["joint"] + stream_probs["bone"])
            fold_row, record_rows = evaluate_fold_stream(
                work_root=work_root,
                data_root=data_root,
                fold=fold,
                condition=condition,
                stream="fusion",
                probabilities=fusion_probabilities,
                background=background,
                axis=axis,
            )
            fold_rows.append(fold_row)
            recording_rows.extend(record_rows)

    summary_rows = aggregate_mean_sd(fold_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "e1_continuous_segmental_fold_metrics.csv", fold_rows)
    write_csv(output_dir / "e1_continuous_segmental_recording_metrics.csv", recording_rows)
    write_csv(output_dir / "e1_continuous_segmental_mean_sd.csv", summary_rows)
    write_json(
        output_dir / "e1_continuous_segmental_summary.json",
        {
            "labels": LABELS,
            "background_label_ids": sorted(background),
            "overlap_axis": axis,
            "fold_metrics": fold_rows,
            "recording_metrics": recording_rows,
            "mean_sd": summary_rows,
            "score_formats": score_format_rows,
            "f1_definition": (
                "For each fold and threshold, TP/FP/FN are summed across "
                "recordings before F1 is computed."
            ),
            "edit_definition": (
                "Normalized Levenshtein similarity is computed per recording "
                "after collapsing consecutive labels, then averaged across "
                "recordings in the fold."
            ),
        },
    )
    (output_dir / "e1_continuous_segmental_summary.md").write_text(
        markdown_report(fold_rows, summary_rows),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "fold_metrics": fold_rows,
        "recording_metrics": recording_rows,
        "mean_sd": summary_rows,
        "score_formats": score_format_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs/rerun/e1"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/radar_v4/rerun/yolo26xpose/pyskl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("rerun/e1/reports"))
    parser.add_argument(
        "--background-labels",
        nargs="*",
        default=[],
        help="Optional background labels by id or label name. Default: none.",
    )
    parser.add_argument(
        "--overlap-axis",
        choices=["center_source_frame", "sequence_index"],
        default="sequence_index",
        help="Axis used for segment IoU intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    background = parse_background_labels(args.background_labels)
    evaluate_continuous_segmental(
        work_root=args.work_root,
        data_root=args.data_root,
        output_dir=args.output_dir,
        background=background,
        axis=args.overlap_axis,
    )
    print(f"[DONE] wrote continuous segmental reports under {args.output_dir}")


if __name__ == "__main__":
    main()
