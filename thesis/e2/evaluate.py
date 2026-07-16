"""E2 evaluation stage for any compatible center-window logit/score CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SCORE_DIR,
        LABELS,
        ScoreRow,
        Segment,
        percent,
        protocol_metadata,
        read_score_csv,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SCORE_DIR,
        LABELS,
        ScoreRow,
        Segment,
        percent,
        protocol_metadata,
        read_score_csv,
        write_json,
    )


def f1_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def per_class_metrics(rows: list[ScoreRow]) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if row.valid_gt]
    metrics: list[dict[str, Any]] = []

    for class_id, label in enumerate(LABELS):
        tp = sum(row.gt_label == label and row.pred_id == class_id for row in valid_rows)
        fp = sum(row.gt_label != label and row.pred_id == class_id for row in valid_rows)
        fn = sum(row.gt_label == label and row.pred_id != class_id for row in valid_rows)
        precision, recall, f1 = f1_counts(tp, fp, fn)
        support = sum(row.gt_label == label for row in valid_rows)
        predicted = sum(row.pred_id == class_id for row in valid_rows)
        metrics.append(
            {
                "class_id": class_id,
                "label": label,
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

    return metrics


def macro_f1_percent(class_rows: list[dict[str, Any]]) -> float:
    active = [
        row
        for row in class_rows
        if int(row["support"]) > 0 or int(row["predicted"]) > 0
    ]
    if not active:
        return 0.0
    return sum(float(row["f1_percent"]) for row in active) / len(active)


def transition_f1_percent(rows: list[ScoreRow]) -> float:
    valid_rows = [row for row in rows if row.valid_gt]
    tp = sum(row.gt_group == "transition" and row.pred_group == "transition" for row in valid_rows)
    fp = sum(row.gt_group != "transition" and row.pred_group == "transition" for row in valid_rows)
    fn = sum(row.gt_group == "transition" and row.pred_group != "transition" for row in valid_rows)
    _, _, f1 = f1_counts(tp, fp, fn)
    return percent(f1)


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


def edit_score_from_segments(gt_segments: list[Segment], pred_segments: list[Segment]) -> float:
    gt_labels = collapse_labels([segment.label for segment in gt_segments])
    pred_labels = collapse_labels([segment.label for segment in pred_segments])
    normalizer = max(len(gt_labels), len(pred_labels))
    if normalizer == 0:
        return 0.0
    distance = levenshtein_distance(pred_labels, gt_labels)
    return (1.0 - distance / normalizer) * 100.0


def rows_to_segments(rows: list[ScoreRow], which: str) -> list[Segment]:
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


def group_rows_by_session(rows: list[ScoreRow]) -> list[list[ScoreRow]]:
    grouped: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.fold, row.session)].append(row)

    sessions = []
    for key in sorted(grouped):
        sessions.append(sorted(grouped[key], key=lambda item: item.window_start))
    return sessions


def sequence_metrics(rows: list[ScoreRow]) -> dict[str, float]:
    edit_scores: list[float] = []
    totals = {
        0.10: [0, 0, 0],
        0.25: [0, 0, 0],
        0.50: [0, 0, 0],
    }

    for session_rows in group_rows_by_session(rows):
        gt_segments = rows_to_segments(session_rows, "gt")
        pred_segments = rows_to_segments(session_rows, "pred")
        if not gt_segments and not pred_segments:
            continue

        edit_scores.append(edit_score_from_segments(gt_segments, pred_segments))
        for threshold in totals:
            counts = segment_counts(gt_segments, pred_segments, threshold)
            for index, count in enumerate(counts):
                totals[threshold][index] += count

    metrics = {
        "Edit": sum(edit_scores) / len(edit_scores) if edit_scores else 0.0,
    }
    for threshold, (tp, fp, fn) in totals.items():
        _, _, f1 = f1_counts(tp, fp, fn)
        metrics[f"F1@{int(threshold * 100)}"] = percent(f1)
    return metrics


def metrics_for_rows(rows: list[ScoreRow]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.valid_gt]
    correct = sum(row.correct for row in evaluated)
    class_rows = per_class_metrics(rows)
    seq = sequence_metrics(rows)

    return {
        "total_windows": len(rows),
        "evaluated_windows": len(evaluated),
        "skipped_windows": len(rows) - len(evaluated),
        "center_time_accuracy_percent": percent(correct / len(evaluated)) if evaluated else 0.0,
        "macro_f1_percent": macro_f1_percent(class_rows),
        "transition_class_f1_percent": transition_f1_percent(rows),
        "Edit": seq["Edit"],
        "F1@10": seq["F1@10"],
        "F1@25": seq["F1@25"],
        "F1@50": seq["F1@50"],
    }


def metric_row(
    scope: str,
    fold: str,
    subject: str,
    session: str,
    rows: list[ScoreRow],
) -> dict[str, Any]:
    return {
        "scope": scope,
        "fold": fold,
        "test_subject": subject,
        "session": session,
        **metrics_for_rows(rows),
    }


def scoped_rows(rows: list[ScoreRow]) -> list[tuple[str, str, str, str, list[ScoreRow]]]:
    grouped_by_fold: dict[str, list[ScoreRow]] = defaultdict(list)
    grouped_by_session: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped_by_fold[row.fold].append(row)
        grouped_by_session[(row.fold, row.session)].append(row)

    scopes: list[tuple[str, str, str, str, list[ScoreRow]]] = [("overall", "", "", "", rows)]
    for fold in sorted(grouped_by_fold):
        fold_rows = grouped_by_fold[fold]
        subject = fold_rows[0].test_subject if fold_rows else ""
        scopes.append(("fold", fold, subject, "", fold_rows))

    for key in sorted(grouped_by_session):
        fold, session = key
        session_rows = grouped_by_session[key]
        subject = session_rows[0].test_subject if session_rows else ""
        scopes.append(("session", fold, subject, session, session_rows))

    return scopes


def write_summary_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "scope",
        "fold",
        "test_subject",
        "session",
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
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_per_class_csv(
    path: Path,
    scopes: list[tuple[str, str, str, str, list[ScoreRow]]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "scope",
        "fold",
        "test_subject",
        "session",
        "class_id",
        "label",
        "support",
        "predicted",
        "tp",
        "fp",
        "fn",
        "precision_percent",
        "recall_percent",
        "f1_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for scope, fold, subject, session, rows in scopes:
            for class_row in per_class_metrics(rows):
                writer.writerow(
                    {
                        "scope": scope,
                        "fold": fold,
                        "test_subject": subject,
                        "session": session,
                        **{
                            key: f"{value:.6f}" if isinstance(value, float) else value
                            for key, value in class_row.items()
                        },
                    }
                )


def evaluate_score_file(
    scores_path: Path,
    output_dir: Path,
    name: str,
    overwrite: bool,
) -> dict[str, Any]:
    rows = read_score_csv(scores_path)
    scopes = scoped_rows(rows)
    summary_rows = [
        metric_row(scope, fold, subject, session, scope_rows)
        for scope, fold, subject, session, scope_rows in scopes
    ]

    write_summary_csv(output_dir / f"{name}_summary.csv", summary_rows, overwrite)
    write_per_class_csv(output_dir / f"{name}_per_class_f1.csv", scopes, overwrite)

    result = {
        "experiment": "E2",
        "stage": "evaluation",
        "scores": str(scores_path),
        "source": rows[0].source,
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "summary": summary_rows,
    }
    write_json(output_dir / f"{name}_metrics.json", result, overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an E2 window-score CSV.")
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORE_DIR / "e2_fusion_joint_limb_logits.csv",
        help="Input score/logit CSV. Default: fused joint+limb logits.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eval")
    parser.add_argument("--name", help="Output filename prefix. Default: score CSV stem.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or args.scores.stem
    result = evaluate_score_file(
        scores_path=args.scores,
        output_dir=args.output_dir,
        name=name,
        overwrite=args.overwrite,
    )
    overall = result["summary"][0]
    print(f"[DONE] wrote E2 evaluation reports to {args.output_dir}")
    print(f"Center-time accuracy: {overall['center_time_accuracy_percent']:.4f}")
    print(f"Macro-F1: {overall['macro_f1_percent']:.4f}")
    print(f"Transition-class F1: {overall['transition_class_f1_percent']:.4f}")
    print(
        "Sequence metrics: "
        f"Edit={overall['Edit']:.4f}, "
        f"F1@10={overall['F1@10']:.4f}, "
        f"F1@25={overall['F1@25']:.4f}, "
        f"F1@50={overall['F1@50']:.4f}"
    )


if __name__ == "__main__":
    main()
