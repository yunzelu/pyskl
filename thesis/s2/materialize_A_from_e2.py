"""Materialize Method A S2 predictions from an existing E2 score CSV.

This is a compute-saving helper for local reproducibility. If the E2 source
contains probabilities instead of logits, the output uses log probabilities in
the logit columns so argmax metrics are unchanged, and records that provenance
in metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import (
        LABELS,
        METHOD_A,
        default_prediction_path,
        logit_column_name,
        prob_column_name,
        protocol_metadata,
        score_column_name,
        softmax,
        write_json,
    )
    from .infer import write_prediction_csv
except ImportError:
    from common import (
        LABELS,
        METHOD_A,
        default_prediction_path,
        logit_column_name,
        prob_column_name,
        protocol_metadata,
        score_column_name,
        softmax,
        write_json,
    )
    from infer import write_prediction_csv

from thesis.e2.common import read_score_csv  # noqa: E402


def rows_from_e2(e2_scores: Path, stream: str, folds: set[str] | None) -> tuple[list[dict], str]:
    rows = []
    e2_rows = read_score_csv(e2_scores)
    source_score_type = e2_rows[0].score_type
    for row in e2_rows:
        if folds is not None and row.fold not in folds:
            continue
        scores = np.asarray(row.scores, dtype=np.float32)
        if row.score_type == "prob":
            probabilities = np.clip(scores, 1e-12, 1.0)
            logits = np.log(probabilities).astype(np.float32)
        elif row.score_type == "logit":
            logits = scores
            probabilities = softmax(logits)
        else:
            raise ValueError(f"Unsupported E2 score_type {row.score_type!r}")

        pred_id = int(np.argmax(logits))
        item = {
            "model_variant": METHOD_A,
            "method": METHOD_A,
            "stream": stream,
            "eta": "",
            "fold": row.fold,
            "subject_id": row.test_subject,
            "validation_subject": "",
            "test_subject": row.test_subject,
            "recording_id": row.session,
            "jsonl_path": str(row.jsonl_path),
            "start_frame": row.window_start,
            "end_frame": row.window_end,
            "center_frame": row.center_frame,
            "center_timestamp": f"{row.center_time_sec:.6f}",
            "raw_ground_truth_center_label": row.raw_gt_label,
            "ground_truth_center_label": row.gt_label,
            "ground_truth_group": row.gt_group,
            "predicted_label": LABELS[pred_id],
            "predicted_id": pred_id,
            "confidence": f"{float(probabilities[pred_id]):.8f}",
            "correct": int(row.valid_gt and row.gt_label == LABELS[pred_id]),
            "valid_detection_frames": row.valid_detection_frames,
            "selected_detection_center": int(row.selected_detection_center),
            "checkpoint": "",
            "config": "",
        }
        for label, value in zip(LABELS, logits):
            item[logit_column_name(label)] = f"{float(value):.8f}"
            item[score_column_name(label)] = f"{float(value):.8f}"
        for label, value in zip(LABELS, probabilities):
            item[prob_column_name(label)] = f"{float(value):.8f}"
        rows.append(item)
    if not rows:
        raise ValueError(f"No E2 rows matched fold filter {sorted(folds) if folds else 'all'}")
    return rows, source_score_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert existing E2 scores into S2 Method A predictions.")
    parser.add_argument(
        "--e2-scores",
        type=Path,
        default=Path("work_dirs/thesis/e2/scores/e2_joint_scores.csv"),
    )
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--folds", nargs="+", help="Fold ids to include. Default: all E2 folds.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_prediction_path("A", args.stream)
    folds = None
    if args.folds:
        folds = {fold.lower().replace("fold_", "") for fold in args.folds}
    rows, source_score_type = rows_from_e2(args.e2_scores, args.stream, folds)
    write_prediction_csv(output, rows, overwrite=args.overwrite)
    write_json(
        output.with_name(f"{output.stem}_metadata.json"),
        {
            "experiment": "S2",
            "stage": "materialize_method_A_from_e2",
            "source_e2_scores": str(args.e2_scores),
            "source_score_type": source_score_type,
            "logit_column_note": (
                "raw logits" if source_score_type == "logit"
                else "log probabilities derived from E2 probabilities; argmax metrics are unchanged"
            ),
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "prediction_csv": str(output),
            "rows": len(rows),
        },
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote Method A predictions to {output}")


if __name__ == "__main__":
    main()
