"""Fuse S2 joint and limb prediction CSVs by weighted logit averaging."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .common import (
        LABELS,
        default_prediction_path,
        logit_column_name,
        prob_column_name,
        score_column_name,
        softmax,
        write_json,
    )
    from .infer import prediction_fieldnames, write_prediction_csv
except ImportError:
    from common import (
        LABELS,
        default_prediction_path,
        logit_column_name,
        prob_column_name,
        score_column_name,
        softmax,
        write_json,
    )
    from infer import prediction_fieldnames, write_prediction_csv


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [field for field in prediction_fieldnames() if field not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing[:10]}")
    return rows


def row_key(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    return (
        row["fold"],
        row["recording_id"],
        int(float(row["start_frame"])),
        int(float(row["end_frame"])),
        int(float(row["center_frame"])),
    )


def logits_from_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[logit_column_name(label)]) for label in LABELS], dtype=np.float32)


def fuse_rows(
    joint_rows: list[dict[str, str]],
    limb_rows: list[dict[str, str]],
    method: str,
    joint_weight: float,
    limb_weight: float,
) -> list[dict[str, str]]:
    if joint_weight < 0 or limb_weight < 0:
        raise ValueError("Fusion weights must be non-negative")
    total_weight = joint_weight + limb_weight
    if total_weight <= 0:
        raise ValueError("At least one fusion weight must be positive")

    joint_map = {row_key(row): row for row in joint_rows}
    limb_map = {row_key(row): row for row in limb_rows}
    if set(joint_map) != set(limb_map):
        raise ValueError(
            "Joint and limb prediction files are not aligned: "
            f"missing_limb={len(set(joint_map) - set(limb_map))}, "
            f"missing_joint={len(set(limb_map) - set(joint_map))}"
        )

    output = []
    for key in sorted(joint_map):
        joint = joint_map[key]
        limb = limb_map[key]
        logits = (joint_weight * logits_from_row(joint) + limb_weight * logits_from_row(limb)) / total_weight
        probabilities = softmax(logits)
        pred_id = int(np.argmax(logits))
        row = dict(joint)
        row["model_variant"] = method
        row["method"] = method
        row["stream"] = "fusion"
        row["predicted_label"] = LABELS[pred_id]
        row["predicted_id"] = str(pred_id)
        row["confidence"] = f"{float(probabilities[pred_id]):.8f}"
        row["correct"] = int(bool(row["ground_truth_center_label"]) and row["ground_truth_center_label"] == LABELS[pred_id])
        row["checkpoint"] = f"joint={joint.get('checkpoint', '')};limb={limb.get('checkpoint', '')}"
        row["config"] = f"joint={joint.get('config', '')};limb={limb.get('config', '')}"
        for label, value in zip(LABELS, logits):
            row[logit_column_name(label)] = f"{float(value):.8f}"
            row[score_column_name(label)] = f"{float(value):.8f}"
        for label, value in zip(LABELS, probabilities):
            row[prob_column_name(label)] = f"{float(value):.8f}"
        output.append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse S2 joint and limb predictions.")
    parser.add_argument("--method", choices=["A", "B", "C"], required=True)
    parser.add_argument("--joint", type=Path)
    parser.add_argument("--limb", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--limb-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joint_path = args.joint or default_prediction_path(args.method, "joint")
    limb_path = args.limb or default_prediction_path(args.method, "limb")
    output = args.output or default_prediction_path(args.method, "fusion")
    rows = fuse_rows(
        read_dict_rows(joint_path),
        read_dict_rows(limb_path),
        method=args.method,
        joint_weight=args.joint_weight,
        limb_weight=args.limb_weight,
    )
    write_prediction_csv(output, rows, overwrite=args.overwrite)
    write_json(
        output.with_name(f"{output.stem}_metadata.json"),
        {
            "experiment": "S2",
            "stage": "joint_limb_fusion",
            "method": args.method,
            "joint_predictions": str(joint_path),
            "limb_predictions": str(limb_path),
            "output": str(output),
            "joint_weight": args.joint_weight,
            "limb_weight": args.limb_weight,
            "rows": len(rows),
        },
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote {len(rows)} fused rows to {output}")


if __name__ == "__main__":
    main()
