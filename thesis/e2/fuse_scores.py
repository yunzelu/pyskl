"""E2 score fusion stage for joint and limb PoseC3D streams."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import (
        DEFAULT_SCORE_DIR,
        LABELS,
        ScoreRow,
        protocol_metadata,
        read_score_csv,
        score_key,
        write_json,
        write_score_csv,
    )
except ImportError:
    from common import (
        DEFAULT_SCORE_DIR,
        LABELS,
        ScoreRow,
        protocol_metadata,
        read_score_csv,
        score_key,
        write_json,
        write_score_csv,
    )


def aligned_score_maps(
    joint_rows: list[ScoreRow],
    limb_rows: list[ScoreRow],
) -> tuple[dict[tuple[str, str, str, int, int, int], ScoreRow], dict[tuple[str, str, str, int, int, int], ScoreRow]]:
    joint_map = {score_key(row): row for row in joint_rows}
    limb_map = {score_key(row): row for row in limb_rows}

    joint_keys = set(joint_map)
    limb_keys = set(limb_map)
    missing_limb = sorted(joint_keys - limb_keys)
    missing_joint = sorted(limb_keys - joint_keys)
    if missing_limb or missing_joint:
        raise ValueError(
            "Joint/limb score files are not aligned: "
            f"missing_limb={len(missing_limb)}, missing_joint={len(missing_joint)}"
        )
    return joint_map, limb_map


def fuse_rows(
    joint_rows: list[ScoreRow],
    limb_rows: list[ScoreRow],
    joint_weight: float,
    limb_weight: float,
) -> list[ScoreRow]:
    if joint_weight < 0 or limb_weight < 0:
        raise ValueError("Fusion weights must be non-negative")
    total_weight = joint_weight + limb_weight
    if total_weight <= 0:
        raise ValueError("At least one fusion weight must be positive")

    joint_map, limb_map = aligned_score_maps(joint_rows, limb_rows)
    score_types = {row.score_type for row in joint_rows + limb_rows}
    if len(score_types) != 1:
        raise ValueError(f"Cannot fuse mixed score types: {sorted(score_types)}")
    score_type = score_types.pop()

    source = f"fusion_joint_limb_j{joint_weight:g}_l{limb_weight:g}"
    fused: list[ScoreRow] = []

    for key in sorted(joint_map):
        joint = joint_map[key]
        limb = limb_map[key]
        scores = (joint_weight * joint.scores + limb_weight * limb.scores) / total_weight
        fused.append(
            ScoreRow(
                source=source,
                score_type=score_type,
                fold=joint.fold,
                test_subject=joint.test_subject,
                session=joint.session,
                jsonl_path=joint.jsonl_path,
                window_start=joint.window_start,
                window_end=joint.window_end,
                center_frame=joint.center_frame,
                center_time_sec=joint.center_time_sec,
                raw_gt_label=joint.raw_gt_label,
                gt_label=joint.gt_label,
                gt_group=joint.gt_group,
                valid_detection_frames=min(joint.valid_detection_frames, limb.valid_detection_frames),
                selected_detection_center=joint.selected_detection_center or limb.selected_detection_center,
                scores=np.asarray(scores, dtype=np.float32),
            )
        )

    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse E2 joint and limb score CSVs.")
    parser.add_argument("--joint-scores", type=Path, default=DEFAULT_SCORE_DIR / "e2_joint_scores.csv")
    parser.add_argument("--limb-scores", type=Path, default=DEFAULT_SCORE_DIR / "e2_limb_scores.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE_DIR / "e2_fusion_joint_limb_scores.csv")
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--limb-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joint_rows = read_score_csv(args.joint_scores)
    limb_rows = read_score_csv(args.limb_scores)
    fused_rows = fuse_rows(
        joint_rows=joint_rows,
        limb_rows=limb_rows,
        joint_weight=args.joint_weight,
        limb_weight=args.limb_weight,
    )

    write_score_csv(args.output, fused_rows, overwrite=args.overwrite)
    write_json(
        args.output.with_name(f"{args.output.stem}_metadata.json"),
        {
            "experiment": "E2",
            "stage": "fusion",
            "method": f"weighted_{fused_rows[0].score_type}_average",
            "joint_scores": str(args.joint_scores),
            "limb_scores": str(args.limb_scores),
            "output_scores": str(args.output),
            "joint_weight": args.joint_weight,
            "limb_weight": args.limb_weight,
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "score_rows": len(fused_rows),
        },
        overwrite=args.overwrite,
    )

    print(
        f"[DONE] wrote {len(fused_rows)} fused score rows to {args.output} "
        f"(joint:limb={args.joint_weight:g}:{args.limb_weight:g})"
    )


if __name__ == "__main__":
    main()
