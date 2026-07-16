"""E2 inference stage: write reusable center-window score CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        CENTER_OFFSET,
        DEFAULT_CONFIG_ROOT,
        DEFAULT_JSONL_ROOT,
        DEFAULT_SCORE_DIR,
        DEFAULT_WORK_ROOT,
        FPS,
        LABELS,
        STRIDE,
        WINDOW_SIZE,
        FoldSpec,
        ScoreRow,
        clean_group,
        clean_label,
        discover_folds,
        discover_subject_sessions,
        protocol_metadata,
        safe_float,
        write_json,
        write_score_csv,
    )
except ImportError:
    from common import (
        CENTER_OFFSET,
        DEFAULT_CONFIG_ROOT,
        DEFAULT_JSONL_ROOT,
        DEFAULT_SCORE_DIR,
        DEFAULT_WORK_ROOT,
        FPS,
        LABELS,
        STRIDE,
        WINDOW_SIZE,
        FoldSpec,
        ScoreRow,
        clean_group,
        clean_label,
        discover_folds,
        discover_subject_sessions,
        protocol_metadata,
        safe_float,
        write_json,
        write_score_csv,
    )
from infer_hpe_jsonl_timeline import load_jsonl_records, read_jsonl_frame_grid
from infer_late_fusion_timeline import StreamSpec, infer_fused_window_predictions
from infer_processed_pose_csv import infer_window_predictions, resolve_device


def infer_session_scores(
    fold: FoldSpec,
    jsonl_path: Path,
    stream: str,
    score_output: str,
    batch_size: int,
    device: str,
    quiet: bool,
) -> tuple[list[ScoreRow], dict[str, Any]]:
    metadata, frame_records = load_jsonl_records(jsonl_path)
    grid, img_shape = read_jsonl_frame_grid(
        jsonl_path=jsonl_path,
        kp_threshold=0.0,
        max_frames=None,
        trust_metadata_count=False,
    )

    if grid.total_frames < WINDOW_SIZE:
        return [], {
            "session": jsonl_path.parent.name,
            "jsonl_path": str(jsonl_path),
            "total_frames": grid.total_frames,
            "total_windows": 0,
            "scored_windows": 0,
            "note": "skipped: shorter than 60 frames",
        }

    if score_output == "prob":
        window_predictions, _covering_windows, total_windows = infer_window_predictions(
            grid=grid,
            config_path=fold.config_path,
            checkpoint_path=fold.checkpoint_path,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            labels=LABELS,
            batch_size=batch_size,
            device=device,
            img_shape=img_shape,
            min_valid_ratio=0.0,
            min_valid_frames=None,
            include_tail=False,
            quiet=quiet,
        )
    elif score_output == "logit":
        window_predictions, _covering_windows, total_windows = infer_fused_window_predictions(
            grid=grid,
            specs=[
                StreamSpec(
                    name=stream,
                    config_path=fold.config_path,
                    checkpoint_path=fold.checkpoint_path,
                    weight=1.0,
                )
            ],
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            num_classes=len(LABELS),
            batch_size=batch_size,
            device=device,
            img_shape=img_shape,
            min_valid_ratio=0.0,
            min_valid_frames=None,
            include_tail=False,
            normalize_weights=True,
            score_output="logit",
            quiet=quiet,
        )
    else:
        raise ValueError(f"Unsupported score output: {score_output}")

    rows: list[ScoreRow] = []
    for prediction in sorted(window_predictions, key=lambda item: item.start):
        center = prediction.start + CENTER_OFFSET
        record = frame_records.get(center, {})
        raw_label_value = record.get("label") if isinstance(record, dict) else None
        raw_label = "" if raw_label_value is None else str(raw_label_value).strip()
        gt_label = clean_label(raw_label_value)
        gt_group = clean_group(record.get("label_group") if isinstance(record, dict) else None, gt_label)

        rows.append(
            ScoreRow(
                source=stream,
                score_type=score_output,
                fold=fold.fold,
                test_subject=fold.test_subject,
                session=jsonl_path.parent.name,
                jsonl_path=jsonl_path,
                window_start=prediction.start,
                window_end=prediction.start + WINDOW_SIZE - 1,
                center_frame=center,
                center_time_sec=center / FPS,
                raw_gt_label=raw_label,
                gt_label=gt_label or "",
                gt_group=gt_group,
                valid_detection_frames=prediction.valid_frames,
                selected_detection_center=bool(grid.selected_detection[center]),
                scores=np.asarray(prediction.scores, dtype=np.float32),
            )
        )

    video_info = metadata.get("video_info", {}) if isinstance(metadata, dict) else {}
    return rows, {
        "session": jsonl_path.parent.name,
        "jsonl_path": str(jsonl_path),
        "total_frames": grid.total_frames,
        "img_shape": list(img_shape),
        "metadata_assumed_fps": safe_float(video_info.get("assumed_fps_used_for_timestamp"), FPS),
        "e2_fps": FPS,
        "total_windows": total_windows,
        "scored_windows": len(window_predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 PoseC3D inference and write window scores.")
    parser.add_argument("--stream", choices=["joint", "limb"], required=True)
    parser.add_argument(
        "--score-output",
        choices=["prob", "logit"],
        default="prob",
        help="Write probabilities by default; use logits for later temperature calibration.",
    )
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def default_output_path(output_dir: Path, stream: str, score_output: str) -> Path:
    suffix = "scores" if score_output == "prob" else "logits"
    return output_dir / f"e2_{stream}_{suffix}.csv"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = resolve_device(args.device)
    folds = discover_folds(args.config_root, args.work_root, args.stream)
    output_path = args.output or default_output_path(args.output_dir, args.stream, args.score_output)

    all_rows: list[ScoreRow] = []
    fold_infos: list[dict[str, Any]] = []
    session_infos: list[dict[str, Any]] = []

    for fold in folds:
        sessions = discover_subject_sessions(args.jsonl_root, fold.test_subject)
        fold_infos.append(
            {
                "fold": fold.fold,
                "test_subject": fold.test_subject,
                "config": str(fold.config_path),
                "checkpoint": str(fold.checkpoint_path),
                "sessions": [path.parent.name for path in sessions],
            }
        )
        if not args.quiet:
            print(
                f"[INFO] fold {fold.fold}: test_subject={fold.test_subject}, "
                f"sessions={len(sessions)}, checkpoint={fold.checkpoint_path.name}"
            )

        for jsonl_path in sessions:
            if not args.quiet:
                print(f"[INFO] inferring {args.stream}: {jsonl_path.parent.name}")
            rows, info = infer_session_scores(
                fold=fold,
                jsonl_path=jsonl_path,
                stream=args.stream,
                score_output=args.score_output,
                batch_size=args.batch_size,
                device=device,
                quiet=args.quiet,
            )
            all_rows.extend(rows)
            session_infos.append({"fold": fold.fold, "test_subject": fold.test_subject, **info})

    if not all_rows:
        raise ValueError("No E2 score rows were produced")

    write_score_csv(output_path, all_rows, overwrite=args.overwrite)
    write_json(
        output_path.with_name(f"{output_path.stem}_metadata.json"),
        {
            "experiment": "E2",
            "stage": "inference",
            "stream": args.stream,
            "score_output": args.score_output,
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "folds": fold_infos,
            "sessions": session_infos,
            "score_csv": str(output_path),
        },
        overwrite=args.overwrite,
    )

    print(f"[DONE] wrote {len(all_rows)} {args.stream} score rows to {output_path}")


if __name__ == "__main__":
    main()
