"""E3 inference stage: write continuous-window logits for calib/test splits."""

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
        DEFAULT_LOGIT_DIR,
        DEFAULT_WORK_ROOT,
        FPS,
        LABELS,
        STRIDE,
        WINDOW_SIZE,
        E3FoldSpec,
        ScoreRow,
        clean_group,
        clean_label,
        default_logits_path,
        discover_e3_folds,
        discover_subject_sessions,
        protocol_metadata,
        safe_float,
        subject_for_split,
        write_json,
        write_score_csv,
    )
except ImportError:
    from common import (
        CENTER_OFFSET,
        DEFAULT_CONFIG_ROOT,
        DEFAULT_JSONL_ROOT,
        DEFAULT_LOGIT_DIR,
        DEFAULT_WORK_ROOT,
        FPS,
        LABELS,
        STRIDE,
        WINDOW_SIZE,
        E3FoldSpec,
        ScoreRow,
        clean_group,
        clean_label,
        default_logits_path,
        discover_e3_folds,
        discover_subject_sessions,
        protocol_metadata,
        safe_float,
        subject_for_split,
        write_json,
        write_score_csv,
    )

from infer_hpe_jsonl_timeline import load_jsonl_records, read_jsonl_frame_grid
from infer_late_fusion_timeline import StreamSpec, infer_fused_window_predictions
from infer_processed_pose_csv import resolve_device


def infer_session_logits(
    fold: E3FoldSpec,
    split: str,
    subject: str,
    jsonl_path: Path,
    stream: str,
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
            "split": split,
            "session": jsonl_path.parent.name,
            "jsonl_path": str(jsonl_path),
            "total_frames": grid.total_frames,
            "total_windows": 0,
            "scored_windows": 0,
            "note": "skipped: shorter than 60 frames",
        }

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
                source=f"{stream}_{split}",
                score_type="logit",
                fold=fold.fold,
                test_subject=subject,
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
        "split": split,
        "session": jsonl_path.parent.name,
        "jsonl_path": str(jsonl_path),
        "total_frames": grid.total_frames,
        "img_shape": list(img_shape),
        "metadata_assumed_fps": safe_float(video_info.get("assumed_fps_used_for_timestamp"), FPS),
        "e3_fps": FPS,
        "total_windows": total_windows,
        "scored_windows": len(window_predictions),
    }


def infer_split(
    folds: list[E3FoldSpec],
    split: str,
    stream: str,
    jsonl_root: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    overwrite: bool,
    quiet: bool,
) -> None:
    all_rows: list[ScoreRow] = []
    fold_infos: list[dict[str, Any]] = []
    session_infos: list[dict[str, Any]] = []

    for fold in folds:
        subject = subject_for_split(fold, split)
        sessions = discover_subject_sessions(jsonl_root, subject)
        fold_infos.append(
            {
                "fold": fold.fold,
                "val_subject": fold.val_subject,
                "calib_subject": fold.calib_subject,
                "test_subject": fold.test_subject,
                "active_subject": subject,
                "config": str(fold.config_path),
                "checkpoint": str(fold.checkpoint_path),
                "sessions": [path.parent.name for path in sessions],
            }
        )
        if not quiet:
            print(
                f"[INFO] fold {fold.fold} {split}: subject={subject}, "
                f"sessions={len(sessions)}, checkpoint={fold.checkpoint_path.name}"
            )

        for jsonl_path in sessions:
            if not quiet:
                print(f"[INFO] inferring {stream} {split}: {jsonl_path.parent.name}")
            rows, info = infer_session_logits(
                fold=fold,
                split=split,
                subject=subject,
                jsonl_path=jsonl_path,
                stream=stream,
                batch_size=batch_size,
                device=device,
                quiet=quiet,
            )
            all_rows.extend(rows)
            session_infos.append({"fold": fold.fold, "subject": subject, **info})

    if not all_rows:
        raise ValueError(f"No E3 logit rows were produced for split={split}")

    write_score_csv(output_path, all_rows, overwrite=overwrite)
    write_json(
        output_path.with_name(f"{output_path.stem}_metadata.json"),
        {
            "experiment": "E3",
            "stage": "inference",
            "split": split,
            "stream": stream,
            "score_output": "logit",
            "protocol": protocol_metadata(),
            "labels": LABELS,
            "folds": fold_infos,
            "sessions": session_infos,
            "score_csv": str(output_path),
        },
        overwrite=overwrite,
    )
    print(f"[DONE] wrote {len(all_rows)} {stream} {split} logit rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 calib/test continuous-window logit inference.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--split", choices=["calib", "test", "both"], default="both")
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOGIT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = resolve_device(args.device)
    folds = discover_e3_folds(args.config_root, args.work_root, args.stream)
    splits = ["calib", "test"] if args.split == "both" else [args.split]

    for split in splits:
        infer_split(
            folds=folds,
            split=split,
            stream=args.stream,
            jsonl_root=args.jsonl_root,
            output_path=default_logits_path(args.output_dir, args.stream, split),
            batch_size=args.batch_size,
            device=device,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
