"""Run PoseC3D radar v4 inference directly on an HPE JSONL session.

PoseC3D does not need raw RGB frames here. The PySKL PoseC3D pipeline consumes
the same one-person keypoint/keypoint_score annotation as the GCN models, then
generates joint or limb heatmaps with GeneratePoseTarget.

Example:
    python tools/data/radar_v4/infer_posec3d_hpe_jsonl_timeline.py ^
        data/radar_v4/raw_jsonl/yolo26xpose/3-han-laysofa ^
        --subject chenzhe ^
        --stream joint ^
        --stride 10 ^
        --output work_dirs/radar_v4_eval/3-han-laysofa_posec3d_joint.csv ^
        --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infer_hpe_jsonl_timeline import iter_jsonl_paths, read_jsonl_frame_grid
from infer_processed_pose_csv import (
    DEFAULT_LABEL_MAP,
    build_timeline_scores,
    infer_window_predictions,
    load_label_map,
    resolve_device,
    write_predictions,
)


DEFAULT_SUBJECT = "chenzhe"
DEFAULT_STREAM = "joint"
DEFAULT_CONFIG_ROOT = Path("configs/posec3d/slowonly_r50_radarv4/911")
DEFAULT_WORK_ROOT = Path("work_dirs/posec3d")
DEFAULT_CHECKPOINT_NAME = "latest.pth"


def pkl_name(subject: str) -> str:
    return f"radarv4_yolo26xpose_clip60_val_mia_test_{subject}"


def default_config_path(subject: str, stream: str) -> Path:
    return DEFAULT_CONFIG_ROOT / subject / f"{stream}.py"


def default_checkpoint_path(subject: str, stream: str, checkpoint_name: str) -> Path:
    return DEFAULT_WORK_ROOT / pkl_name(subject) / stream / checkpoint_name


def default_output_path(input_path: Path, stream: str) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_posec3d_{stream}_predictions.csv")
    return input_path / f"posec3d_{stream}_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-stream PoseC3D radar v4 inference on an HPE JSONL session."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="HPE JSONL file or a session folder containing one JSONL file.",
    )
    parser.add_argument("--output", type=Path, help="Output frame-level prediction CSV.")
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="LOSO test subject used to resolve default config/checkpoint paths.",
    )
    parser.add_argument(
        "--stream",
        choices=["joint", "limb"],
        default=DEFAULT_STREAM,
        help="PoseC3D stream to run. Default: joint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="PoseC3D config path. Default is resolved from --subject and --stream.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path. Default is resolved from --subject, --stream, and --checkpoint-name.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint filename under work_dirs/posec3d/<fold>/<stream>. Default: latest.pth.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size. PoseC3D heatmaps are larger than GCN tensors.",
    )
    parser.add_argument(
        "--timeline-mode",
        choices=["center", "window-average"],
        default="center",
        help="Default center assigns each window to its center and fills by nearest center.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument("--kp-threshold", type=float, default=0.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--min-valid-frames", type=int)
    parser.add_argument(
        "--no-tail-window",
        action="store_true",
        help="Do not add a final tail window when stride misses the final full window.",
    )
    parser.add_argument(
        "--ignore-metadata-frame-count",
        action="store_true",
        help="Use max JSONL frame_idx + 1 instead of metadata frame_count_from_cv2.",
    )
    parser.add_argument("--max-frames", type=int, help="Debug helper: read at most this many frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_valid_ratio < 0:
        raise ValueError("--min-valid-ratio must be non-negative")
    if args.min_valid_frames is not None and args.min_valid_frames < 0:
        raise ValueError("--min-valid-frames must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    import numpy as np

    config_path = args.config or default_config_path(args.subject, args.stream)
    checkpoint_path = args.checkpoint or default_checkpoint_path(
        args.subject, args.stream, args.checkpoint_name)
    output_path = args.output or default_output_path(args.input_path, args.stream)

    jsonl_paths = iter_jsonl_paths(args.input_path)
    if len(jsonl_paths) != 1:
        raise ValueError(
            f"Expected exactly one JSONL file under {args.input_path}, found {len(jsonl_paths)}"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"PoseC3D config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PoseC3D checkpoint not found: {checkpoint_path}")

    labels = load_label_map(args.label_map)
    device = resolve_device(args.device)
    grid, img_shape = read_jsonl_frame_grid(
        jsonl_path=jsonl_paths[0],
        kp_threshold=args.kp_threshold,
        max_frames=args.max_frames,
        trust_metadata_count=not args.ignore_metadata_frame_count,
    )
    if grid.total_frames == 0:
        raise ValueError("Input contains no frames")

    if not args.quiet:
        print(f"[INFO] JSONL: {jsonl_paths[0]}")
        print(f"[INFO] Config: {config_path}")
        print(f"[INFO] Checkpoint: {checkpoint_path}")
        print(
            f"[INFO] Frames={grid.total_frames}, "
            f"selected_detection_frames={int(np.count_nonzero(grid.selected_detection))}, "
            f"img_shape={img_shape}, device={device}"
        )

    window_predictions, covering_windows, total_windows = infer_window_predictions(
        grid=grid,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        window_size=args.window_size,
        stride=args.stride,
        labels=labels,
        batch_size=args.batch_size,
        device=device,
        img_shape=img_shape,
        min_valid_ratio=args.min_valid_ratio,
        min_valid_frames=args.min_valid_frames,
        include_tail=not args.no_tail_window,
        quiet=args.quiet,
    )
    timeline = build_timeline_scores(
        total_frames=grid.total_frames,
        num_classes=len(labels),
        window_predictions=window_predictions,
        timeline_mode=args.timeline_mode,
    )
    write_predictions(
        output_path=output_path,
        grid=grid,
        labels=labels,
        timeline=timeline,
        covering_windows=covering_windows,
        overwrite=args.overwrite,
    )

    if not args.quiet:
        predicted_frames = int(np.count_nonzero(timeline.contributing_windows))
        print(f"[DONE] Wrote {grid.total_frames} frame predictions to {output_path}")
        print(f"[DONE] Valid windows: {len(window_predictions)}/{total_windows}")
        print(f"[DONE] Frames with model predictions: {predicted_frames}")


if __name__ == "__main__":
    main()
