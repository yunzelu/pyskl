"""Run two-stream PoseC3D late-fusion inference from CSV or HPE JSONL input.

This is the PoseC3D counterpart of infer_late_fusion_timeline.py. It defaults
to joint+limb streams and keeps the existing one-person window construction:
one selected skeleton per frame, shaped as M=1 before the PoseC3D pipeline
generates heatmaps.

Example:
    python tools/data/radar_v4/infer_posec3d_late_fusion_timeline.py ^
        data/radar_v4/raw_jsonl/yolo26xpose/3-han-laysofa ^
        --subject chenzhe ^
        --stride 10 ^
        --output work_dirs/radar_v4_eval/3-han-laysofa_posec3d_fusion.csv ^
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

from infer_late_fusion_timeline import (
    DEFAULT_LABEL_MAP,
    build_timeline_scores,
    infer_fused_window_predictions,
    load_label_map,
    mask_empty_frame_predictions,
    parse_weights,
    read_input_grid,
    resolve_device,
    resolve_stream_specs,
    split_values,
    write_predictions,
)


DEFAULT_SUBJECT = "chenzhe"
DEFAULT_STREAMS = ("joint", "limb")
DEFAULT_WEIGHTS = (1.0, 1.0)
DEFAULT_CONFIG_ROOT = Path("configs/posec3d/slowonly_r50_radarv4/911")
DEFAULT_WORK_ROOT = Path("work_dirs/posec3d")
DEFAULT_CHECKPOINT_NAME = "latest.pth"


def pkl_name(subject: str) -> str:
    return f"radarv4_yolo26xpose_clip60_val_mia_test_{subject}"


def default_config_dir(subject: str) -> Path:
    return DEFAULT_CONFIG_ROOT / subject


def default_model_dir(subject: str) -> Path:
    return DEFAULT_WORK_ROOT / pkl_name(subject)


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_posec3d_fusion_predictions.csv")
    return input_path / "posec3d_fusion_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted logit late-fusion PoseC3D radar v4 inference."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Processed pose CSV file/dir, or HPE JSONL file/session dir.",
    )
    parser.add_argument("--output", type=Path, help="Output frame-level prediction CSV.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "jsonl"],
        default="auto",
        help="Input parser to use. Default: auto.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="LOSO test subject used to resolve default config/model dirs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing joint/ and limb/ checkpoint folders. Default is resolved from --subject.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory containing joint.py and limb.py. Default is resolved from --subject.",
    )
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAMS),
        help="Comma- or colon-separated stream names. Default: joint,limb.",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=list(DEFAULT_WEIGHTS),
        help="Comma- or colon-separated stream weights aligned to --streams. Default: 1,1.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        help="Optional explicit config paths aligned to --streams.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="Optional explicit checkpoint paths aligned to --streams.",
    )
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
        help="Default center assigns each fused window to its center and fills by nearest center.",
    )
    parser.add_argument(
        "--score-output",
        choices=["logit", "prob"],
        default="prob",
        help="Values written to confidence/score_* columns after logit fusion. Default: prob.",
    )
    parser.add_argument(
        "--no-normalize-weights",
        action="store_true",
        help="Use weighted logit sum instead of weighted average.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument("--timestamp-col", default="Timestamp")
    parser.add_argument(
        "--unix-time-col",
        default="UnixTime",
        help="CSV fallback numeric timestamp column. Default: UnixTime.",
    )
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument(
        "--person-selection",
        choices=["track-id", "track-id-or-highest-score", "highest-score"],
        default="track-id",
        help="CSV only: how to choose one person row per frame. Default: track-id.",
    )
    parser.add_argument(
        "--img-shape",
        nargs=2,
        type=int,
        default=(720, 1280),
        metavar=("HEIGHT", "WIDTH"),
        help="CSV image shape used by the PySKL annotation. JSONL uses metadata. Default: 720 1280.",
    )
    parser.add_argument(
        "--ignore-metadata-frame-count",
        action="store_true",
        help="JSONL only: use max frame_idx + 1 instead of metadata frame_count_from_cv2.",
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
        "--keep-empty-frame-predictions",
        action="store_true",
        help=(
            "Keep nearest-window predictions on frames with no selected skeleton input. "
            "By default those frames are written as NoDetection or NoPrediction."
        ),
    )
    parser.add_argument("--max-frames", type=int, help="Debug helper: read at most this many frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = default_model_dir(args.subject)
    if args.config_dir is None:
        args.config_dir = default_config_dir(args.subject)
    return args


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

    labels = load_label_map(args.label_map)
    streams = split_values(args.streams)
    specs = resolve_stream_specs(
        streams=streams,
        weights=args.weights,
        model_dir=args.model_dir,
        config_dir=args.config_dir,
        checkpoint_name=args.checkpoint_name,
        configs=args.configs,
        checkpoints=args.checkpoints,
    )
    device = resolve_device(args.device)
    output_path = args.output or default_output_path(args.input_path)

    grid, img_shape, input_format = read_input_grid(args)
    if grid.total_frames == 0:
        raise ValueError("Input contains no frames")

    if not args.quiet:
        stream_summary = ", ".join(
            f"{spec.name}:{spec.weight:g}" for spec in specs
        )
        print(f"[INFO] Input format={input_format}, frames={grid.total_frames}")
        print(f"[INFO] Config dir={args.config_dir}")
        print(f"[INFO] Model dir={args.model_dir}")
        print(
            f"[INFO] selected_detection_frames="
            f"{int(np.count_nonzero(grid.selected_detection))}, "
            f"img_shape={img_shape}, device={device}"
        )
        print(
            f"[INFO] Fusion streams={stream_summary}, "
            f"score_output={args.score_output}, "
            f"normalize_weights={not args.no_normalize_weights}"
        )

    window_predictions, covering_windows, total_windows = infer_fused_window_predictions(
        grid=grid,
        specs=specs,
        window_size=args.window_size,
        stride=args.stride,
        num_classes=len(labels),
        batch_size=args.batch_size,
        device=device,
        img_shape=img_shape,
        min_valid_ratio=args.min_valid_ratio,
        min_valid_frames=args.min_valid_frames,
        include_tail=not args.no_tail_window,
        normalize_weights=not args.no_normalize_weights,
        score_output=args.score_output,
        quiet=args.quiet,
    )
    timeline = build_timeline_scores(
        total_frames=grid.total_frames,
        num_classes=len(labels),
        window_predictions=window_predictions,
        timeline_mode=args.timeline_mode,
    )
    masked_empty_frames = 0
    if not args.keep_empty_frame_predictions:
        timeline, masked_empty_frames = mask_empty_frame_predictions(
            timeline=timeline,
            selected_detection=grid.selected_detection,
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
        print(f"[DONE] Valid fused windows: {len(window_predictions)}/{total_windows}")
        print(f"[DONE] Frames with model predictions: {predicted_frames}")
        if not args.keep_empty_frame_predictions:
            print(f"[DONE] Masked empty-frame predictions: {masked_empty_frames}")


if __name__ == "__main__":
    main()
