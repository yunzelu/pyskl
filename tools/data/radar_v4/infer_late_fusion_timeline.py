"""Run radar v4 PySKL late-fusion inference from CSV or HPE JSONL input.

This script reuses the same frame-grid readers and output CSV format as
infer_processed_pose_csv.py and infer_hpe_jsonl_timeline.py, but runs multiple
single-stream CTR-GCN models per sliding window and fuses their logits. By
default the fused logits are converted to probabilities before writing the CSV,
matching the older confidence/score column semantics.

Default fusion expects the four radar v4 streams:

    j:jm:b:bm = 2:1:2:1

Example:
    python tools/data/radar_v4/infer_late_fusion_timeline.py ^
        data/radar_v4/raw_jsonl/yolo26xpose/3-han-laysofa ^
        --model-dir work_dirs/ctrgcn/radar_v4_yolo26xpose_clip60 ^
        --output work_dirs/radar_v4_eval/3-han-laysofa_fusion.csv ^
        --stride 10 ^
        --overwrite
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_hpe_jsonl_timeline import (
    iter_jsonl_paths,
    read_jsonl_frame_grid,
)
from infer_processed_pose_csv import (
    DEFAULT_LABEL_MAP,
    FrameGrid,
    TimelineScores,
    Window,
    WindowPrediction,
    build_timeline_scores,
    iter_csv_paths,
    load_label_map,
    make_window,
    resolve_device,
    run_batch,
    update_pipeline_clip_len,
    window_center,
    window_starts,
    read_frame_grid,
    write_predictions,
)


DEFAULT_MODEL_DIR = Path("work_dirs/ctrgcn/radar_v4_yolo26xpose_clip60")
DEFAULT_CONFIG_DIR = Path("configs/ctrgcn/ctrgcn_pyskl_radarv4_loso_2d")
DEFAULT_STREAMS = ("j", "jm", "b", "bm")
DEFAULT_WEIGHTS = (2.0, 1.0, 2.0, 1.0)
DEFAULT_CHECKPOINT_NAME = "epoch_20.pth"


@dataclass(frozen=True)
class StreamSpec:
    name: str
    config_path: Path
    checkpoint_path: Path
    weight: float


@dataclass(frozen=True)
class StreamRuntime:
    spec: StreamSpec
    model: Any
    pipeline: Any


def split_values(value: str) -> list[str]:
    items = [item.strip() for item in value.replace(":", ",").split(",")]
    return [item for item in items if item]


def parse_weights(value: str) -> list[float]:
    weights = []
    for item in split_values(value):
        try:
            weights.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid weight {item!r}") from exc
    return weights


def resolve_config_path(stream: str, model_dir: Path, config_dir: Path) -> Path:
    model_config = model_dir / stream / f"{stream}.py"
    if model_config.exists():
        return model_config
    return config_dir / f"{stream}.py"


def resolve_stream_specs(
    streams: list[str],
    weights: list[float],
    model_dir: Path,
    config_dir: Path,
    checkpoint_name: str,
    configs: list[Path] | None,
    checkpoints: list[Path] | None,
) -> list[StreamSpec]:
    if len(streams) != len(weights):
        raise ValueError(
            f"--streams has {len(streams)} entries but --weights has {len(weights)}"
        )
    if not streams:
        raise ValueError("At least one stream is required")
    if any(weight < 0 for weight in weights):
        raise ValueError("--weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("At least one stream weight must be positive")

    if configs is not None and len(configs) != len(streams):
        raise ValueError(
            f"--configs has {len(configs)} paths but --streams has {len(streams)}"
        )
    if checkpoints is not None and len(checkpoints) != len(streams):
        raise ValueError(
            f"--checkpoints has {len(checkpoints)} paths but --streams has {len(streams)}"
        )

    specs = []
    for index, stream in enumerate(streams):
        config_path = (
            configs[index]
            if configs is not None
            else resolve_config_path(stream, model_dir=model_dir, config_dir=config_dir)
        )
        checkpoint_path = (
            checkpoints[index]
            if checkpoints is not None
            else model_dir / stream / checkpoint_name
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Config for stream {stream!r}: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint for stream {stream!r}: {checkpoint_path}")

        specs.append(
            StreamSpec(
                name=stream,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                weight=weights[index],
            )
        )

    return specs


def set_logit_test_cfg(config: Any) -> None:
    if "test_cfg" not in config.model or config.model.test_cfg is None:
        config.model["test_cfg"] = {}
    config.model.test_cfg["average_clips"] = "score"


def load_stream_runtimes(
    specs: list[StreamSpec],
    window_size: int,
    device: str,
    quiet: bool,
) -> list[StreamRuntime]:
    import mmcv

    from pyskl.apis import init_recognizer
    from pyskl.datasets.pipelines import Compose

    runtimes: list[StreamRuntime] = []
    for spec in specs:
        if not quiet:
            print(
                f"[INFO] Loading stream={spec.name} weight={spec.weight:g} "
                f"config={spec.config_path} checkpoint={spec.checkpoint_path}"
            )

        config = mmcv.Config.fromfile(str(spec.config_path))
        update_pipeline_clip_len(config, window_size)
        set_logit_test_cfg(config)

        model = init_recognizer(config, str(spec.checkpoint_path), device=device)
        pipeline = Compose(config.data.test.pipeline)
        runtimes.append(StreamRuntime(spec=spec, model=model, pipeline=pipeline))

    return runtimes


def softmax(scores: "np.ndarray") -> "np.ndarray":
    import numpy as np

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def run_fused_batch(
    runtimes: list[StreamRuntime],
    windows: list[Window],
    img_shape: tuple[int, int],
    normalize_weights: bool,
    score_output: str,
    num_classes: int,
) -> "np.ndarray":
    import numpy as np

    fused_logits: np.ndarray | None = None
    weight_sum = sum(runtime.spec.weight for runtime in runtimes)

    for runtime in runtimes:
        stream_logits = run_batch(
            model=runtime.model,
            pipeline=runtime.pipeline,
            windows=windows,
            img_shape=img_shape,
        )
        if stream_logits.ndim != 2 or stream_logits.shape[1] != num_classes:
            raise ValueError(
                f"Stream {runtime.spec.name!r} returned shape {stream_logits.shape}; "
                f"expected (batch, {num_classes})"
            )

        if fused_logits is None:
            fused_logits = np.zeros_like(stream_logits, dtype=np.float32)
        fused_logits += runtime.spec.weight * stream_logits.astype(np.float32, copy=False)

    if fused_logits is None:
        raise ValueError("No stream runtimes were provided")

    if normalize_weights:
        fused_logits = fused_logits / weight_sum

    if score_output == "prob":
        return softmax(fused_logits).astype(np.float32, copy=False)
    if score_output == "logit":
        return fused_logits.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported score output: {score_output}")


def infer_fused_window_predictions(
    grid: FrameGrid,
    specs: list[StreamSpec],
    window_size: int,
    stride: int,
    num_classes: int,
    batch_size: int,
    device: str,
    img_shape: tuple[int, int],
    min_valid_ratio: float,
    min_valid_frames: int | None,
    include_tail: bool,
    normalize_weights: bool,
    score_output: str,
    quiet: bool,
) -> tuple[list[WindowPrediction], "np.ndarray", int]:
    import numpy as np

    runtimes = load_stream_runtimes(
        specs=specs,
        window_size=window_size,
        device=device,
        quiet=quiet,
    )

    total_frames = grid.total_frames
    covering_windows = np.zeros(total_frames, dtype=np.int32)
    window_predictions: list[WindowPrediction] = []

    batch: list[Window] = []
    batch_valid_frames: list[int] = []
    starts = window_starts(total_frames, window_size, stride, include_tail=include_tail)
    valid_window_total = 0

    def flush_batch() -> None:
        nonlocal batch, batch_valid_frames
        if not batch:
            return

        scores = run_fused_batch(
            runtimes=runtimes,
            windows=batch,
            img_shape=img_shape,
            normalize_weights=normalize_weights,
            score_output=score_output,
            num_classes=num_classes,
        )
        for window, valid_frames, score in zip(batch, batch_valid_frames, scores):
            window_predictions.append(
                WindowPrediction(
                    start=window.start,
                    end=window.end,
                    center=window_center(window.start, window.end),
                    valid_frames=valid_frames,
                    scores=score,
                )
            )
        batch = []
        batch_valid_frames = []

    for index, start in enumerate(starts, start=1):
        window = make_window(grid=grid, start=start, window_size=window_size)
        covering_windows[window.start:window.end] += 1

        denominator = max(1, window.end - window.start)
        required = (
            min_valid_frames
            if min_valid_frames is not None
            else math.ceil(min_valid_ratio * denominator)
        )
        valid_frames = int(np.count_nonzero(grid.selected_detection[window.start:window.end]))

        if valid_frames >= required:
            batch.append(window)
            batch_valid_frames.append(valid_frames)
            valid_window_total += 1

        if len(batch) >= batch_size:
            flush_batch()

        if not quiet and index % 1000 == 0:
            print(
                f"[INFO] queued {index}/{len(starts)} windows "
                f"(valid={valid_window_total})"
            )

    flush_batch()
    if not quiet:
        print(f"[INFO] Inferred {valid_window_total}/{len(starts)} fused windows")

    return window_predictions, covering_windows, len(starts)


def mask_empty_frame_predictions(
    timeline: TimelineScores,
    selected_detection: "np.ndarray",
) -> tuple[TimelineScores, int]:
    import numpy as np

    keep = selected_detection.astype(bool, copy=False)
    masked_count = int(np.count_nonzero((~keep) & (timeline.contributing_windows > 0)))
    if masked_count == 0:
        return timeline, 0

    scores = timeline.scores.copy()
    contributing_windows = timeline.contributing_windows.copy()
    assigned_centers = timeline.assigned_centers.copy()
    assigned_window_starts = timeline.assigned_window_starts.copy()
    assigned_window_ends = timeline.assigned_window_ends.copy()
    center_distances = timeline.center_distances.copy()

    scores[~keep] = 0
    contributing_windows[~keep] = 0
    assigned_centers[~keep] = -1
    assigned_window_starts[~keep] = -1
    assigned_window_ends[~keep] = -1
    center_distances[~keep] = -1

    return (
        TimelineScores(
            scores=scores,
            contributing_windows=contributing_windows,
            assigned_centers=assigned_centers,
            assigned_window_starts=assigned_window_starts,
            assigned_window_ends=assigned_window_ends,
            center_distances=center_distances,
        ),
        masked_count,
    )


def detect_input_format(input_path: Path) -> str:
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix == ".jsonl":
            return "jsonl"
        raise ValueError(f"Cannot infer input format from file suffix: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    csv_count = len(list(input_path.glob("*.csv")))
    jsonl_count = len(list(input_path.glob("*.jsonl")))
    if csv_count and not jsonl_count:
        return "csv"
    if jsonl_count and not csv_count:
        return "jsonl"
    if csv_count and jsonl_count:
        raise ValueError(
            f"{input_path} contains both CSV and JSONL files; pass --input-format"
        )
    raise ValueError(f"No CSV or JSONL files found in {input_path}")


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_late_fusion_predictions.csv")
    return input_path / "late_fusion_predictions.csv"


def read_input_grid(args: argparse.Namespace) -> tuple[FrameGrid, tuple[int, int], str]:
    input_format = args.input_format
    if input_format == "auto":
        input_format = detect_input_format(args.input_path)

    if input_format == "csv":
        csv_paths = iter_csv_paths(args.input_path)
        if not csv_paths:
            raise ValueError(f"No CSV files found in {args.input_path}")

        grid = read_frame_grid(
            csv_paths=csv_paths,
            timestamp_col=args.timestamp_col,
            unix_time_col=args.unix_time_col,
            id_col=args.id_col,
            track_id=args.track_id,
            person_selection=args.person_selection,
            kp_threshold=args.kp_threshold,
            max_frames=args.max_frames,
        )
        return grid, tuple(args.img_shape), input_format

    if input_format == "jsonl":
        jsonl_paths = iter_jsonl_paths(args.input_path)
        if len(jsonl_paths) != 1:
            raise ValueError(
                f"Expected exactly one JSONL file under {args.input_path}, "
                f"found {len(jsonl_paths)}"
            )

        grid, img_shape = read_jsonl_frame_grid(
            jsonl_path=jsonl_paths[0],
            kp_threshold=args.kp_threshold,
            max_frames=args.max_frames,
            trust_metadata_count=not args.ignore_metadata_frame_count,
        )
        return grid, img_shape, input_format

    raise ValueError(f"Unsupported input format: {input_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted logit late-fusion CTR-GCN radar v4 inference."
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

    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAMS),
        help="Comma- or colon-separated stream names. Default: j,jm,b,bm.",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=list(DEFAULT_WEIGHTS),
        help="Comma- or colon-separated stream weights aligned to --streams. Default: 2,1,2,1.",
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
    parser.add_argument("--batch-size", type=int, default=128)
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
