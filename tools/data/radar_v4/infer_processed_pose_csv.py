"""Run radar v4 PySKL inference on processed YOLO pose CSV files.

The expected input is the fixed-FPS CSV produced by preprocess_pose_csv.py in
the auto_labeling_pipeline project:

    Timestamp,ID,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2,Box_Conf,KP0_X,KP0_Y,KP0_C,...

Missing artificial frames are represented by one all-zero row. This script
creates sliding windows over those artificial frames, runs the trained PySKL
model, assigns each window prediction to the center frame, and fills a dense
frame timeline by nearest center prediction.

Example:
    python tools/data/radar_v4/infer_processed_pose_csv.py ^
        D:/lu/project/auto_labeling_pipeline/data/Willowbend/618/260614/processed_pose_fps30 ^
        --output D:/tmp/radarv4_predictions.csv ^
        --overwrite

PoseC3D single-stream example:
    python tools/data/radar_v4/infer_processed_pose_csv.py ^
        D:/lu/project/auto_labeling_pipeline/data/Willowbend/618/260614/processed_pose_fps30 ^
        --baseline posec3d ^
        --subject chenzhe ^
        --stream joint ^
        --output D:/tmp/radarv4_posec3d_joint_predictions.csv ^
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASELINE = "ctrgcn"
DEFAULT_CTRGCN_STREAM = "jm"
DEFAULT_POSEC3D_STREAM = "joint"
DEFAULT_SUBJECT = "chenzhe"
DEFAULT_CHECKPOINT_NAME = "latest.pth"

DEFAULT_CONFIG = Path("configs/ctrgcn/ctrgcn_pyskl_radarv4_loso_2d/jm.py")
DEFAULT_CHECKPOINT = Path(
    r"work_dirs/ctrgcn/ctrgcn_pyskl_radarv4_loso_mia_2d/jm/epoch_14.pth"
)
DEFAULT_POSEC3D_CONFIG_ROOT = Path("configs/posec3d/slowonly_r50_radarv4/911")
DEFAULT_POSEC3D_WORK_ROOT = Path("work_dirs/posec3d")
DEFAULT_LABEL_MAP = Path("tools/data/label_map/radarv4.txt")
NUM_KEYPOINTS = 17


@dataclass
class FrameGrid:
    timestamps: list[str]
    timestamps_unix: np.ndarray
    keypoint: np.ndarray
    keypoint_score: np.ndarray
    detection_count: np.ndarray
    selected_detection: np.ndarray

    @property
    def total_frames(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    keypoint: np.ndarray
    keypoint_score: np.ndarray


@dataclass(frozen=True)
class WindowPrediction:
    start: int
    end: int
    center: int
    valid_frames: int
    scores: np.ndarray


@dataclass(frozen=True)
class TimelineScores:
    scores: np.ndarray
    contributing_windows: np.ndarray
    assigned_centers: np.ndarray
    assigned_window_starts: np.ndarray
    assigned_window_ends: np.ndarray
    center_distances: np.ndarray


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: object) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return math.nan

    try:
        return float(text)
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return math.nan


def iter_csv_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.csv") if path.is_file())
    raise FileNotFoundError(input_path)


def load_label_map(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"{path} does not contain any labels")
    return labels


def require_keypoint_columns(fieldnames: Iterable[str]) -> None:
    fields = set(fieldnames)
    missing = []
    for index in range(NUM_KEYPOINTS):
        for suffix in ("X", "Y"):
            column = f"KP{index}_{suffix}"
            if column not in fields:
                missing.append(column)
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        raise ValueError(f"Missing required keypoint columns: {preview}")


def row_has_detection(row: dict[str, str], kp_threshold: float) -> bool:
    if safe_float(row.get("Box_Conf")) > 0:
        return True
    return any(
        safe_float(row.get(f"KP{index}_C")) > kp_threshold
        for index in range(NUM_KEYPOINTS)
    )


def row_keypoint_score_sum(row: dict[str, str]) -> float:
    score_sum = safe_float(row.get("Box_Conf"))
    for index in range(NUM_KEYPOINTS):
        score_sum += safe_float(row.get(f"KP{index}_C"))
    return score_sum


def row_to_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    keypoint = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    keypoint_score = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

    for index in range(NUM_KEYPOINTS):
        x = safe_float(row.get(f"KP{index}_X"))
        y = safe_float(row.get(f"KP{index}_Y"))
        keypoint[index, 0] = x
        keypoint[index, 1] = y

        score_value = row.get(f"KP{index}_C")
        if score_value is None:
            keypoint_score[index] = 1.0 if x != 0 or y != 0 else 0.0
        else:
            keypoint_score[index] = safe_float(score_value)

    return keypoint, keypoint_score


def select_row(
    rows: list[dict[str, str]],
    id_col: str,
    track_id: int,
    person_selection: str,
) -> dict[str, str] | None:
    detection_rows = [row for row in rows if row_has_detection(row, kp_threshold=0.0)]
    if person_selection == "highest-score":
        candidates = detection_rows or rows
    else:
        candidates = [
            row
            for row in rows
            if safe_int(row.get(id_col), default=-999999) == track_id
        ]
        if not candidates and person_selection == "track-id-or-highest-score":
            candidates = detection_rows or rows

    if not candidates:
        return None
    return max(candidates, key=row_keypoint_score_sum)


def timestamp_values(
    row: dict[str, str],
    timestamp_col: str,
    unix_time_col: str,
) -> tuple[str, float]:
    raw = str(row.get(timestamp_col, "")).strip()
    raw_unix = parse_timestamp(raw)
    unix_col_value = parse_timestamp(row.get(unix_time_col)) if unix_time_col in row else math.nan

    try:
        float(raw)
        raw_is_numeric = True
    except ValueError:
        raw_is_numeric = False

    if raw_is_numeric:
        unix = raw_unix
    elif math.isfinite(unix_col_value):
        unix = unix_col_value
    else:
        unix = raw_unix

    if raw:
        return raw, unix
    if math.isfinite(unix):
        return f"{unix:.9f}".rstrip("0").rstrip("."), unix
    return "", unix


def append_frame_group(
    rows: list[dict[str, str]],
    timestamp_raw: str,
    timestamp_unix: float,
    timestamps: list[str],
    timestamps_unix: list[float],
    keypoints: list[np.ndarray],
    keypoint_scores: list[np.ndarray],
    detection_counts: list[int],
    selected_detections: list[bool],
    id_col: str,
    track_id: int,
    person_selection: str,
    kp_threshold: float,
) -> None:
    import numpy as np

    selected = select_row(
        rows=rows,
        id_col=id_col,
        track_id=track_id,
        person_selection=person_selection,
    )

    if selected is None:
        keypoint = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        keypoint_score = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        selected_detection = False
    else:
        keypoint, keypoint_score = row_to_pose(selected)
        selected_detection = row_has_detection(selected, kp_threshold=kp_threshold)

    timestamps.append(timestamp_raw)
    timestamps_unix.append(timestamp_unix)
    keypoints.append(keypoint)
    keypoint_scores.append(keypoint_score)
    detection_counts.append(
        sum(1 for row in rows if row_has_detection(row, kp_threshold=kp_threshold))
    )
    selected_detections.append(selected_detection)


def read_frame_grid(
    csv_paths: list[Path],
    timestamp_col: str,
    unix_time_col: str,
    id_col: str,
    track_id: int,
    person_selection: str,
    kp_threshold: float,
    max_frames: int | None = None,
) -> FrameGrid:
    timestamps: list[str] = []
    timestamps_unix: list[float] = []
    keypoints: list[np.ndarray] = []
    keypoint_scores: list[np.ndarray] = []
    detection_counts: list[int] = []
    selected_detections: list[bool] = []

    current_key: str | None = None
    current_raw = ""
    current_unix = math.nan
    current_rows: list[dict[str, str]] = []

    for csv_path in csv_paths:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{csv_path} has no CSV header")
            if timestamp_col not in reader.fieldnames and unix_time_col not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path} must contain {timestamp_col!r} or {unix_time_col!r}"
                )
            if id_col not in reader.fieldnames:
                raise ValueError(f"{csv_path} is missing ID column {id_col!r}")
            require_keypoint_columns(reader.fieldnames)

            for row in reader:
                timestamp_raw, timestamp_unix = timestamp_values(
                    row=row,
                    timestamp_col=timestamp_col,
                    unix_time_col=unix_time_col,
                )
                timestamp_key = timestamp_raw or str(timestamp_unix)

                if current_key is None:
                    current_key = timestamp_key
                    current_raw = timestamp_raw
                    current_unix = timestamp_unix

                if timestamp_key != current_key:
                    append_frame_group(
                        rows=current_rows,
                        timestamp_raw=current_raw,
                        timestamp_unix=current_unix,
                        timestamps=timestamps,
                        timestamps_unix=timestamps_unix,
                        keypoints=keypoints,
                        keypoint_scores=keypoint_scores,
                        detection_counts=detection_counts,
                        selected_detections=selected_detections,
                        id_col=id_col,
                        track_id=track_id,
                        person_selection=person_selection,
                        kp_threshold=kp_threshold,
                    )
                    if max_frames is not None and len(timestamps) >= max_frames:
                        return make_frame_grid(
                            timestamps,
                            timestamps_unix,
                            keypoints,
                            keypoint_scores,
                            detection_counts,
                            selected_detections,
                        )
                    current_key = timestamp_key
                    current_raw = timestamp_raw
                    current_unix = timestamp_unix
                    current_rows = []

                current_rows.append(row)

    if current_rows:
        append_frame_group(
            rows=current_rows,
            timestamp_raw=current_raw,
            timestamp_unix=current_unix,
            timestamps=timestamps,
            timestamps_unix=timestamps_unix,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            detection_counts=detection_counts,
            selected_detections=selected_detections,
            id_col=id_col,
            track_id=track_id,
            person_selection=person_selection,
            kp_threshold=kp_threshold,
        )

    return make_frame_grid(
        timestamps,
        timestamps_unix,
        keypoints,
        keypoint_scores,
        detection_counts,
        selected_detections,
    )


def make_frame_grid(
    timestamps: list[str],
    timestamps_unix: list[float],
    keypoints: list[np.ndarray],
    keypoint_scores: list[np.ndarray],
    detection_counts: list[int],
    selected_detections: list[bool],
) -> FrameGrid:
    import numpy as np

    if keypoints:
        keypoint_array = np.stack(keypoints).astype(np.float32, copy=False)
        keypoint_score_array = np.stack(keypoint_scores).astype(np.float32, copy=False)
    else:
        keypoint_array = np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32)
        keypoint_score_array = np.zeros((0, NUM_KEYPOINTS), dtype=np.float32)

    return FrameGrid(
        timestamps=timestamps,
        timestamps_unix=np.asarray(timestamps_unix, dtype=np.float64),
        keypoint=keypoint_array,
        keypoint_score=keypoint_score_array,
        detection_count=np.asarray(detection_counts, dtype=np.int32),
        selected_detection=np.asarray(selected_detections, dtype=bool),
    )


def window_starts(total_frames: int, window_size: int, stride: int, include_tail: bool) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= window_size:
        return [0]

    starts = list(range(0, total_frames - window_size + 1, stride))
    tail_start = total_frames - window_size
    if include_tail and starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def nonzero_pose_frame_mask(keypoint: np.ndarray, keypoint_score: np.ndarray) -> np.ndarray:
    import numpy as np

    keypoint_nonzero = np.any(np.abs(keypoint) > 1e-5, axis=(1, 2))
    score_nonzero = np.any(np.abs(keypoint_score) > 1e-5, axis=1)
    return np.logical_or(keypoint_nonzero, score_nonzero)


def make_window(grid: FrameGrid, start: int, window_size: int, squeeze_zero_frames: bool = False) -> Window:
    import numpy as np

    end = min(start + window_size, grid.total_frames)
    actual_len = end - start
    if squeeze_zero_frames:
        source_keypoint = grid.keypoint[start:end]
        source_keypoint_score = grid.keypoint_score[start:end]
        keep = nonzero_pose_frame_mask(source_keypoint, source_keypoint_score)
        keypoint = source_keypoint[keep][None, ...].copy()
        keypoint_score = source_keypoint_score[keep][None, ...].copy()
    else:
        keypoint = np.zeros((1, window_size, NUM_KEYPOINTS, 2), dtype=np.float32)
        keypoint_score = np.zeros((1, window_size, NUM_KEYPOINTS), dtype=np.float32)
        keypoint[0, :actual_len] = grid.keypoint[start:end]
        keypoint_score[0, :actual_len] = grid.keypoint_score[start:end]
    return Window(start=start, end=end, keypoint=keypoint, keypoint_score=keypoint_score)


def window_center(start: int, end: int) -> int:
    return start + (end - start) // 2


def make_fake_anno(window: Window, img_shape: tuple[int, int]) -> dict:
    return {
        "frame_dir": f"processed_pose_{window.start:09d}_{window.end - 1:09d}",
        "label": -1,
        "img_shape": img_shape,
        "original_shape": img_shape,
        "start_index": 0,
        "modality": "Pose",
        "total_frames": window.keypoint.shape[1],
        "test_mode": True,
        "keypoint": window.keypoint,
        "keypoint_score": window.keypoint_score,
    }


def update_pipeline_clip_len(config: Any, window_size: int) -> None:
    for stage in config.data.test.pipeline:
        if stage.get("type") in {"UniformSample", "UniformSampleFrames"}:
            stage["clip_len"] = window_size


def run_batch(
    model: Any,
    pipeline: Any,
    windows: list[Window],
    img_shape: tuple[int, int],
) -> np.ndarray:
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter

    data = [
        pipeline(make_fake_anno(window=window, img_shape=img_shape))
        for window in windows
    ]
    data_batch = collate(data, samples_per_gpu=len(data))
    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data_batch = scatter(data_batch, [device])[0]

    with torch.no_grad():
        scores = model(return_loss=False, **data_batch)

    return np.asarray(scores, dtype=np.float32)


def infer_window_predictions(
    grid: FrameGrid,
    config_path: Path,
    checkpoint_path: Path,
    window_size: int,
    stride: int,
    labels: list[str],
    batch_size: int,
    device: str,
    img_shape: tuple[int, int],
    min_valid_ratio: float,
    min_valid_frames: int | None,
    include_tail: bool,
    quiet: bool,
    squeeze_zero_frames: bool = False,
) -> tuple[list[WindowPrediction], np.ndarray, int]:
    import mmcv
    import numpy as np

    from pyskl.apis import init_recognizer
    from pyskl.datasets.pipelines import Compose

    config = mmcv.Config.fromfile(str(config_path))
    update_pipeline_clip_len(config, window_size)

    model = init_recognizer(config, str(checkpoint_path), device=device)
    pipeline = Compose(config.data.test.pipeline)

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

        scores = run_batch(
            model=model,
            pipeline=pipeline,
            windows=batch,
            img_shape=img_shape,
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
        window = make_window(
            grid=grid,
            start=start,
            window_size=window_size,
            squeeze_zero_frames=squeeze_zero_frames,
        )
        covering_windows[window.start:window.end] += 1

        denominator = max(1, window.end - window.start)
        required = (
            min_valid_frames
            if min_valid_frames is not None
            else math.ceil(min_valid_ratio * denominator)
        )
        if squeeze_zero_frames:
            valid_frames = int(window.keypoint.shape[1])
        else:
            valid_frames = int(np.count_nonzero(grid.selected_detection[window.start:window.end]))

        if squeeze_zero_frames and valid_frames == 0:
            continue
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
        print(f"[INFO] Inferred {valid_window_total}/{len(starts)} windows")

    return window_predictions, covering_windows, len(starts)


def build_center_timeline_scores(
    total_frames: int,
    num_classes: int,
    window_predictions: list[WindowPrediction],
) -> TimelineScores:
    import numpy as np

    scores = np.zeros((total_frames, num_classes), dtype=np.float32)
    contributing_windows = np.zeros(total_frames, dtype=np.int32)
    assigned_centers = np.full(total_frames, -1, dtype=np.int32)
    assigned_window_starts = np.full(total_frames, -1, dtype=np.int32)
    assigned_window_ends = np.full(total_frames, -1, dtype=np.int32)
    center_distances = np.full(total_frames, -1, dtype=np.int32)

    if total_frames == 0 or not window_predictions:
        return TimelineScores(
            scores=scores,
            contributing_windows=contributing_windows,
            assigned_centers=assigned_centers,
            assigned_window_starts=assigned_window_starts,
            assigned_window_ends=assigned_window_ends,
            center_distances=center_distances,
        )

    predictions = sorted(window_predictions, key=lambda item: (item.center, item.start))
    centers = np.asarray([item.center for item in predictions], dtype=np.int32)

    for frame_index in range(total_frames):
        right = int(np.searchsorted(centers, frame_index, side="left"))
        left = right - 1

        if left < 0:
            chosen = right
        elif right >= len(predictions):
            chosen = left
        else:
            left_distance = frame_index - int(centers[left])
            right_distance = int(centers[right]) - frame_index
            # Tie goes to the later center, matching intervals like 35-44
            # for centers 30 and 40.
            chosen = left if left_distance < right_distance else right

        prediction = predictions[chosen]
        scores[frame_index] = prediction.scores
        contributing_windows[frame_index] = 1
        assigned_centers[frame_index] = prediction.center
        assigned_window_starts[frame_index] = prediction.start
        assigned_window_ends[frame_index] = prediction.end - 1
        center_distances[frame_index] = abs(frame_index - prediction.center)

    return TimelineScores(
        scores=scores,
        contributing_windows=contributing_windows,
        assigned_centers=assigned_centers,
        assigned_window_starts=assigned_window_starts,
        assigned_window_ends=assigned_window_ends,
        center_distances=center_distances,
    )


def build_window_average_timeline_scores(
    total_frames: int,
    num_classes: int,
    window_predictions: list[WindowPrediction],
) -> TimelineScores:
    import numpy as np

    score_sums = np.zeros((total_frames, num_classes), dtype=np.float32)
    contributing_windows = np.zeros(total_frames, dtype=np.int32)
    assigned_centers = np.full(total_frames, -1, dtype=np.int32)
    assigned_window_starts = np.full(total_frames, -1, dtype=np.int32)
    assigned_window_ends = np.full(total_frames, -1, dtype=np.int32)
    center_distances = np.full(total_frames, -1, dtype=np.int32)

    for prediction in window_predictions:
        score_sums[prediction.start:prediction.end] += prediction.scores
        contributing_windows[prediction.start:prediction.end] += 1

    valid = contributing_windows > 0
    scores = score_sums.copy()
    scores[valid] = scores[valid] / contributing_windows[valid].reshape(-1, 1)

    return TimelineScores(
        scores=scores,
        contributing_windows=contributing_windows,
        assigned_centers=assigned_centers,
        assigned_window_starts=assigned_window_starts,
        assigned_window_ends=assigned_window_ends,
        center_distances=center_distances,
    )


def build_timeline_scores(
    total_frames: int,
    num_classes: int,
    window_predictions: list[WindowPrediction],
    timeline_mode: str,
) -> TimelineScores:
    if timeline_mode == "center":
        return build_center_timeline_scores(
            total_frames=total_frames,
            num_classes=num_classes,
            window_predictions=window_predictions,
        )
    if timeline_mode == "window-average":
        return build_window_average_timeline_scores(
            total_frames=total_frames,
            num_classes=num_classes,
            window_predictions=window_predictions,
        )
    raise ValueError(f"Unsupported timeline mode: {timeline_mode}")


def valid_window_coverage(
    total_frames: int,
    window_predictions: list[WindowPrediction],
) -> np.ndarray:
    import numpy as np

    keep = np.zeros(total_frames, dtype=bool)
    for prediction in window_predictions:
        keep[prediction.start:prediction.end] = True
    return keep


def mask_timeline_predictions(
    timeline: TimelineScores,
    keep: np.ndarray,
) -> tuple[TimelineScores, int]:
    import numpy as np

    if keep.shape[0] != timeline.contributing_windows.shape[0]:
        raise ValueError("Frame mask length does not match timeline length")

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


def resolve_frame_mask_mode(baseline: str, frame_mask: str) -> str:
    if frame_mask != "auto":
        return frame_mask
    return "both" if baseline == "posec3d" else "none"


def write_predictions(
    output_path: Path,
    grid: FrameGrid,
    labels: list[str],
    timeline: TimelineScores,
    covering_windows: np.ndarray,
    overwrite: bool,
) -> None:
    import numpy as np

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_columns = [f"score_{label}" for label in labels]
    columns = [
        "frame_index",
        "timestamp",
        "timestamp_unix",
        "prediction",
        "prediction_id",
        "confidence",
        "contributing_windows",
        "covering_windows",
        "detection_count",
        "selected_detection",
        "assigned_center_frame",
        "assigned_window_start",
        "assigned_window_end",
        "center_distance",
        "is_prediction_center",
        *score_columns,
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)

        for frame_index, timestamp in enumerate(grid.timestamps):
            window_count = int(timeline.contributing_windows[frame_index])
            if window_count > 0:
                scores = timeline.scores[frame_index]
                prediction_id = int(np.argmax(scores))
                prediction = labels[prediction_id]
                confidence = float(scores[prediction_id])
            else:
                scores = np.zeros(len(labels), dtype=np.float32)
                prediction_id = -1
                confidence = 0.0
                prediction = (
                    "NoPrediction"
                    if bool(grid.detection_count[frame_index])
                    else "NoDetection"
                )

            timestamp_unix = grid.timestamps_unix[frame_index]
            timestamp_unix_value = (
                f"{timestamp_unix:.9f}".rstrip("0").rstrip(".")
                if math.isfinite(float(timestamp_unix))
                else ""
            )

            writer.writerow(
                [
                    frame_index,
                    timestamp,
                    timestamp_unix_value,
                    prediction,
                    prediction_id,
                    f"{confidence:.8f}",
                    window_count,
                    int(covering_windows[frame_index]),
                    int(grid.detection_count[frame_index]),
                    int(grid.selected_detection[frame_index]),
                    int(timeline.assigned_centers[frame_index]),
                    int(timeline.assigned_window_starts[frame_index]),
                    int(timeline.assigned_window_ends[frame_index]),
                    int(timeline.center_distances[frame_index]),
                    int(
                        window_count > 0
                        and int(timeline.assigned_centers[frame_index]) == frame_index
                    ),
                    *[f"{float(score):.8f}" for score in scores],
                ]
            )


def pkl_name(subject: str) -> str:
    return f"radarv4_yolo26xpose_clip60_val_mia_test_{subject}"


def default_posec3d_config_path(subject: str, stream: str) -> Path:
    return DEFAULT_POSEC3D_CONFIG_ROOT / subject / f"{stream}.py"


def default_posec3d_checkpoint_path(
    subject: str,
    stream: str,
    checkpoint_name: str,
) -> Path:
    return DEFAULT_POSEC3D_WORK_ROOT / pkl_name(subject) / stream / checkpoint_name


def resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.baseline == "posec3d":
        config_path = args.config or default_posec3d_config_path(args.subject, args.stream)
        checkpoint_path = args.checkpoint or default_posec3d_checkpoint_path(
            args.subject,
            args.stream,
            args.checkpoint_name,
        )
    else:
        config_path = args.config or DEFAULT_CONFIG
        checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINT

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return config_path, checkpoint_path


def default_output_path(
    input_path: Path,
    baseline: str = DEFAULT_BASELINE,
    stream: str | None = None,
) -> Path:
    if baseline == "posec3d":
        suffix = f"_posec3d_{stream or DEFAULT_POSEC3D_STREAM}_predictions.csv"
        directory_name = (
            f"radarv4_posec3d_{stream or DEFAULT_POSEC3D_STREAM}_predictions.csv"
        )
    else:
        suffix = "_predictions.csv"
        directory_name = "radarv4_predictions.csv"

    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}{suffix}")
    return input_path / directory_name


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run radar v4 PySKL inference on processed YOLO pose CSV frames."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Processed pose CSV file, or a directory of processed split CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prediction CSV. Default is input-derived and includes PoseC3D stream when --baseline posec3d.",
    )
    parser.add_argument(
        "--baseline",
        choices=["ctrgcn", "posec3d"],
        default=DEFAULT_BASELINE,
        help="Default model family to resolve when --config/--checkpoint are omitted.",
    )
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="LOSO test subject used for PoseC3D default config/checkpoint paths.",
    )
    parser.add_argument(
        "--stream",
        choices=["j", "b", "jm", "bm", "joint", "limb"],
        help=(
            "Model stream. Defaults to jm for CTRGCN legacy paths and joint for PoseC3D. "
            "For explicit --config/--checkpoint this only affects default output naming."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config path. For PoseC3D default: configs/posec3d/slowonly_r50_radarv4/911/<subject>/<stream>.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path. For PoseC3D default: work_dirs/posec3d/<fold>/<stream>/<checkpoint-name>.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint filename used when resolving PoseC3D default checkpoint paths.",
    )
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size. Default: 128 for CTRGCN, 16 for PoseC3D.",
    )
    parser.add_argument(
        "--timeline-mode",
        choices=["center", "window-average"],
        default="center",
        help=(
            "How to convert sliding-window predictions to frame predictions. "
            "center assigns each window to its center and fills by nearest center. "
            "window-average averages all windows covering each frame. Default: center."
        ),
    )
    parser.add_argument(
        "--frame-mask",
        choices=["auto", "none", "empty", "valid-window", "both"],
        default="auto",
        help=(
            "Frame-level mask after window inference. auto keeps legacy dense CTRGCN "
            "output, but for PoseC3D masks frames with no selected skeleton and frames "
            "outside any window that passed --min-valid-ratio/--min-valid-frames."
        ),
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
        help="Fallback numeric timestamp column for older CSVs. Default: UnixTime.",
    )
    parser.add_argument("--id-col", default="ID")
    parser.add_argument(
        "--track-id",
        type=int,
        default=0,
        help="Track ID to classify when using track-id selection. Default: 0.",
    )
    parser.add_argument(
        "--person-selection",
        choices=["track-id", "track-id-or-highest-score", "highest-score"],
        default="track-id",
        help="How to choose one person row per artificial frame. Default: track-id.",
    )
    parser.add_argument(
        "--kp-threshold",
        type=float,
        default=0.0,
        help="Keypoint confidence threshold for deciding whether a row has a detection. Default: 0.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.5,
        help="Minimum selected-detection frame ratio required before a window is classified. Default: 0.5.",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        help="Override --min-valid-ratio with an absolute frame count.",
    )
    parser.add_argument(
        "--squeeze-zero-frames",
        action="store_true",
        help="Remove all-zero pose frames inside each accepted model window.",
    )
    parser.add_argument(
        "--img-shape",
        nargs=2,
        type=int,
        default=(720, 1280),
        metavar=("HEIGHT", "WIDTH"),
        help="Image shape used by the PySKL annotation. Default: 720 1280.",
    )
    parser.add_argument(
        "--no-tail-window",
        action="store_true",
        help="Do not add one final tail window when stride does not land on the final full window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Debug helper: read at most this many artificial frames.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np

    if args.stream is None:
        args.stream = (
            DEFAULT_POSEC3D_STREAM
            if args.baseline == "posec3d"
            else DEFAULT_CTRGCN_STREAM
        )
    if args.baseline == "posec3d" and args.stream not in {"joint", "limb"}:
        raise ValueError("--baseline posec3d requires --stream joint or --stream limb")
    if args.baseline == "ctrgcn" and args.stream not in {"j", "b", "jm", "bm"}:
        raise ValueError("--baseline ctrgcn requires --stream j, b, jm, or bm")
    if args.batch_size is None:
        args.batch_size = 16 if args.baseline == "posec3d" else 128

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

    input_path = args.input_path
    output_path = args.output or default_output_path(
        input_path,
        baseline=args.baseline,
        stream=args.stream,
    )
    config_path, checkpoint_path = resolve_model_paths(args)
    csv_paths = iter_csv_paths(input_path)
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_path}")

    labels = load_label_map(args.label_map)
    device = resolve_device(args.device)

    if not args.quiet:
        print(f"[INFO] Reading {len(csv_paths)} CSV file(s)")
        print(f"[INFO] Baseline={args.baseline}, stream={args.stream}")
        print(f"[INFO] Config: {config_path}")
        print(f"[INFO] Checkpoint: {checkpoint_path}")

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

    if grid.total_frames == 0:
        raise ValueError("Input contains no artificial frames")

    if not args.quiet:
        print(
            f"[INFO] Frames={grid.total_frames}, "
            f"selected_detection_frames={int(np.count_nonzero(grid.selected_detection))}, "
            f"device={device}"
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
        img_shape=tuple(args.img_shape),
        min_valid_ratio=args.min_valid_ratio,
        min_valid_frames=args.min_valid_frames,
        include_tail=not args.no_tail_window,
        quiet=args.quiet,
        squeeze_zero_frames=args.squeeze_zero_frames,
    )
    timeline = build_timeline_scores(
        total_frames=grid.total_frames,
        num_classes=len(labels),
        window_predictions=window_predictions,
        timeline_mode=args.timeline_mode,
    )
    frame_mask_mode = resolve_frame_mask_mode(args.baseline, args.frame_mask)
    masked_outside_valid_window = 0
    masked_empty_frames = 0
    if frame_mask_mode in {"valid-window", "both"}:
        timeline, masked_outside_valid_window = mask_timeline_predictions(
            timeline=timeline,
            keep=valid_window_coverage(
                total_frames=grid.total_frames,
                window_predictions=window_predictions,
            ),
        )
    if frame_mask_mode in {"empty", "both"}:
        timeline, masked_empty_frames = mask_timeline_predictions(
            timeline=timeline,
            keep=grid.selected_detection,
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
        print(f"[DONE] Frame mask mode: {frame_mask_mode}")
        if frame_mask_mode in {"valid-window", "both"}:
            print(
                "[DONE] Masked frames outside valid windows: "
                f"{masked_outside_valid_window}"
            )
        if frame_mask_mode in {"empty", "both"}:
            print(f"[DONE] Masked empty-frame predictions: {masked_empty_frames}")


if __name__ == "__main__":
    main()
