"""Render radar v4 skeleton keypoints and prediction HUD on original video.

Example:
    python tools/data/radar_v4/visualize_skeleton_predictions.py ^
        --session 35-mia-sit ^
        --predictions data/radar_v4/eval/35-mia-sit_predictions.csv ^
        --output data/radar_v4/eval/35-mia-sit_overlay.mp4 ^
        --max-frames 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_RADAR_ROOT = Path("data/radar_v4")
DEFAULT_JSONL_ROOT = DEFAULT_RADAR_ROOT / "raw_jsonl" / "yolo26xpose"
DEFAULT_ORIGIN_ROOT = DEFAULT_RADAR_ROOT / "origin"
COCO_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)
cv2 = None
np = None


def require_visual_deps() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise SystemExit(
            "This visualizer requires OpenCV and NumPy. Install them with: "
            "python -m pip install opencv-python numpy"
        ) from exc

    cv2 = cv2_module
    np = np_module


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compact_label(label: str, max_length: int = 24) -> str:
    if len(label) <= max_length:
        return label
    return label[: max_length - 1] + "~"


def discover_single_file(folder: Path, pattern: str, description: str) -> Path:
    paths = sorted(path for path in folder.glob(pattern) if path.is_file())
    if len(paths) != 1:
        raise ValueError(f"Expected one {description} in {folder}, found {len(paths)}")
    return paths[0]


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video:
        return args.video
    if not args.session:
        raise ValueError("Pass --video or --session")
    return args.origin_root / args.session / "output_video.avi"


def resolve_jsonl_path(args: argparse.Namespace) -> Path:
    if args.jsonl:
        return args.jsonl
    if not args.session:
        raise ValueError("Pass --jsonl or --session")
    return discover_single_file(args.jsonl_root / args.session, "*.jsonl", "JSONL file")


def load_jsonl_frames(jsonl_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    frames: dict[int, dict[str, Any]] = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "metadata":
                metadata = record
            elif record_type == "frame":
                frame_idx = safe_int(record.get("frame_idx"), default=-1)
                if frame_idx >= 0:
                    frames[frame_idx] = record

    if not frames:
        raise ValueError(f"{jsonl_path} contains no frame records")
    return metadata, frames


def metadata_video_shape(metadata: dict[str, Any]) -> tuple[int, int]:
    video_info = metadata.get("video_info", {}) if metadata else {}
    width = safe_int(video_info.get("width"), default=0)
    height = safe_int(video_info.get("height"), default=0)
    return width, height


def metadata_fps(metadata: dict[str, Any], fallback: float) -> float:
    video_info = metadata.get("video_info", {}) if metadata else {}
    fps = safe_float(video_info.get("assumed_fps_used_for_timestamp"), default=0.0)
    if fps <= 0:
        fps = safe_float(video_info.get("reported_fps_from_cv2"), default=0.0)
    if fps <= 0:
        fps = fallback
    return fps if fps > 0 else 30.0


def load_predictions(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}

    predictions: dict[int, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        if "frame_index" not in reader.fieldnames:
            raise ValueError(f"{path} is missing frame_index column")

        for row in reader:
            frame_index = safe_int(row.get("frame_index"), default=-1)
            if frame_index >= 0:
                predictions[frame_index] = row

    return predictions


def top_score_items(row: dict[str, str], limit: int = 5) -> list[tuple[str, float]]:
    scores = [
        (column.removeprefix("score_"), safe_float(value))
        for column, value in row.items()
        if column.startswith("score_")
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:limit]


def scale_xy(
    x: float,
    y: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        return round(x), round(y)
    return (
        round(x * target_width / source_width),
        round(y * target_height / source_height),
    )


def keypoints_from_record(record: dict[str, Any]) -> tuple[list[tuple[float, float]], list[float]]:
    keypoints_xy = record.get("keypoints_xy")
    keypoints_conf = record.get("keypoints_conf")

    points: list[tuple[float, float]] = [(0.0, 0.0)] * 17
    scores: list[float] = [0.0] * 17

    if isinstance(keypoints_xy, list):
        for index, xy in enumerate(keypoints_xy[:17]):
            if isinstance(xy, list) and len(xy) >= 2:
                points[index] = (safe_float(xy[0]), safe_float(xy[1]))

    if isinstance(keypoints_conf, list):
        for index, confidence in enumerate(keypoints_conf[:17]):
            scores[index] = safe_float(confidence)
    else:
        for index, point in enumerate(points):
            if point != (0.0, 0.0):
                scores[index] = 1.0

    return points, scores


def draw_bbox(
    frame: "np.ndarray",
    record: dict[str, Any],
    source_width: int,
    source_height: int,
    color: tuple[int, int, int],
) -> None:
    bbox = record.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return

    height, width = frame.shape[:2]
    x1, y1 = scale_xy(safe_float(bbox[0]), safe_float(bbox[1]), source_width, source_height, width, height)
    x2, y2 = scale_xy(safe_float(bbox[2]), safe_float(bbox[3]), source_width, source_height, width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"bbox {safe_float(record.get('bbox_conf')):.2f}",
        (x1, max(14, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_skeleton(
    frame: "np.ndarray",
    record: dict[str, Any] | None,
    source_width: int,
    source_height: int,
    threshold: float,
    draw_boxes: bool,
    draw_labels: bool,
) -> tuple[int, int]:
    if record is None or not record.get("detected"):
        return 0, 0

    points, scores = keypoints_from_record(record)
    height, width = frame.shape[:2]
    scaled_points = [
        scale_xy(x, y, source_width, source_height, width, height)
        for x, y in points
    ]

    high_color = (60, 230, 80)
    low_color = (45, 80, 255)
    high_count = 0
    low_count = 0

    for start_index, end_index in COCO_SKELETON:
        score_a = scores[start_index]
        score_b = scores[end_index]
        if score_a <= 0 or score_b <= 0:
            continue

        both_high = score_a >= threshold and score_b >= threshold
        cv2.line(
            frame,
            scaled_points[start_index],
            scaled_points[end_index],
            high_color if both_high else low_color,
            2 if both_high else 1,
            cv2.LINE_AA,
        )

    for index, ((x, y), score) in enumerate(zip(scaled_points, scores)):
        raw_x, raw_y = points[index]
        if score <= 0 or (raw_x == 0 and raw_y == 0):
            continue

        if score >= threshold:
            high_count += 1
            cv2.circle(frame, (x, y), 4, high_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            label_color = high_color
        else:
            low_count += 1
            cv2.circle(frame, (x, y), 4, low_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            label_color = low_color

        if draw_labels:
            cv2.putText(
                frame,
                f"{index}:{score:.2f}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                label_color,
                1,
                cv2.LINE_AA,
            )

    if draw_boxes:
        draw_bbox(frame, record, source_width, source_height, (255, 180, 60))

    return high_count, low_count


def draw_hud(
    frame: "np.ndarray",
    frame_index: int,
    record: dict[str, Any] | None,
    prediction: dict[str, str] | None,
    high_count: int,
    low_count: int,
    threshold: float,
    alpha: float,
) -> None:
    timestamp_sec = safe_float(record.get("timestamp_sec")) if record else frame_index / 30.0
    detected = bool(record and record.get("detected"))
    lines = [
        f"frame {frame_index}   t={timestamp_sec:.3f}s   detected={int(detected)}",
        f"keypoints >= {threshold:.2f}: {high_count}   < {threshold:.2f}: {low_count}",
    ]

    if record:
        lines.append(
            f"bbox_conf={safe_float(record.get('bbox_conf')):.3f}   "
            f"detections={safe_int(record.get('num_detections'))}"
        )

    if prediction:
        lines.append(
            f"prediction: {prediction.get('prediction', '')} "
            f"{safe_float(prediction.get('confidence')):.4f}   "
            f"center={prediction.get('assigned_center_frame', '')}"
        )
        top5 = top_score_items(prediction, limit=5)
        for rank, (label, score) in enumerate(top5, start=1):
            lines.append(f"{rank}. {compact_label(label)}  {score:.4f}")
    else:
        lines.append("prediction: none")

    legend_y = frame.shape[0] - 18
    cv2.circle(frame, (18, legend_y - 4), 5, (60, 230, 80), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        ">= threshold",
        (30, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (130, legend_y - 4), 5, (45, 80, 255), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "< threshold",
        (142, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    hud_width = min(frame.shape[1] - 16, max(360, 9 * max(len(line) for line in lines)))
    hud_height = 16 + 18 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + hud_width, 8 + hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    for line_index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


class ProgressBar:
    def __init__(self, total: int | None, enabled: bool = True) -> None:
        self.total = total
        self.enabled = enabled
        self.start_time = time.monotonic()
        self.last_draw = 0.0

    def update(self, current: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_draw < 0.25:
            return
        self.last_draw = now
        elapsed = max(1e-6, now - self.start_time)
        rate = current / elapsed
        if self.total:
            pct = min(100.0, 100.0 * current / self.total)
            sys.stderr.write(f"\rRendering {current}/{self.total} ({pct:5.1f}%) {rate:5.1f} fps")
        else:
            sys.stderr.write(f"\rRendering {current} frames {rate:5.1f} fps")
        sys.stderr.flush()

    def finish(self, current: int) -> None:
        if not self.enabled:
            return
        self.last_draw = 0.0
        self.update(current)
        sys.stderr.write("\n")
        sys.stderr.flush()


def render_overlay(
    video_path: Path,
    jsonl_path: Path,
    output_path: Path,
    predictions_path: Path | None,
    threshold: float,
    start_frame: int,
    end_frame: int | None,
    max_frames: int | None,
    output_fps: float | None,
    output_width: int | None,
    output_height: int | None,
    draw_boxes: bool,
    draw_kp_labels: bool,
    hud_alpha: float,
    show_progress: bool,
) -> int:
    require_visual_deps()

    metadata, jsonl_frames = load_jsonl_frames(jsonl_path)
    predictions = load_predictions(predictions_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    meta_width, meta_height = metadata_video_shape(metadata)
    if meta_width > 0 and meta_height > 0:
        source_width, source_height = meta_width, meta_height

    if output_fps is None:
        output_fps = metadata_fps(metadata, fallback=video_fps)
    if output_width is None:
        output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if output_height is None:
        output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = video_frame_count - 1 if video_frame_count > 0 else max(jsonl_frames)
    if max_frames is not None:
        end_frame = min(end_frame, start_frame + max_frames - 1)
    if end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (output_width, output_height))
    if not writer.isOpened():
        raise ValueError(f"Could not open writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total = end_frame - start_frame + 1
    progress = ProgressBar(total, enabled=show_progress)
    rendered = 0

    try:
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)

            record = jsonl_frames.get(frame_index)
            high_count, low_count = draw_skeleton(
                frame=frame,
                record=record,
                source_width=source_width,
                source_height=source_height,
                threshold=threshold,
                draw_boxes=draw_boxes,
                draw_labels=draw_kp_labels,
            )
            draw_hud(
                frame=frame,
                frame_index=frame_index,
                record=record,
                prediction=predictions.get(frame_index),
                high_count=high_count,
                low_count=low_count,
                threshold=threshold,
                alpha=hud_alpha,
            )
            writer.write(frame)
            rendered += 1
            progress.update(rendered)
    finally:
        cap.release()
        writer.release()
        progress.finish(rendered)

    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay radar v4 HPE JSONL skeletons and inference top-5 HUD on original video."
    )
    parser.add_argument("--session", help="Session name, e.g. 35-mia-sit.")
    parser.add_argument("--video", type=Path, help="Explicit original video path.")
    parser.add_argument("--jsonl", type=Path, help="Explicit HPE JSONL path.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Frame-level prediction CSV from infer_*_timeline.py.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 path.")
    parser.add_argument("--origin-root", type=Path, default=DEFAULT_ORIGIN_ROOT)
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for high vs low keypoints. Default: 0.5.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, help="Output FPS. Default: JSONL assumed FPS, then video FPS.")
    parser.add_argument("--width", type=int, help="Output width. Default: original video width.")
    parser.add_argument("--height", type=int, help="Output height. Default: original video height.")
    parser.add_argument("--no-bbox", action="store_true", help="Do not draw bbox.")
    parser.add_argument("--no-kp-labels", action="store_true", help="Do not draw keypoint index:confidence labels.")
    parser.add_argument("--hud-alpha", type=float, default=0.58, help="HUD background opacity. Default: 0.58.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = resolve_video_path(args)
    jsonl_path = resolve_jsonl_path(args)

    frame_count = render_overlay(
        video_path=video_path,
        jsonl_path=jsonl_path,
        output_path=args.output,
        predictions_path=args.predictions,
        threshold=args.threshold,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        output_fps=args.fps,
        output_width=args.width,
        output_height=args.height,
        draw_boxes=not args.no_bbox,
        draw_kp_labels=not args.no_kp_labels,
        hud_alpha=max(0.0, min(1.0, args.hud_alpha)),
        show_progress=not args.no_progress,
    )
    print(f"Rendered {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
