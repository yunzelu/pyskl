"""Stream timestamp-grouped pose CSVs through strict masks and sliding windows."""

from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def decimal_time(value: Any, context: str) -> Decimal:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean timestamp")
        timestamp = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid timestamp in {context}: {value!r}") from exc
    if not timestamp.is_finite():
        raise ValueError(f"Non-finite timestamp in {context}: {value!r}")
    return timestamp


class IntervalMask:
    """Union of closed intervals, matched without float rounding or tolerance."""

    def __init__(self, intervals: Iterable[tuple[Decimal, Decimal]]):
        merged = []
        for start, end in sorted(intervals):
            if start > end:
                raise ValueError(f"Reversed mask interval: {start} > {end}")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self.intervals = tuple(merged)
        self.starts = tuple(start for start, _ in merged)

    def contains(self, timestamp: Decimal) -> bool:
        index = bisect_right(self.starts, timestamp) - 1
        return index >= 0 and timestamp <= self.intervals[index][1]


def read_mask(path: Path) -> IntervalMask:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle, parse_float=Decimal)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of multiperson intervals in {path}")
    intervals = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not {"start_unix_time", "end_unix_time"}.issubset(item):
            raise ValueError(f"Missing start_unix_time/end_unix_time in {path}, interval {index}")
        intervals.append((
            decimal_time(item["start_unix_time"], f"{path} interval {index}"),
            decimal_time(item["end_unix_time"], f"{path} interval {index}"),
        ))
    return IntervalMask(intervals)


def new_stats() -> dict[str, Any]:
    return {key: 0 for key in (
        "input_rows", "masked_rows", "source_frames", "masked_frames", "unmasked_frames",
        "sampling_dropped_frames", "retained_frames", "candidate_windows", "valid_windows",
        "dropped_max_adjacent_gap", "dropped_max_window_span", "dropped_validity_any",
        "uncovered_tail_frames",
    )}


@dataclass(frozen=True)
class Frame:
    timestamp: Decimal
    timestamp_text: str
    csv_row_number: int
    original_id: str
    frame_index: int
    retained_index: int
    keypoint: np.ndarray
    keypoint_score: np.ndarray


def _groups(csv_path: Path, stats: dict[str, Any]):
    """Keep only a group's first detection and count, even for many detections."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Timestamp", "ID"} | {f"KP{i}_{axis}" for i in range(17) for axis in ("X", "Y", "C")}
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV header fields in {csv_path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing CSV columns in {csv_path}: {sorted(missing)}")
        first = None
        group_timestamp = None
        first_line = 0
        count = 0
        for row in reader:
            line_number = reader.line_num
            stats["input_rows"] += 1
            timestamp = decimal_time(row["Timestamp"], f"{csv_path}:{line_number}")
            if group_timestamp is not None and timestamp < group_timestamp:
                raise ValueError(f"CSV timestamps are not sorted at {csv_path}:{line_number}")
            if group_timestamp is not None and timestamp != group_timestamp:
                yield group_timestamp, first_line, first, count
                first = None
                count = 0
            if first is None:
                first = row
                first_line = line_number
                group_timestamp = timestamp
            count += 1
        if first is not None:
            yield group_timestamp, first_line, first, count


def iter_frames(
    csv_path: Path, mask: IntervalMask, stats: dict[str, Any], frame_step: int = 1, phase: int = 0,
) -> Iterator[Frame]:
    if frame_step < 1 or not 0 <= phase < frame_step:
        raise ValueError("frame_step must be positive and phase must satisfy 0 <= phase < frame_step")
    for frame_index, (timestamp, line_number, row, count) in enumerate(_groups(csv_path, stats)):
        stats["source_frames"] += 1
        if mask.contains(timestamp):
            stats["masked_frames"] += 1
            stats["masked_rows"] += count
            continue
        # Validate all unmasked groups, including groups that sampling would skip.
        if count != 1:
            raise ValueError(
                f"Unmasked timestamp {timestamp} in {csv_path} has {count} detections "
                f"starting at CSV row {line_number}. Expand/fix the paired multiperson mask."
            )
        stats["unmasked_frames"] += 1
        if frame_index % frame_step != phase:
            stats["sampling_dropped_frames"] += 1
            continue
        try:
            pose = np.asarray([
                [float(row[f"KP{i}_{axis}"]) for axis in ("X", "Y", "C")] for i in range(17)
            ], dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid keypoints in {csv_path}:{line_number}") from exc
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite keypoints in {csv_path}:{line_number}")
        if np.any(pose[:, 2] < 0) or np.any(pose[:, 2] > 1):
            raise ValueError(f"Keypoint confidence outside [0, 1] in {csv_path}:{line_number}")
        retained_index = stats["retained_frames"]
        stats["retained_frames"] += 1
        yield Frame(
            timestamp=timestamp,
            timestamp_text=row["Timestamp"].strip(),
            csv_row_number=line_number,
            original_id=str(row["ID"]),
            frame_index=frame_index,
            retained_index=retained_index,
            keypoint=np.ascontiguousarray(pose[:, :2]),
            keypoint_score=np.ascontiguousarray(pose[:, 2]),
        )


@dataclass(frozen=True)
class Window:
    frames: tuple[Frame, ...]
    candidate_index: int
    row_start: int
    max_gap_sec: float
    span_sec: float

    @property
    def center(self) -> Frame:
        return self.frames[len(self.frames) // 2]

    def to_annotation(self) -> dict[str, Any]:
        return {
            "frame_dir": f"window_{self.candidate_index:09d}",
            "keypoint": np.stack([frame.keypoint for frame in self.frames])[None, ...],
            "keypoint_score": np.stack([frame.keypoint_score for frame in self.frames])[None, ...],
            "total_frames": len(self.frames),
            "start_index": 0,
            "modality": "Pose",
            "test_mode": True,
            "label": -1,
            "img_shape": (720, 1280),
            "original_shape": (720, 1280),
        }


def iter_windows(
    frames: Iterable[Frame], stats: dict[str, Any], window_size: int = 20, stride: int = 4,
    max_gap_sec: Decimal = Decimal("0.5"), max_span_sec: Decimal = Decimal("2.5"),
) -> Iterator[Window]:
    if window_size < 2 or window_size % 2 or stride < 1:
        raise ValueError("Window size must be even and >= 2; stride must be positive")
    gap_limit = decimal_time(max_gap_sec, "max_gap_sec")
    span_limit = decimal_time(max_span_sec, "max_span_sec")
    if gap_limit <= 0 or span_limit <= 0:
        raise ValueError("Timestamp gap and span limits must be positive")
    buffer = deque(maxlen=window_size)
    seen = 0
    covered = 0
    previous_timestamp = None
    for frame in frames:
        if previous_timestamp is not None and frame.timestamp <= previous_timestamp:
            raise ValueError("Retained frame timestamps must be strictly increasing")
        previous_timestamp = frame.timestamp
        buffer.append(frame)
        seen += 1
        if len(buffer) < window_size:
            continue
        row_start = seen - window_size
        if row_start % stride:
            continue
        covered = seen
        stats["candidate_windows"] += 1
        window_frames = tuple(buffer)
        max_gap = max(b.timestamp - a.timestamp for a, b in zip(window_frames, window_frames[1:]))
        span = window_frames[-1].timestamp - window_frames[0].timestamp
        failed_gap, failed_span = max_gap > gap_limit, span > span_limit
        stats["dropped_max_adjacent_gap"] += int(failed_gap)
        stats["dropped_max_window_span"] += int(failed_span)
        if failed_gap or failed_span:
            stats["dropped_validity_any"] += 1
            continue
        stats["valid_windows"] += 1
        yield Window(window_frames, row_start // stride, row_start, float(max_gap), float(span))
    stats["uncovered_tail_frames"] = seen - covered


def iter_chunks(iterator: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size < 1:
        raise ValueError("Chunk size must be positive")
    iterator = iter(iterator)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk
