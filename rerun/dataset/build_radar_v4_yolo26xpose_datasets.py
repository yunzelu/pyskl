"""Build RADAR v4 rerun datasets from YOLO26x-pose JSONL skeletons.

This script implements the thesis rerun dataset protocol:

1. Preprocess source JSONL files by removing frame rows with ``detected: false``.
2. Build activity-aligned samples from ``annotation_info/segments`` in JSONL
   metadata, with strict detected-frame and timestamp-gap validity checks, but
   without transition expansion, interpolation, or fixed-duration constraints.
3. Build continuous-window samples from detected skeleton rows using a
   60-frame window, 12-frame stride, center-row labels, and timestamp validity
   checks.
4. Build an optional triangular temporal-composition variant of the continuous
   windows for soft-label training.
5. Save PYSKL-compatible pickle files plus sidecar statistics for the fixed
   subject-wise folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ID = "radarv4_yolo26xpose"
RAW_JSONL_ROOT = Path("data/radar_v4/raw_jsonl/yolo26xpose")
OUTPUT_ROOT = Path("data/radar_v4/rerun/yolo26xpose")

INCLUDED_SESSION_FAMILIES = {"fall", "sit", "laysofa"}
OPEN_ENDED_TAIL_RECORDINGS = {"12-xilai-sit2", "19-saad-laysofa", "7-han-sit"}
ACTIVITY_MIN_DETECTED_FRAMES = 2

FINAL_LABELS = [
    "lie-stationary",
    "sit-stationary",
    "walk",
    "fall",
    "transition-lie-to-sit",
    "transition-lie-to-stand",
    "transition-sit-to-lie",
    "transition-sit-to-stand",
    "transition-stand-to-sit",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(FINAL_LABELS)}

LABEL_ALIASES = {
    "layfloor-stationary": "lie-stationary",
    "laybed-stationary": "lie-stationary",
    "lie-stationary": "lie-stationary",
    "lying-stationary": "lie-stationary",
    "sit-stationary": "sit-stationary",
    "walking": "walk",
    "walk": "walk",
    "falling": "fall",
    "fall": "fall",
    "transition-layfloor-to-sit": "transition-lie-to-sit",
    "transition-laybed-to-sit": "transition-lie-to-sit",
    "transition-lie-to-sit": "transition-lie-to-sit",
    "transition-layfloor-to-stand": "transition-lie-to-stand",
    "transition-laybed-to-stand": "transition-lie-to-stand",
    "transition-lie-to-stand": "transition-lie-to-stand",
    "transition-sit-to-layfloor": "transition-sit-to-lie",
    "transition-sit-to-laybed": "transition-sit-to-lie",
    "transition-sit-to-lie": "transition-sit-to-lie",
    "transition-sit-to-stand": "transition-sit-to-stand",
    "transition-stand-to-sit": "transition-stand-to-sit",
}

FOLDS = {
    "fold_a": {
        "train": ["chenzhe", "dengdeng", "hui", "jiadi", "mia", "rose", "xilai", "yunze"],
        "val": ["han"],
        "calib": ["saad"],
        "test": ["li"],
    },
    "fold_b": {
        "train": ["han", "jiadi", "li", "mia", "rose", "saad", "xilai", "yunze"],
        "val": ["hui"],
        "calib": ["chenzhe"],
        "test": ["dengdeng"],
    },
    "fold_c": {
        "train": ["chenzhe", "dengdeng", "han", "hui", "jiadi", "li", "saad", "xilai"],
        "val": ["rose"],
        "calib": ["mia"],
        "test": ["yunze"],
    },
}


@dataclass(frozen=True)
class SessionId:
    directory_name: str
    recording_index: int
    subject: str
    session: str
    session_family: str


@dataclass
class PreprocessRecord:
    session_dir: str
    subject: str
    session: str
    session_family: str
    source_jsonl: str
    processed_jsonl: str
    source_frame_rows: int
    kept_frame_rows: int
    removed_false_detection_rows: int


@dataclass
class SessionData:
    identity: SessionId
    jsonl_path: Path
    raw_jsonl_path: str | None
    img_shape: tuple[int, int]
    keypoint: np.ndarray
    keypoint_score: np.ndarray
    frame_indices: np.ndarray
    timestamps_sec: np.ndarray
    frame_labels: list[Any]
    segments: list[dict[str, Any]]
    num_keypoints: int


@dataclass
class ProtocolResult:
    protocol_id: str
    annotations: list[dict[str, Any]]
    stats: dict[str, Any]


def sanitize_id(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    key = str(label).strip().lower()
    if not key:
        return None
    return LABEL_ALIASES.get(key)


def bartlett_triangular_weights(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Triangular target length must be positive.")
    if length == 1:
        return np.ones(1, dtype=np.float64)

    positions = np.arange(length, dtype=np.float64)
    raw = 1.0 - np.abs(2.0 * positions - float(length - 1)) / float(length - 1)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("Triangular target weights have zero total mass.")
    return raw / total


def parse_session_dir_name(name: str) -> SessionId:
    parts = name.split("-", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Expected session directory '<index>-<subject>-<session>', got {name!r}."
        )

    index_text, subject, session = parts
    if not index_text.isdigit():
        raise ValueError(f"Expected numeric session index in {name!r}.")

    session_lower = session.strip().lower()
    session_family = re.sub(r"\d+$", "", session_lower)
    return SessionId(
        directory_name=name,
        recording_index=int(index_text),
        subject=subject.strip().lower(),
        session=session_lower,
        session_family=session_family,
    )


def include_session(identity: SessionId) -> bool:
    return identity.session_family in INCLUDED_SESSION_FAMILIES


def find_single_jsonl(session_dir: Path) -> Path:
    jsonl_paths = sorted(session_dir.glob("*.jsonl"))
    if len(jsonl_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one JSONL file in {session_dir}, found {len(jsonl_paths)}."
        )
    return jsonl_paths[0]


def selected_session_dirs(raw_jsonl_root: Path, scope: str) -> list[tuple[SessionId, Path]]:
    if not raw_jsonl_root.exists():
        raise FileNotFoundError(f"Raw JSONL root not found: {raw_jsonl_root}")

    selected: list[tuple[SessionId, Path]] = []
    for session_dir in sorted(path for path in raw_jsonl_root.iterdir() if path.is_dir()):
        identity = parse_session_dir_name(session_dir.name)
        if scope == "included" and not include_session(identity):
            continue
        selected.append((identity, session_dir))
    return selected


def scan_jsonl(path: Path) -> tuple[dict[str, Any], int, int, int]:
    metadata: dict[str, Any] | None = None
    source_frame_rows = 0
    kept_frame_rows = 0
    removed_false_detection_rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "metadata" and metadata is None:
                metadata = record
            elif record_type == "frame":
                source_frame_rows += 1
                if record.get("detected") is False:
                    removed_false_detection_rows += 1
                else:
                    kept_frame_rows += 1

    if metadata is None:
        raise RuntimeError(f"No metadata line found in {path}")

    return metadata, source_frame_rows, kept_frame_rows, removed_false_detection_rows


def json_dump_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def add_preprocess_metadata(
    metadata: dict[str, Any],
    identity: SessionId,
    source_jsonl: Path,
    processed_jsonl: Path,
    source_frame_rows: int,
    kept_frame_rows: int,
    removed_false_detection_rows: int,
) -> dict[str, Any]:
    result = dict(metadata)
    result["preprocess_info"] = {
        "operation": "remove_frame_rows_where_detected_is_false",
        "source_jsonl": str(source_jsonl),
        "processed_jsonl": str(processed_jsonl),
        "source_frame_rows": source_frame_rows,
        "kept_frame_rows": kept_frame_rows,
        "removed_false_detection_rows": removed_false_detection_rows,
        "frame_idx_policy": "preserve original frame_idx values",
        "timestamp_policy": "preserve original timestamp_sec values",
        "session_dir": identity.directory_name,
        "subject": identity.subject,
        "session": identity.session,
        "session_family": identity.session_family,
    }
    return result


def preprocess_jsonl_files(
    raw_jsonl_root: Path,
    detected_jsonl_root: Path,
    scope: str,
) -> list[PreprocessRecord]:
    detected_jsonl_root.mkdir(parents=True, exist_ok=True)
    records: list[PreprocessRecord] = []

    for identity, session_dir in selected_session_dirs(raw_jsonl_root, scope):
        source_jsonl = find_single_jsonl(session_dir)
        target_dir = detected_jsonl_root / identity.directory_name
        target_dir.mkdir(parents=True, exist_ok=True)
        processed_jsonl = target_dir / f"{source_jsonl.stem}__detected-only.jsonl"

        metadata, source_rows, kept_rows, removed_rows = scan_jsonl(source_jsonl)
        dataset_info = metadata.get("dataset_info", {})
        metadata_subject = str(dataset_info.get("subject", "")).strip().lower()
        if metadata_subject and metadata_subject != identity.subject:
            raise RuntimeError(
                f"Subject mismatch for {source_jsonl}: directory has "
                f"{identity.subject!r}, metadata has {metadata_subject!r}."
            )

        metadata_written = False
        tmp_path = processed_jsonl.with_suffix(processed_jsonl.suffix + ".tmp")
        with source_jsonl.open("r", encoding="utf-8") as src, tmp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "metadata":
                    if metadata_written:
                        raise RuntimeError(f"Multiple metadata lines found in {source_jsonl}")
                    dst.write(
                        json_dump_line(
                            add_preprocess_metadata(
                                metadata=record,
                                identity=identity,
                                source_jsonl=source_jsonl,
                                processed_jsonl=processed_jsonl,
                                source_frame_rows=source_rows,
                                kept_frame_rows=kept_rows,
                                removed_false_detection_rows=removed_rows,
                            )
                        )
                    )
                    metadata_written = True
                    continue

                if record.get("type") == "frame" and record.get("detected") is False:
                    continue
                dst.write(json_dump_line(record))

        tmp_path.replace(processed_jsonl)
        records.append(
            PreprocessRecord(
                session_dir=identity.directory_name,
                subject=identity.subject,
                session=identity.session,
                session_family=identity.session_family,
                source_jsonl=str(source_jsonl),
                processed_jsonl=str(processed_jsonl),
                source_frame_rows=source_rows,
                kept_frame_rows=kept_rows,
                removed_false_detection_rows=removed_rows,
            )
        )

    return records


def discover_processed_jsonls(detected_jsonl_root: Path) -> list[Path]:
    if not detected_jsonl_root.exists():
        raise FileNotFoundError(f"Detected-only JSONL root not found: {detected_jsonl_root}")
    return sorted(detected_jsonl_root.rglob("*.jsonl"))


def read_processed_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "metadata":
                if metadata is not None:
                    raise RuntimeError(f"Multiple metadata lines found in {path}")
                metadata = record
            elif record_type == "frame":
                if record.get("detected") is False:
                    raise RuntimeError(f"Detected-only JSONL still has detected=false row: {path}")
                frames.append(record)

    if metadata is None:
        raise RuntimeError(f"No metadata line found in {path}")

    frames.sort(key=lambda item: int(item["frame_idx"]))
    return metadata, frames


def infer_num_keypoints(frames: list[dict[str, Any]], path: Path) -> int:
    for frame in frames:
        keypoints_xy = frame.get("keypoints_xy")
        if isinstance(keypoints_xy, list) and keypoints_xy:
            return len(keypoints_xy)
    raise RuntimeError(f"No keypoints found in detected frames for {path}")


def read_image_shape(metadata: dict[str, Any], path: Path) -> tuple[int, int]:
    video_info = metadata.get("video_info", {})
    height = video_info.get("height")
    width = video_info.get("width")
    if height is None or width is None:
        raise RuntimeError(f"Missing video_info.height/width in {path}")
    return int(height), int(width)


def metadata_segments(metadata: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    segments = metadata.get("annotation_info", {}).get("segments")
    if not isinstance(segments, list):
        raise RuntimeError(f"Missing annotation_info.segments list in {path}")
    return [dict(segment) for segment in segments if isinstance(segment, dict)]


def frames_to_arrays(
    frames: list[dict[str, Any]],
    num_keypoints: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    frame_count = len(frames)
    keypoint = np.zeros((frame_count, num_keypoints, 2), dtype=np.float32)
    keypoint_score = np.zeros((frame_count, num_keypoints), dtype=np.float32)
    frame_indices = np.zeros(frame_count, dtype=np.int32)
    timestamps_sec = np.zeros(frame_count, dtype=np.float64)
    labels: list[Any] = []

    for row_index, frame in enumerate(frames):
        if "frame_idx" not in frame:
            raise RuntimeError(f"Missing frame_idx in {path}")
        if "timestamp_sec" not in frame:
            raise RuntimeError(f"Missing timestamp_sec in {path}")

        frame_indices[row_index] = int(frame["frame_idx"])
        timestamps_sec[row_index] = float(frame["timestamp_sec"])
        labels.append(frame.get("label"))

        keypoints_xy = frame.get("keypoints_xy")
        if isinstance(keypoints_xy, list):
            for joint_index, xy in enumerate(keypoints_xy[:num_keypoints]):
                if isinstance(xy, list) and len(xy) >= 2:
                    keypoint[row_index, joint_index, 0] = float(xy[0])
                    keypoint[row_index, joint_index, 1] = float(xy[1])

        keypoints_conf = frame.get("keypoints_conf")
        if isinstance(keypoints_conf, list):
            for joint_index, confidence in enumerate(keypoints_conf[:num_keypoints]):
                if confidence is not None:
                    keypoint_score[row_index, joint_index] = float(confidence)

    return keypoint, keypoint_score, frame_indices, timestamps_sec, labels


def load_session(path: Path) -> SessionData | None:
    identity = parse_session_dir_name(path.parent.name)
    if not include_session(identity):
        return None

    metadata, frames = read_processed_jsonl(path)
    dataset_info = metadata.get("dataset_info", {})
    metadata_session = str(dataset_info.get("session_name", "")).strip().lower()
    metadata_subject = str(dataset_info.get("subject", "")).strip().lower()
    if metadata_session and metadata_session != identity.directory_name.lower():
        raise RuntimeError(
            f"Session mismatch for {path}: directory has {identity.directory_name!r}, "
            f"metadata has {metadata_session!r}."
        )
    if metadata_subject and metadata_subject != identity.subject:
        raise RuntimeError(
            f"Subject mismatch for {path}: directory has {identity.subject!r}, "
            f"metadata has {metadata_subject!r}."
        )

    if not frames:
        raise RuntimeError(f"No detected frame rows found in {path}")

    num_keypoints = infer_num_keypoints(frames, path)
    keypoint, keypoint_score, frame_indices, timestamps_sec, labels = frames_to_arrays(
        frames=frames,
        num_keypoints=num_keypoints,
        path=path,
    )
    return SessionData(
        identity=identity,
        jsonl_path=path,
        raw_jsonl_path=metadata.get("preprocess_info", {}).get("source_jsonl"),
        img_shape=read_image_shape(metadata, path),
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        frame_indices=frame_indices,
        timestamps_sec=timestamps_sec,
        frame_labels=labels,
        segments=metadata_segments(metadata, path),
        num_keypoints=num_keypoints,
    )


def load_sessions(detected_jsonl_root: Path) -> list[SessionData]:
    sessions: list[SessionData] = []
    for path in discover_processed_jsonls(detected_jsonl_root):
        session = load_session(path)
        if session is not None:
            sessions.append(session)
    sessions.sort(key=lambda session: session.identity.recording_index)
    return sessions


def make_frame_dir(
    session: SessionData,
    protocol_id: str,
    sample_token: str,
    label_name: str,
    first_frame_idx: int,
    last_frame_idx: int,
) -> str:
    return sanitize_id(
        f"{DATASET_ID}__{session.identity.directory_name}__{protocol_id}"
        f"__{sample_token}__{label_name}__f{first_frame_idx:06d}_f{last_frame_idx:06d}"
    )


def make_annotation(
    session: SessionData,
    protocol_id: str,
    sample_token: str,
    label_name: str,
    raw_label: Any,
    row_positions: np.ndarray | slice,
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    frame_indices = np.ascontiguousarray(session.frame_indices[row_positions])
    timestamps = np.ascontiguousarray(session.timestamps_sec[row_positions])
    first_frame_idx = int(frame_indices[0])
    last_frame_idx = int(frame_indices[-1])

    keypoint = np.ascontiguousarray(session.keypoint[row_positions][None, ...])
    keypoint_score = np.ascontiguousarray(session.keypoint_score[row_positions][None, ...])

    annotation = {
        "frame_dir": make_frame_dir(
            session=session,
            protocol_id=protocol_id,
            sample_token=sample_token,
            label_name=label_name,
            first_frame_idx=first_frame_idx,
            last_frame_idx=last_frame_idx,
        ),
        "total_frames": int(frame_indices.size),
        "img_shape": session.img_shape,
        "original_shape": session.img_shape,
        "label": LABEL_TO_ID[label_name],
        "label_name": label_name,
        "raw_label": raw_label,
        "subject": session.identity.subject,
        "session_name": session.identity.directory_name,
        "session_index": session.identity.recording_index,
        "session": session.identity.session,
        "session_family": session.identity.session_family,
        "sample_protocol": protocol_id,
        "source_jsonl_path": session.raw_jsonl_path,
        "processed_jsonl_path": str(session.jsonl_path),
        "source_frame_indices": frame_indices,
        "timestamps_sec": timestamps,
        "start_frame": first_frame_idx,
        "end_frame": last_frame_idx,
        "start_timestamp_sec": float(timestamps[0]),
        "end_timestamp_sec": float(timestamps[-1]),
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
    }
    annotation.update(extra_fields)
    return annotation


def segment_labels_for_detected_rows(session: SessionData) -> list[Any]:
    """Map each retained skeleton row to the metadata segment covering it."""

    row_labels: list[Any] = [None] * int(session.frame_indices.size)
    overlaps: list[int] = []

    for segment in session.segments:
        if "start_frame" not in segment or "end_frame" not in segment:
            continue

        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        if segment_start > segment_end:
            raise RuntimeError(
                f"Invalid segment range in {session.jsonl_path}: {segment}"
            )

        left = int(np.searchsorted(session.frame_indices, segment_start, side="left"))
        right = int(np.searchsorted(session.frame_indices, segment_end, side="right"))
        for row_index in range(left, right):
            if row_labels[row_index] is not None:
                overlaps.append(int(session.frame_indices[row_index]))
            row_labels[row_index] = segment.get("label")

    if overlaps:
        preview = ", ".join(str(item) for item in overlaps[:10])
        raise RuntimeError(
            f"Overlapping annotation segments in {session.jsonl_path}; "
            f"example frame_idx values: {preview}"
        )

    return row_labels


def triangular_soft_target(
    raw_frame_labels: list[Any],
    normalized_weights: np.ndarray,
) -> tuple[list[float], float, int, Counter]:
    if len(raw_frame_labels) != len(normalized_weights):
        raise ValueError(
            "Frame-label count must match triangular target weight count."
        )

    class_mass = np.zeros(len(FINAL_LABELS), dtype=np.float64)
    valid_weight_mass = 0.0
    valid_frame_count = 0
    invalid_frame_labels: Counter = Counter()

    for raw_label, weight in zip(raw_frame_labels, normalized_weights):
        label_name = normalize_label(raw_label)
        if label_name is None:
            invalid_frame_labels[str(raw_label)] += 1
            continue

        valid_frame_count += 1
        valid_weight_mass += float(weight)
        class_mass[LABEL_TO_ID[label_name]] += float(weight)

    if valid_weight_mass <= 0:
        raise RuntimeError(
            "Triangular target has zero valid label mass. This should not "
            "happen when the center frame has a valid final label."
        )

    class_mass /= valid_weight_mass
    return (
        [float(value) for value in class_mass.astype(np.float32)],
        float(valid_weight_mass),
        valid_frame_count,
        invalid_frame_labels,
    )


def is_open_ended_tail_segment(session: SessionData, segment_index: int) -> bool:
    return (
        session.identity.directory_name.lower() in OPEN_ENDED_TAIL_RECORDINGS
        and segment_index == len(session.segments) - 1
    )


def build_activity_aligned(
    sessions: list[SessionData],
    min_detected_frames: int,
    max_adjacent_gap_sec: float,
) -> ProtocolResult:
    protocol_id = "activity_aligned"
    annotations: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "protocol_id": protocol_id,
        "source_recordings": len(sessions),
        "min_detected_frames_per_segment": min_detected_frames,
        "max_adjacent_gap_sec": max_adjacent_gap_sec,
        "open_ended_tail_recordings": sorted(OPEN_ENDED_TAIL_RECORDINGS),
        "source_segments": 0,
        "segments_checked_for_gap": 0,
        "samples_created": 0,
        "max_observed_adjacent_gap_sec": 0.0,
        "dropped_segments_by_reason": Counter(),
        "dropped_segments_by_label": Counter(),
        "dropped_open_ended_tail_by_session": Counter(),
    }

    for session in sessions:
        for segment_index, segment in enumerate(session.segments):
            stats["source_segments"] += 1
            raw_label = segment.get("label")

            if is_open_ended_tail_segment(session, segment_index):
                stats["dropped_segments_by_reason"]["open_ended_tail_interval"] += 1
                stats["dropped_segments_by_label"][str(raw_label)] += 1
                stats["dropped_open_ended_tail_by_session"][
                    session.identity.directory_name
                ] += 1
                continue

            label_name = normalize_label(raw_label)
            if label_name is None:
                stats["dropped_segments_by_reason"]["label_not_in_final_set"] += 1
                stats["dropped_segments_by_label"][str(raw_label)] += 1
                continue

            if "start_frame" not in segment or "end_frame" not in segment:
                raise RuntimeError(
                    f"Segment missing start_frame/end_frame in {session.jsonl_path}: {segment}"
                )

            segment_start = int(segment["start_frame"])
            segment_end = int(segment["end_frame"])
            if segment_start > segment_end:
                raise RuntimeError(
                    f"Invalid segment range in {session.jsonl_path}: {segment}"
                )

            mask = (session.frame_indices >= segment_start) & (
                session.frame_indices <= segment_end
            )
            row_positions = np.flatnonzero(mask)
            if row_positions.size == 0:
                stats["dropped_segments_by_reason"]["no_valid_detection_in_segment"] += 1
                stats["dropped_segments_by_label"][str(raw_label)] += 1
                continue
            if row_positions.size < min_detected_frames:
                stats["dropped_segments_by_reason"]["below_min_detected_frames"] += 1
                stats["dropped_segments_by_label"][str(raw_label)] += 1
                continue

            timestamps = session.timestamps_sec[row_positions]
            adjacent_gaps = np.diff(timestamps)
            gmax = float(np.max(adjacent_gaps))
            stats["segments_checked_for_gap"] += 1
            stats["max_observed_adjacent_gap_sec"] = max(
                float(stats["max_observed_adjacent_gap_sec"]),
                gmax,
            )
            if gmax > max_adjacent_gap_sec:
                stats["dropped_segments_by_reason"]["max_adjacent_gap"] += 1
                stats["dropped_segments_by_label"][str(raw_label)] += 1
                continue

            annotation = make_annotation(
                session=session,
                protocol_id=protocol_id,
                sample_token=f"seg{segment_index:04d}",
                label_name=label_name,
                raw_label=raw_label,
                row_positions=row_positions,
                extra_fields={
                    "source_segment_id": segment_index,
                    "segment_start_frame": segment_start,
                    "segment_end_frame": segment_end,
                    "segment_length_frames": int(
                        segment.get("length_frames", segment_end - segment_start + 1)
                    ),
                    "detected_frames_in_segment": int(row_positions.size),
                    "max_adjacent_gap_sec": gmax,
                    "segment_span_sec": float(timestamps[-1] - timestamps[0]),
                },
            )
            annotations.append(annotation)

    annotations.sort(key=lambda item: item["frame_dir"])
    stats["samples_created"] = len(annotations)
    return ProtocolResult(protocol_id=protocol_id, annotations=annotations, stats=stats)


def build_continuous_window(
    sessions: list[SessionData],
    window_size: int,
    stride: int,
    max_adjacent_gap_sec: float,
    max_window_span_sec: float,
    triangular_soft_labels: bool = False,
) -> ProtocolResult:
    base_protocol_id = f"continuous_window_w{window_size}_s{stride}"
    protocol_id = (
        f"{base_protocol_id}_triangular"
        if triangular_soft_labels
        else base_protocol_id
    )
    triangular_weights = (
        bartlett_triangular_weights(window_size) if triangular_soft_labels else None
    )
    annotations: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "protocol_id": protocol_id,
        "base_protocol_id": base_protocol_id,
        "source_recordings": len(sessions),
        "window_size": window_size,
        "stride": stride,
        "center_offset": window_size // 2,
        "max_adjacent_gap_sec": max_adjacent_gap_sec,
        "max_window_span_sec": max_window_span_sec,
        "candidate_windows": 0,
        "samples_created": 0,
        "dropped_windows_by_reason": Counter(),
        "dropped_windows_by_label": Counter(),
        "tail_rows_dropped_by_session": {},
        "triangular_soft_labels": triangular_soft_labels,
    }
    if triangular_soft_labels:
        assert triangular_weights is not None
        stats["triangular_target"] = {
            "field": "label_soft_triangular",
            "num_classes": len(FINAL_LABELS),
            "label_source": "annotation_info.segments matched by original frame_idx",
            "weight_formula": "w_t proportional to 1 - abs(2*t - (L - 1)) / (L - 1)",
            "renormalization_rule": (
                "Frames whose labels are outside the final nine classes are "
                "ignored and the remaining triangular mass is renormalized."
            ),
            "normalized_weights": [float(value) for value in triangular_weights],
        }
        stats["triangular_windows_with_invalid_frame_labels"] = 0
        stats["triangular_windows_with_renormalization"] = 0
        stats["triangular_invalid_frame_labels"] = Counter()
        stats["triangular_invalid_windows_by_label"] = Counter()
        stats["triangular_valid_weight_mass_min"] = 1.0
        stats["triangular_valid_weight_mass_max"] = 0.0
        stats["triangular_valid_frame_count_min"] = window_size
        stats["triangular_valid_frame_count_max"] = 0

    for session in sessions:
        segment_row_labels = (
            segment_labels_for_detected_rows(session)
            if triangular_soft_labels
            else None
        )
        row_count = int(session.frame_indices.size)
        if row_count < window_size:
            stats["tail_rows_dropped_by_session"][session.identity.directory_name] = row_count
            continue

        last_start = row_count - window_size
        final_full_start = (last_start // stride) * stride
        tail_start = final_full_start + stride
        tail_rows = max(0, row_count - tail_start)
        stats["tail_rows_dropped_by_session"][session.identity.directory_name] = tail_rows

        session_window_index = 0
        for row_start in range(0, row_count - window_size + 1, stride):
            row_end = row_start + window_size
            stats["candidate_windows"] += 1
            timestamps = session.timestamps_sec[row_start:row_end]
            adjacent_gaps = np.diff(timestamps)
            gmax = float(np.max(adjacent_gaps))
            window_span = float(timestamps[-1] - timestamps[0])

            failed_validity = False
            if gmax > max_adjacent_gap_sec:
                stats["dropped_windows_by_reason"]["max_adjacent_gap"] += 1
                failed_validity = True
            if window_span > max_window_span_sec:
                stats["dropped_windows_by_reason"]["max_window_span"] += 1
                failed_validity = True
            if failed_validity:
                stats["dropped_windows_by_reason"]["validity_any"] += 1
                continue

            center_row = row_start + (window_size // 2)
            raw_label = session.frame_labels[center_row]
            label_name = normalize_label(raw_label)
            if label_name is None:
                stats["dropped_windows_by_reason"]["center_label_not_in_final_set"] += 1
                stats["dropped_windows_by_label"][str(raw_label)] += 1
                continue

            row_slice = slice(row_start, row_end)
            extra_fields = {
                "window_row_start": row_start,
                "window_row_end_exclusive": row_end,
                "window_size": window_size,
                "stride": stride,
                "center_row_offset": window_size // 2,
                "center_source_frame": int(session.frame_indices[center_row]),
                "center_timestamp_sec": float(session.timestamps_sec[center_row]),
                "center_raw_label": raw_label,
                "max_adjacent_gap_sec": gmax,
                "window_span_sec": window_span,
            }
            if triangular_soft_labels:
                assert triangular_weights is not None
                assert segment_row_labels is not None
                (
                    label_soft_triangular,
                    valid_weight_mass,
                    valid_frame_count,
                    invalid_frame_labels,
                ) = triangular_soft_target(
                    raw_frame_labels=segment_row_labels[row_start:row_end],
                    normalized_weights=triangular_weights,
                )
                invalid_frame_count = int(window_size - valid_frame_count)
                if invalid_frame_count > 0:
                    stats["triangular_windows_with_invalid_frame_labels"] += 1
                    stats["triangular_invalid_frame_labels"].update(
                        invalid_frame_labels
                    )
                    for invalid_label in invalid_frame_labels:
                        stats["triangular_invalid_windows_by_label"][invalid_label] += 1
                if valid_weight_mass < 1.0 - 1e-6:
                    stats["triangular_windows_with_renormalization"] += 1

                stats["triangular_valid_weight_mass_min"] = min(
                    float(stats["triangular_valid_weight_mass_min"]),
                    valid_weight_mass,
                )
                stats["triangular_valid_weight_mass_max"] = max(
                    float(stats["triangular_valid_weight_mass_max"]),
                    valid_weight_mass,
                )
                stats["triangular_valid_frame_count_min"] = min(
                    int(stats["triangular_valid_frame_count_min"]),
                    valid_frame_count,
                )
                stats["triangular_valid_frame_count_max"] = max(
                    int(stats["triangular_valid_frame_count_max"]),
                    valid_frame_count,
                )
                extra_fields.update(
                    {
                        "label_soft_triangular": label_soft_triangular,
                        "triangular_valid_weight_mass": valid_weight_mass,
                        "triangular_valid_frame_count": valid_frame_count,
                        "triangular_invalid_frame_count": invalid_frame_count,
                        "triangular_label_source": (
                            "annotation_info.segments matched by original frame_idx"
                        ),
                    }
                )

            annotation = make_annotation(
                session=session,
                protocol_id=protocol_id,
                sample_token=f"win{session_window_index:06d}",
                label_name=label_name,
                raw_label=raw_label,
                row_positions=row_slice,
                extra_fields=extra_fields,
            )
            annotations.append(annotation)
            session_window_index += 1

    annotations.sort(key=lambda item: item["frame_dir"])
    stats["samples_created"] = len(annotations)
    return ProtocolResult(protocol_id=protocol_id, annotations=annotations, stats=stats)


def validate_folds(available_subjects: set[str]) -> None:
    for fold_name, fold in FOLDS.items():
        assigned: dict[str, str] = {}
        for split_name, subjects in fold.items():
            for subject in subjects:
                if subject in assigned:
                    raise RuntimeError(
                        f"{fold_name}: subject {subject!r} appears in both "
                        f"{assigned[subject]!r} and {split_name!r}."
                    )
                assigned[subject] = split_name

        missing = available_subjects - set(assigned)
        unknown = set(assigned) - available_subjects
        if missing or unknown:
            raise RuntimeError(
                f"{fold_name}: fold subjects do not match available subjects. "
                f"Missing={sorted(missing)}, unknown={sorted(unknown)}."
            )


def make_split(annotations: list[dict[str, Any]], fold: dict[str, list[str]]) -> dict[str, list[str]]:
    subject_to_split: dict[str, str] = {}
    for split_name, subjects in fold.items():
        for subject in subjects:
            subject_to_split[subject] = split_name

    split = {"train": [], "val": [], "calib": [], "test": []}
    for annotation in annotations:
        subject = str(annotation["subject"])
        split_name = subject_to_split.get(subject)
        if split_name is None:
            raise RuntimeError(f"Subject {subject!r} is not assigned in fold.")
        split[split_name].append(annotation["frame_dir"])
    return split


def annotation_by_id(annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {annotation["frame_dir"]: annotation for annotation in annotations}


def count_samples_by_label(annotations: list[dict[str, Any]]) -> Counter:
    return Counter(str(annotation["label_name"]) for annotation in annotations)


def count_samples_by_subject(annotations: list[dict[str, Any]]) -> Counter:
    return Counter(str(annotation["subject"]) for annotation in annotations)


def count_for_ids(
    lookup: dict[str, dict[str, Any]],
    ids: list[str],
    field_name: str,
) -> Counter:
    return Counter(str(lookup[item_id][field_name]) for item_id in ids)


def counter_to_regular(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: counter_to_regular(item) for key, item in value.items()}
    if isinstance(value, list):
        return [counter_to_regular(item) for item in value]
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(counter_to_regular(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def preprocess_stats_rows(records: list[PreprocessRecord]) -> list[dict[str, Any]]:
    return [
        {
            "session_dir": record.session_dir,
            "subject": record.subject,
            "session": record.session,
            "session_family": record.session_family,
            "source_frame_rows": record.source_frame_rows,
            "kept_frame_rows": record.kept_frame_rows,
            "removed_false_detection_rows": record.removed_false_detection_rows,
            "source_jsonl": record.source_jsonl,
            "processed_jsonl": record.processed_jsonl,
        }
        for record in records
    ]


def save_preprocess_stats(output_root: Path, records: list[PreprocessRecord]) -> None:
    stats_dir = output_root / "stats"
    write_csv(
        stats_dir / "preprocess_jsonl.csv",
        [
            "session_dir",
            "subject",
            "session",
            "session_family",
            "source_frame_rows",
            "kept_frame_rows",
            "removed_false_detection_rows",
            "source_jsonl",
            "processed_jsonl",
        ],
        preprocess_stats_rows(records),
    )
    write_json(
        stats_dir / "preprocess_summary.json",
        {
            "processed_recordings": len(records),
            "source_frame_rows": sum(record.source_frame_rows for record in records),
            "kept_frame_rows": sum(record.kept_frame_rows for record in records),
            "removed_false_detection_rows": sum(
                record.removed_false_detection_rows for record in records
            ),
            "scope": "included recordings by default unless --preprocess-scope all is used",
        },
    )


def save_label_map(output_root: Path) -> None:
    label_map_path = output_root / "stats" / "label_map.csv"
    write_csv(
        label_map_path,
        ["label_id", "label_name"],
        [{"label_id": idx, "label_name": label} for idx, label in enumerate(FINAL_LABELS)],
    )


def protocol_summary(
    result: ProtocolResult,
    fold_splits: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    annotations = result.annotations
    lookup = annotation_by_id(annotations)
    available_subjects = sorted(count_samples_by_subject(annotations))
    folds: dict[str, Any] = {}

    for fold_name, split in fold_splits.items():
        fold_payload: dict[str, Any] = {
            "subjects": FOLDS[fold_name],
            "num_samples": {split_name: len(ids) for split_name, ids in split.items()},
            "samples_per_class": {},
            "samples_per_subject": {},
        }
        for split_name, ids in split.items():
            class_counts = count_for_ids(lookup, ids, "label_name")
            subject_counts = count_for_ids(lookup, ids, "subject")
            fold_payload["samples_per_class"][split_name] = {
                label: class_counts.get(label, 0) for label in FINAL_LABELS
            }
            fold_payload["samples_per_subject"][split_name] = {
                subject: subject_counts.get(subject, 0) for subject in available_subjects
            }
        folds[fold_name] = fold_payload

    return {
        "dataset_id": DATASET_ID,
        "protocol_id": result.protocol_id,
        "label_to_id": LABEL_TO_ID,
        "num_annotations": len(annotations),
        "samples_per_class": {
            label: count_samples_by_label(annotations).get(label, 0)
            for label in FINAL_LABELS
        },
        "samples_per_subject": dict(sorted(count_samples_by_subject(annotations).items())),
        "protocol_stats": result.stats,
        "folds": folds,
    }


def save_protocol_stats(
    output_root: Path,
    result: ProtocolResult,
    fold_splits: dict[str, dict[str, list[str]]],
) -> None:
    stats_dir = output_root / "stats" / result.protocol_id
    stats_dir.mkdir(parents=True, exist_ok=True)

    annotations = result.annotations
    lookup = annotation_by_id(annotations)
    subjects = sorted(count_samples_by_subject(annotations))

    write_json(stats_dir / "summary.json", protocol_summary(result, fold_splits))

    subject_counts = count_samples_by_subject(annotations)
    write_csv(
        stats_dir / "samples_by_subject.csv",
        ["subject", "num_samples"],
        [{"subject": subject, "num_samples": subject_counts.get(subject, 0)} for subject in subjects],
    )

    label_counts = count_samples_by_label(annotations)
    write_csv(
        stats_dir / "samples_by_class.csv",
        ["label_id", "label_name", "num_samples"],
        [
            {
                "label_id": LABEL_TO_ID[label],
                "label_name": label,
                "num_samples": label_counts.get(label, 0),
            }
            for label in FINAL_LABELS
        ],
    )

    subject_class_counts = Counter(
        (str(annotation["subject"]), str(annotation["label_name"]))
        for annotation in annotations
    )
    write_csv(
        stats_dir / "samples_by_subject_class.csv",
        ["subject", "label_id", "label_name", "num_samples"],
        [
            {
                "subject": subject,
                "label_id": LABEL_TO_ID[label],
                "label_name": label,
                "num_samples": subject_class_counts.get((subject, label), 0),
            }
            for subject in subjects
            for label in FINAL_LABELS
        ],
    )

    split_rows: list[dict[str, Any]] = []
    split_class_rows: list[dict[str, Any]] = []
    split_subject_rows: list[dict[str, Any]] = []
    for fold_name, split in fold_splits.items():
        for split_name, ids in split.items():
            split_rows.append(
                {
                    "fold": fold_name,
                    "split": split_name,
                    "num_samples": len(ids),
                    "subjects": ",".join(FOLDS[fold_name][split_name]),
                }
            )

            class_counts = count_for_ids(lookup, ids, "label_name")
            for label in FINAL_LABELS:
                split_class_rows.append(
                    {
                        "fold": fold_name,
                        "split": split_name,
                        "label_id": LABEL_TO_ID[label],
                        "label_name": label,
                        "num_samples": class_counts.get(label, 0),
                    }
                )

            subject_counts = count_for_ids(lookup, ids, "subject")
            for subject in subjects:
                split_subject_rows.append(
                    {
                        "fold": fold_name,
                        "split": split_name,
                        "subject": subject,
                        "num_samples": subject_counts.get(subject, 0),
                    }
                )

    write_csv(
        stats_dir / "samples_by_fold_split.csv",
        ["fold", "split", "num_samples", "subjects"],
        split_rows,
    )
    write_csv(
        stats_dir / "samples_by_fold_split_class.csv",
        ["fold", "split", "label_id", "label_name", "num_samples"],
        split_class_rows,
    )
    write_csv(
        stats_dir / "samples_by_fold_split_subject.csv",
        ["fold", "split", "subject", "num_samples"],
        split_subject_rows,
    )


def save_pkl(path: Path, annotations: list[dict[str, Any]], split: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {"split": split, "annotations": annotations},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def save_protocol_pkls(output_root: Path, result: ProtocolResult) -> dict[str, dict[str, list[str]]]:
    annotations = result.annotations
    subjects = set(count_samples_by_subject(annotations))
    validate_folds(subjects)

    fold_splits: dict[str, dict[str, list[str]]] = {}
    pkl_dir = output_root / "pyskl" / result.protocol_id
    for fold_name, fold in FOLDS.items():
        split = make_split(annotations, fold)
        fold_splits[fold_name] = split
        pkl_path = pkl_dir / f"{DATASET_ID}_{result.protocol_id}_{fold_name}.pkl"
        save_pkl(pkl_path, annotations, split)
        write_json(
            pkl_path.with_name(f"{pkl_path.stem}_summary.json"),
            {
                "dataset_id": DATASET_ID,
                "protocol_id": result.protocol_id,
                "fold": fold_name,
                "subjects": fold,
                "label_to_id": LABEL_TO_ID,
                "num_annotations": len(annotations),
                "num_samples_by_split": {
                    split_name: len(ids) for split_name, ids in split.items()
                },
                "pkl_path": str(pkl_path),
            },
        )

    return fold_splits


def save_protocol(output_root: Path, result: ProtocolResult) -> None:
    if not result.annotations:
        raise RuntimeError(f"No annotations were created for {result.protocol_id}")
    fold_splits = save_protocol_pkls(output_root, result)
    save_protocol_stats(output_root, result, fold_splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RADAR v4 YOLO26x-pose rerun datasets."
    )
    parser.add_argument(
        "--raw-jsonl-root",
        type=Path,
        default=RAW_JSONL_ROOT,
        help="Source root containing raw YOLO26x-pose JSONL session folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root for rerun artifacts: detected JSONLs, PKLs, and stats.",
    )
    parser.add_argument(
        "--detected-jsonl-root",
        type=Path,
        default=None,
        help="Detected-only JSONL root. Defaults to <output-root>/detected_jsonl.",
    )
    parser.add_argument(
        "--preprocess-scope",
        choices=["included", "all"],
        default="included",
        help=(
            "Which raw JSONL recordings to preprocess. 'included' means only "
            "fall/sit/laysofa session families used by the datasets."
        ),
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing detected-only JSONLs instead of regenerating them.",
    )
    parser.add_argument(
        "--protocol",
        choices=[
            "all",
            "both",
            "activity_aligned",
            "continuous_window",
            "continuous_window_triangular",
        ],
        default="all",
        help="Dataset protocol to build.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Continuous-window size in detected skeleton rows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Continuous-window stride in detected skeleton rows.",
    )
    parser.add_argument(
        "--max-adjacent-gap-sec",
        type=float,
        default=0.5,
        help=(
            "Reject activity-aligned segments and continuous windows with any "
            "adjacent timestamp gap above this value."
        ),
    )
    parser.add_argument(
        "--activity-min-detected-frames",
        type=int,
        default=ACTIVITY_MIN_DETECTED_FRAMES,
        help="Reject activity-aligned segments with fewer detected skeleton rows.",
    )
    parser.add_argument(
        "--max-window-span-sec",
        type=float,
        default=2.5,
        help="Reject continuous windows whose first-to-last timestamp span exceeds this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.window_size % 2 != 0:
        raise ValueError("--window-size must be even so center offset is start + window_size / 2.")
    if args.max_adjacent_gap_sec <= 0:
        raise ValueError("--max-adjacent-gap-sec must be positive.")
    if args.activity_min_detected_frames < 2:
        raise ValueError("--activity-min-detected-frames must be at least 2.")

    output_root = args.output_root
    detected_jsonl_root = args.detected_jsonl_root or (output_root / "detected_jsonl")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.skip_preprocess:
        print(f"[INFO] Reusing detected-only JSONLs from {detected_jsonl_root}")
    else:
        print(
            f"[INFO] Preprocessing JSONLs from {args.raw_jsonl_root} "
            f"to {detected_jsonl_root} (scope={args.preprocess_scope})"
        )
        preprocess_records = preprocess_jsonl_files(
            raw_jsonl_root=args.raw_jsonl_root,
            detected_jsonl_root=detected_jsonl_root,
            scope=args.preprocess_scope,
        )
        save_preprocess_stats(output_root, preprocess_records)
        print(f"[INFO] Preprocessed {len(preprocess_records)} recordings.")

    save_label_map(output_root)

    sessions = load_sessions(detected_jsonl_root)
    if not sessions:
        raise RuntimeError("No included detected-only sessions were loaded.")
    subjects = sorted({session.identity.subject for session in sessions})
    print(f"[INFO] Loaded {len(sessions)} included sessions for subjects: {subjects}")

    if args.protocol in {"all", "both", "activity_aligned"}:
        print("[INFO] Building activity-aligned dataset.")
        save_protocol(
            output_root,
            build_activity_aligned(
                sessions=sessions,
                min_detected_frames=args.activity_min_detected_frames,
                max_adjacent_gap_sec=args.max_adjacent_gap_sec,
            ),
        )

    if args.protocol in {"all", "both", "continuous_window"}:
        print("[INFO] Building continuous-window dataset.")
        save_protocol(
            output_root,
            build_continuous_window(
                sessions=sessions,
                window_size=args.window_size,
                stride=args.stride,
                max_adjacent_gap_sec=args.max_adjacent_gap_sec,
                max_window_span_sec=args.max_window_span_sec,
            ),
        )

    if args.protocol in {"all", "continuous_window_triangular"}:
        print("[INFO] Building continuous-window triangular soft-label dataset.")
        save_protocol(
            output_root,
            build_continuous_window(
                sessions=sessions,
                window_size=args.window_size,
                stride=args.stride,
                max_adjacent_gap_sec=args.max_adjacent_gap_sec,
                max_window_span_sec=args.max_window_span_sec,
                triangular_soft_labels=True,
            ),
        )

    print(f"[DONE] Rerun dataset artifacts written under {output_root}")


if __name__ == "__main__":
    main()
