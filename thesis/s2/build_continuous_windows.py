"""Build Study 2 continuous-window PYSKL annotations and manifests."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        CENTER_OFFSET,
        DEFAULT_ETAS,
        DEFAULT_JSONL_ROOT,
        DEFAULT_PKL_DIR,
        DEFAULT_S2_PKL,
        FPS,
        LABELS,
        LABEL_TO_ID,
        STRIDE,
        WINDOW_SIZE,
        S2FoldSpec,
        clean_group,
        clean_label,
        discover_s2_folds,
        eta_slug,
        protocol_metadata,
        write_json,
        write_rows_csv,
    )
except ImportError:
    from common import (
        CENTER_OFFSET,
        DEFAULT_ETAS,
        DEFAULT_JSONL_ROOT,
        DEFAULT_PKL_DIR,
        DEFAULT_S2_PKL,
        FPS,
        LABELS,
        LABEL_TO_ID,
        STRIDE,
        WINDOW_SIZE,
        S2FoldSpec,
        clean_group,
        clean_label,
        discover_s2_folds,
        eta_slug,
        protocol_metadata,
        write_json,
        write_rows_csv,
    )

from infer_hpe_jsonl_timeline import load_jsonl_records, metadata_fps, read_jsonl_frame_grid  # noqa: E402
from stats_common import read_frame_label_rows, rows_to_segments  # noqa: E402
from thesis.e2.common import is_walk_session, subject_from_session_name  # noqa: E402


@dataclass
class ContinuousSession:
    jsonl_path: Path
    recording_id: str
    subject_id: str
    img_shape: tuple[int, int]
    fps: float
    keypoint: np.ndarray
    keypoint_score: np.ndarray
    selected_detection: np.ndarray
    timestamps: np.ndarray
    raw_labels: list[str | None]
    clean_label_ids: np.ndarray
    label_groups: list[str]
    segments: list[dict[str, Any]]
    label_source: str
    metadata: dict[str, Any]
    frame_records: dict[int, dict[str, Any]]

    @property
    def total_frames(self) -> int:
        return int(self.keypoint.shape[0])


@dataclass
class BuildStats:
    jsonl_files_seen: int = 0
    jsonl_files_used: int = 0
    walk_sessions_skipped: int = 0
    windows_created: int = 0
    dropped_windows_by_reason: Counter = field(default_factory=Counter)
    label_source_counts: Counter = field(default_factory=Counter)
    label_mismatches: list[dict[str, Any]] = field(default_factory=list)


def sanitize_id(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def one_hot(label_id: int) -> np.ndarray:
    target = np.zeros(len(LABELS), dtype=np.float32)
    target[label_id] = 1.0
    return target


def assert_probability_distribution(name: str, target: np.ndarray, tolerance: float = 1e-5) -> None:
    if target.shape != (len(LABELS),):
        raise ValueError(f"{name} shape {target.shape} != ({len(LABELS)},)")
    if not np.issubdtype(target.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {target.dtype}")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(target < -tolerance):
        raise ValueError(f"{name} contains negative values: {target}")
    if abs(float(target.sum()) - 1.0) >= tolerance:
        raise ValueError(f"{name} sum {float(target.sum())} is not one")


def annotation_segments_from_origin(origin_root: Path, recording_id: str, total_frames: int) -> list[dict[str, Any]]:
    label_path = origin_root / recording_id / "frame_labels.csv"
    if not label_path.exists():
        return []
    return rows_to_segments(read_frame_label_rows(label_path), final_frame=total_frames - 1)


def metadata_segments(metadata: dict[str, Any], total_frames: int) -> list[dict[str, Any]]:
    segments = metadata.get("annotation_info", {}).get("segments", [])
    output = []
    if not isinstance(segments, list):
        return output
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        start = int(segment.get("start_frame", 0))
        end = segment.get("end_frame")
        end = total_frames - 1 if end is None else int(end)
        start = max(0, start)
        end = min(total_frames - 1, end)
        if start > end:
            continue
        item = dict(segment)
        item["source_segment_id"] = int(segment.get("source_segment_id", index))
        item["start_frame"] = start
        item["end_frame"] = end
        item["length_frames"] = end - start + 1
        output.append(item)
    return output


def frame_record_segments(frame_records: dict[int, dict[str, Any]], total_frames: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_label: str | None = None
    current_group = "unlabeled"
    current_start: int | None = None

    def close(end_frame: int) -> None:
        nonlocal current_label, current_start, current_group
        if current_label is None or current_start is None:
            return
        segments.append(
            {
                "source_segment_id": len(segments),
                "start_frame": current_start,
                "end_frame": end_frame,
                "label": current_label,
                "label_group": current_group,
                "length_frames": end_frame - current_start + 1,
            }
        )

    last_frame = -1
    for frame_idx in range(total_frames):
        record = frame_records.get(frame_idx, {})
        label = record.get("label") if isinstance(record, dict) else None
        label = label.strip() if isinstance(label, str) and label.strip() else None
        group = record.get("label_group", "unlabeled") if isinstance(record, dict) else "unlabeled"
        if label != current_label:
            if last_frame >= 0:
                close(last_frame)
            current_label = label
            current_group = str(group)
            current_start = frame_idx if label is not None else None
        last_frame = frame_idx
    if last_frame >= 0:
        close(last_frame)
    return segments


def choose_segments(
    metadata: dict[str, Any],
    frame_records: dict[int, dict[str, Any]],
    origin_root: Path,
    recording_id: str,
    total_frames: int,
    label_source: str,
) -> tuple[list[dict[str, Any]], str]:
    origin = annotation_segments_from_origin(origin_root, recording_id, total_frames)
    meta = metadata_segments(metadata, total_frames)
    frame = frame_record_segments(frame_records, total_frames)

    if label_source == "origin":
        if origin:
            return origin, "origin"
        if meta:
            return meta, "jsonl_metadata_fallback"
        return frame, "jsonl_frame_fallback"
    if label_source == "jsonl":
        if meta:
            return meta, "jsonl_metadata"
        return frame, "jsonl_frame"
    if origin:
        return origin, "origin"
    if meta:
        return meta, "jsonl_metadata_fallback"
    return frame, "jsonl_frame_fallback"


def build_frame_label_arrays(
    segments: list[dict[str, Any]],
    total_frames: int,
) -> tuple[list[str | None], np.ndarray, list[str]]:
    raw_labels: list[str | None] = [None] * total_frames
    clean_label_ids = np.full(total_frames, -1, dtype=np.int16)
    label_groups = ["unlabeled"] * total_frames

    for segment in segments:
        start = max(0, int(segment["start_frame"]))
        end = min(total_frames - 1, int(segment["end_frame"]))
        if start > end:
            continue

        raw_label = segment.get("label")
        raw_label_text = None if raw_label is None else str(raw_label).strip()
        cleaned = clean_label(raw_label)
        group = clean_group(segment.get("label_group"), cleaned)
        label_id = -1 if cleaned is None else LABEL_TO_ID[cleaned]
        for frame_idx in range(start, end + 1):
            raw_labels[frame_idx] = raw_label_text
            clean_label_ids[frame_idx] = label_id
            label_groups[frame_idx] = group

    return raw_labels, clean_label_ids, label_groups


def load_continuous_session(
    jsonl_path: Path,
    origin_root: Path,
    label_source: str,
) -> ContinuousSession:
    metadata, frame_records = load_jsonl_records(jsonl_path)
    grid, img_shape = read_jsonl_frame_grid(
        jsonl_path=jsonl_path,
        kp_threshold=0.0,
        max_frames=None,
        trust_metadata_count=False,
    )
    recording_id = jsonl_path.parent.name
    dataset_info = metadata.get("dataset_info", {}) if isinstance(metadata, dict) else {}
    subject_id = str(dataset_info.get("subject") or subject_from_session_name(recording_id)).lower()
    fps = metadata_fps(metadata)

    timestamps = np.zeros(grid.total_frames, dtype=np.float64)
    for frame_idx in range(grid.total_frames):
        record = frame_records.get(frame_idx)
        if isinstance(record, dict) and record.get("timestamp_sec") is not None:
            timestamps[frame_idx] = float(record["timestamp_sec"])
        else:
            timestamps[frame_idx] = frame_idx / fps

    segments, used_label_source = choose_segments(
        metadata=metadata,
        frame_records=frame_records,
        origin_root=origin_root,
        recording_id=recording_id,
        total_frames=grid.total_frames,
        label_source=label_source,
    )
    raw_labels, clean_label_ids, label_groups = build_frame_label_arrays(
        segments=segments,
        total_frames=grid.total_frames,
    )
    return ContinuousSession(
        jsonl_path=jsonl_path,
        recording_id=recording_id,
        subject_id=subject_id,
        img_shape=img_shape,
        fps=fps,
        keypoint=grid.keypoint,
        keypoint_score=grid.keypoint_score,
        selected_detection=grid.selected_detection,
        timestamps=timestamps,
        raw_labels=raw_labels,
        clean_label_ids=clean_label_ids,
        label_groups=label_groups,
        segments=segments,
        label_source=used_label_source,
        metadata=metadata,
        frame_records=frame_records,
    )


def window_starts(total_frames: int, window_size: int, stride: int) -> list[int]:
    if total_frames < window_size:
        return []
    return list(range(0, total_frames - window_size + 1, stride))


def has_large_timestamp_gap(timestamps: np.ndarray, start: int, end: int, max_gap_sec: float | None) -> bool:
    if max_gap_sec is None:
        return False
    window_timestamps = timestamps[start:end + 1]
    if window_timestamps.shape[0] < 2:
        return False
    diffs = np.diff(window_timestamps)
    finite = np.isfinite(diffs)
    return bool(np.any(finite & (diffs > max_gap_sec)))


def temporal_distribution(
    session: ContinuousSession,
    start: int,
    end: int,
    center: int,
) -> np.ndarray:
    weights = np.zeros(end - start + 1, dtype=np.float64)
    center_timestamp = float(session.timestamps[center])
    half_window_duration = CENTER_OFFSET / session.fps if session.fps > 0 else CENTER_OFFSET / FPS
    if half_window_duration <= 0:
        half_window_duration = 1.0

    for offset, frame_idx in enumerate(range(start, end + 1)):
        distance = abs(float(session.timestamps[frame_idx]) - center_timestamp)
        weights[offset] = max(0.0, 1.0 - distance / half_window_duration)

    target = np.zeros(len(LABELS), dtype=np.float64)
    denominator = 0.0
    for offset, frame_idx in enumerate(range(start, end + 1)):
        label_id = int(session.clean_label_ids[frame_idx])
        if label_id < 0:
            continue
        weight = float(weights[offset])
        target[label_id] += weight
        denominator += weight

    if denominator <= 0:
        center_label_id = int(session.clean_label_ids[center])
        if center_label_id < 0:
            raise ValueError("Cannot build q_temporal without a valid center label")
        return one_hot(center_label_id)

    target = (target / denominator).astype(np.float32)
    assert_probability_distribution("q_temporal", target)
    return target


def final_targets(label_id: int, q_temporal: np.ndarray, etas: tuple[float, ...]) -> dict[str, np.ndarray]:
    targets = {}
    hard = one_hot(label_id)
    for eta in etas:
        target = ((1.0 - eta) * hard + eta * q_temporal).astype(np.float32)
        assert_probability_distribution(f"target_probs_{eta_slug(eta)}", target)
        targets[f"target_probs_{eta_slug(eta)}"] = target
    return targets


def build_annotation(
    session: ContinuousSession,
    start: int,
    window_size: int,
    etas: tuple[float, ...],
) -> dict[str, Any]:
    end = start + window_size - 1
    center = start + CENTER_OFFSET
    label_id = int(session.clean_label_ids[center])
    label_name = LABELS[label_id]
    q_temporal = temporal_distribution(session, start=start, end=end, center=center)
    targets = final_targets(label_id, q_temporal, etas)
    frame_dir = (
        f"{sanitize_id(session.recording_id)}"
        f"__s{start:06d}_e{end:06d}_c{center:06d}"
    )

    keypoint = session.keypoint[start:end + 1][None, ...].copy()
    keypoint_score = session.keypoint_score[start:end + 1][None, ...].copy()
    per_frame_label_ids = session.clean_label_ids[start:end + 1].copy()

    return {
        "frame_dir": frame_dir,
        "label": label_id,
        "label_name": label_name,
        "hard_label": label_id,
        "hard_label_name": label_name,
        "raw_center_label": session.raw_labels[center] or "",
        "gt_group": session.label_groups[center],
        "subject": session.subject_id,
        "subject_id": session.subject_id,
        "recording_id": session.recording_id,
        "session_name": session.recording_id,
        "jsonl_path": str(session.jsonl_path),
        "label_source": session.label_source,
        "sample_type": "continuous_window",
        "start_frame": start,
        "end_frame": end,
        "center_frame": center,
        "center_timestamp": float(session.timestamps[center]),
        "source_total_frames": session.total_frames,
        "valid_detection_frames": int(np.count_nonzero(session.selected_detection[start:end + 1])),
        "selected_detection_center": bool(session.selected_detection[center]),
        "total_frames": window_size,
        "img_shape": session.img_shape,
        "original_shape": session.img_shape,
        "keypoint": np.ascontiguousarray(keypoint),
        "keypoint_score": np.ascontiguousarray(keypoint_score),
        "per_frame_label_ids": np.ascontiguousarray(per_frame_label_ids),
        "q_temporal": q_temporal,
        **targets,
    }


def manifest_row(fold: str, split: str, annotation: dict[str, Any], output_pkl: Path) -> dict[str, Any]:
    label_ids = [int(value) for value in annotation["per_frame_label_ids"].tolist()]
    labels = [LABELS[value] if value >= 0 else "" for value in label_ids]
    row: dict[str, Any] = {
        "fold": fold,
        "split": split,
        "recording_id": annotation["recording_id"],
        "subject_id": annotation["subject_id"],
        "frame_dir": annotation["frame_dir"],
        "start_frame": annotation["start_frame"],
        "end_frame": annotation["end_frame"],
        "center_frame": annotation["center_frame"],
        "center_timestamp": f"{float(annotation['center_timestamp']):.9f}",
        "manual_center_label": annotation["label_name"],
        "manual_center_label_id": annotation["hard_label"],
        "raw_center_label": annotation["raw_center_label"],
        "gt_group": annotation["gt_group"],
        "valid_detection_frames": annotation["valid_detection_frames"],
        "selected_detection_center": int(annotation["selected_detection_center"]),
        "label_source": annotation["label_source"],
        "jsonl_path": annotation["jsonl_path"],
        "pkl_path": str(output_pkl),
        "keypoint_shape": json_dumps_compact(list(annotation["keypoint"].shape)),
        "keypoint_score_shape": json_dumps_compact(list(annotation["keypoint_score"].shape)),
        "per_frame_label_ids": json_dumps_compact(label_ids),
        "per_frame_labels": json_dumps_compact(labels),
        "q_temporal": json_dumps_compact([float(x) for x in annotation["q_temporal"]]),
    }
    for key, value in annotation.items():
        if key.startswith("target_probs_eta"):
            row[key] = json_dumps_compact([float(x) for x in value])
    return row


def validate_fold_subjects(fold: S2FoldSpec) -> None:
    split_sets = {
        "train": set(fold.train_subjects),
        "val": {fold.val_subject},
        "test": {fold.test_subject},
    }
    if fold.calibration_subject is not None:
        split_sets["calib"] = {fold.calibration_subject}

    seen: dict[str, str] = {}
    for split_name, subjects in split_sets.items():
        for subject in subjects:
            if subject in seen:
                raise ValueError(
                    f"Subject {subject!r} appears in both {seen[subject]} and {split_name} "
                    f"for fold {fold.fold}"
                )
            seen[subject] = split_name


def split_for_subject(fold: S2FoldSpec, subject: str) -> str | None:
    subject = subject.lower()
    if subject in fold.train_subjects:
        return "train"
    if subject == fold.val_subject:
        return "val"
    if subject == fold.test_subject:
        return "test"
    if fold.calibration_subject is not None and subject == fold.calibration_subject:
        return "calib"
    return None


def build_continuous_dataset(
    jsonl_root: Path,
    origin_root: Path,
    output_pkl: Path,
    manifest_dir: Path,
    folds: list[S2FoldSpec],
    etas: tuple[float, ...],
    min_valid_ratio: float,
    min_valid_frames: int | None,
    max_timestamp_gap_sec: float | None,
    include_walk_sessions: bool,
    label_source: str,
    allow_label_mismatch: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if output_pkl.exists() and not overwrite:
        raise FileExistsError(f"{output_pkl} exists; pass --overwrite to replace it")
    if not etas:
        raise ValueError("At least one eta is required")
    for eta in etas:
        if eta < 0 or eta > 1:
            raise ValueError(f"eta must be in [0, 1], got {eta}")

    for fold in folds:
        validate_fold_subjects(fold)

    stats = BuildStats()
    annotations_by_frame_dir: dict[str, dict[str, Any]] = {}
    subject_counts = Counter()
    session_counts = Counter()
    label_counts = Counter()

    jsonl_paths = sorted(jsonl_root.glob("*/*.jsonl"))
    for jsonl_path in jsonl_paths:
        stats.jsonl_files_seen += 1
        recording_id = jsonl_path.parent.name
        if is_walk_session(recording_id) and not include_walk_sessions:
            stats.walk_sessions_skipped += 1
            continue

        session = load_continuous_session(
            jsonl_path=jsonl_path,
            origin_root=origin_root,
            label_source=label_source,
        )
        stats.jsonl_files_used += 1
        stats.label_source_counts[session.label_source] += 1

        starts = window_starts(session.total_frames, WINDOW_SIZE, STRIDE)
        if not starts:
            stats.dropped_windows_by_reason["recording_shorter_than_window"] += 1
            continue

        required_valid_frames = (
            min_valid_frames
            if min_valid_frames is not None
            else math.ceil(min_valid_ratio * WINDOW_SIZE)
        )

        for start in starts:
            end = start + WINDOW_SIZE - 1
            center = start + CENTER_OFFSET
            if end >= session.total_frames:
                stats.dropped_windows_by_reason["incomplete_window"] += 1
                continue
            if has_large_timestamp_gap(session.timestamps, start, end, max_timestamp_gap_sec):
                stats.dropped_windows_by_reason["timestamp_gap"] += 1
                continue

            containing_segments = [
                segment for segment in session.segments
                if int(segment["start_frame"]) <= center <= int(segment["end_frame"])
            ]
            if len(containing_segments) != 1:
                stats.dropped_windows_by_reason["center_segment_count"] += 1
                continue

            center_label_id = int(session.clean_label_ids[center])
            if center_label_id < 0:
                stats.dropped_windows_by_reason["invalid_center_label"] += 1
                continue

            segment_center_label = clean_label(containing_segments[0].get("label"))
            if segment_center_label != LABELS[center_label_id]:
                stats.dropped_windows_by_reason["center_segment_label_mismatch"] += 1
                continue

            valid_detection_frames = int(np.count_nonzero(session.selected_detection[start:end + 1]))
            if valid_detection_frames < required_valid_frames:
                stats.dropped_windows_by_reason["missing_skeleton_threshold"] += 1
                continue

            frame_record = session.frame_records.get(center, {})
            jsonl_center_label = None
            if isinstance(frame_record, dict):
                jsonl_center_label = clean_label(frame_record.get("label"))
            if jsonl_center_label is not None and jsonl_center_label != LABELS[center_label_id]:
                stats.label_mismatches.append(
                    {
                        "recording_id": session.recording_id,
                        "center_frame": center,
                        "origin_label": LABELS[center_label_id],
                        "jsonl_label": jsonl_center_label,
                    }
                )

            annotation = build_annotation(
                session=session,
                start=start,
                window_size=WINDOW_SIZE,
                etas=etas,
            )
            annotations_by_frame_dir[annotation["frame_dir"]] = annotation
            stats.windows_created += 1
            subject_counts[session.subject_id] += 1
            session_counts[session.recording_id] += 1
            label_counts[annotation["label_name"]] += 1

    if stats.label_mismatches and not allow_label_mismatch:
        preview = stats.label_mismatches[:5]
        raise ValueError(
            "Hard center labels do not match existing JSONL center-label logic. "
            f"First mismatches: {preview}"
        )

    annotations = [annotations_by_frame_dir[key] for key in sorted(annotations_by_frame_dir)]
    if not annotations:
        raise RuntimeError("No continuous-window annotations were created")

    split: dict[str, list[str]] = {}
    manifest_rows_by_split = {"train": [], "val": [], "test": []}
    for fold in folds:
        for split_name in ("train", "val", "calib", "test"):
            split[fold.split_key(split_name)] = []

    for annotation in annotations:
        subject = str(annotation["subject_id"]).lower()
        for fold in folds:
            split_name = split_for_subject(fold, subject)
            if split_name is None:
                continue
            split_key = fold.split_key(split_name)
            split[split_key].append(annotation["frame_dir"])
            if split_name in manifest_rows_by_split:
                manifest_rows_by_split[split_name].append(
                    manifest_row(fold.fold, split_name, annotation, output_pkl)
                )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    recording_scope = "all sessions" if include_walk_sessions else "non-walk sessions"

    with output_pkl.open("wb") as handle:
        pickle.dump(
            {
                "split": split,
                "annotations": annotations,
                "labels": LABELS,
                "protocol": protocol_metadata(recording_scope),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_fieldnames = [
        "fold",
        "split",
        "recording_id",
        "subject_id",
        "frame_dir",
        "start_frame",
        "end_frame",
        "center_frame",
        "center_timestamp",
        "manual_center_label",
        "manual_center_label_id",
        "raw_center_label",
        "gt_group",
        "valid_detection_frames",
        "selected_detection_center",
        "label_source",
        "jsonl_path",
        "pkl_path",
        "keypoint_shape",
        "keypoint_score_shape",
        "per_frame_label_ids",
        "per_frame_labels",
        "q_temporal",
        *[f"target_probs_{eta_slug(eta)}" for eta in etas],
    ]
    manifest_paths = {}
    for split_name, rows in manifest_rows_by_split.items():
        path = manifest_dir / f"{split_name}_continuous_windows.csv"
        write_rows_csv(path, manifest_fieldnames, rows, overwrite=overwrite)
        manifest_paths[split_name] = str(path)

    split_counts = {
        key: len(value)
        for key, value in sorted(split.items())
    }
    split_subjects = {
        f"fold_{fold.fold}": {
            "train": list(fold.train_subjects),
            "val": [fold.val_subject],
            "calib": [] if fold.calibration_subject is None else [fold.calibration_subject],
            "test": [fold.test_subject],
        }
        for fold in folds
    }
    summary = {
        "output_pkl": str(output_pkl),
        "manifest_paths": manifest_paths,
        "num_annotations": len(annotations),
        "windows_created": stats.windows_created,
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "etas": list(etas),
        "protocol": protocol_metadata(recording_scope),
        "recording_scope": recording_scope,
        "min_valid_ratio": min_valid_ratio,
        "min_valid_frames": min_valid_frames,
        "max_timestamp_gap_sec": max_timestamp_gap_sec,
        "label_source": label_source,
        "split_subjects": split_subjects,
        "split_counts": split_counts,
        "samples_per_subject": dict(sorted(subject_counts.items())),
        "samples_per_recording": dict(sorted(session_counts.items())),
        "samples_per_class": {label: int(label_counts.get(label, 0)) for label in LABELS},
        "jsonl_files_seen": stats.jsonl_files_seen,
        "jsonl_files_used": stats.jsonl_files_used,
        "walk_sessions_skipped": stats.walk_sessions_skipped,
        "label_source_counts": dict(stats.label_source_counts),
        "dropped_windows_by_reason": dict(stats.dropped_windows_by_reason),
        "label_mismatches": stats.label_mismatches,
    }
    summary_path = output_pkl.with_name(f"{output_pkl.stem}_summary.json")
    write_json(summary_path, summary, overwrite=overwrite)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Study 2 continuous-window manifests and PYSKL pkl.")
    parser.add_argument("--jsonl-root", type=Path, default=DEFAULT_JSONL_ROOT)
    parser.add_argument("--origin-root", type=Path, default=Path("data/radar_v4/origin"))
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_S2_PKL)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_PKL_DIR)
    parser.add_argument("--folds", nargs="+", help="Fold ids to include, e.g. a b c. Default: all discovered folds.")
    parser.add_argument("--etas", nargs="+", type=float, default=list(DEFAULT_ETAS))
    parser.add_argument("--min-valid-ratio", type=float, default=0.0)
    parser.add_argument("--min-valid-frames", type=int)
    parser.add_argument(
        "--max-timestamp-gap-sec",
        type=float,
        default=0.5,
        help="Drop windows with adjacent timestamp jumps above this value. Use a negative value to disable.",
    )
    parser.set_defaults(include_walk_sessions=False)
    parser.add_argument(
        "--include-walk-sessions",
        dest="include_walk_sessions",
        action="store_true",
        help="Include walk-only recordings as an explicit ablation/debug override.",
    )
    parser.add_argument(
        "--exclude-walk-sessions",
        dest="include_walk_sessions",
        action="store_false",
        help="Skip walk-only recordings. This is the default S2 protocol.",
    )
    parser.add_argument("--label-source", choices=["origin", "jsonl", "auto"], default="origin")
    parser.add_argument("--allow-label-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_valid_ratio < 0:
        raise ValueError("--min-valid-ratio must be non-negative")
    if args.min_valid_frames is not None and args.min_valid_frames < 0:
        raise ValueError("--min-valid-frames must be non-negative")

    folds = discover_s2_folds()
    if args.folds:
        requested = {item.lower().replace("fold_", "") for item in args.folds}
        folds = [fold for fold in folds if fold.fold in requested]
        missing = sorted(requested - {fold.fold for fold in folds})
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")

    max_gap = None if args.max_timestamp_gap_sec < 0 else args.max_timestamp_gap_sec
    summary = build_continuous_dataset(
        jsonl_root=args.jsonl_root,
        origin_root=args.origin_root,
        output_pkl=args.output_pkl,
        manifest_dir=args.manifest_dir,
        folds=folds,
        etas=tuple(args.etas),
        min_valid_ratio=args.min_valid_ratio,
        min_valid_frames=args.min_valid_frames,
        max_timestamp_gap_sec=max_gap,
        include_walk_sessions=args.include_walk_sessions,
        label_source=args.label_source,
        allow_label_mismatch=args.allow_label_mismatch,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote {summary['num_annotations']} continuous windows to {summary['output_pkl']}")
    for split_name, path in summary["manifest_paths"].items():
        print(f"[DONE] {split_name} manifest: {path}")


if __name__ == "__main__":
    main()
