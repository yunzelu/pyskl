"""Build project-only 10 fps JSONLs and one PYSKL train/validation dataset.

Run from the repository root with Python and NumPy installed. Existing pose
predictions are reused; timestamps come from the original recording CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun.dataset import build_radar_v4_yolo26xpose_datasets as rerun


DOWNSAMPLE_FACTOR = 3
NOMINAL_FPS = 10
WINDOW_SIZE = 20
STRIDE = 4
MAX_ADJACENT_GAP_SEC = 0.5
MAX_WINDOW_SPAN_SEC = 2.5
TIMESTAMP_POLICY = "timestamps_csv_unix_seconds_matched_by_original_frame_idx"
SPLIT_SUBJECTS = {
    "train": ["chenzhe", "dengdeng", "han", "hui", "jiadi", "li", "mia", "rose", "saad", "xilai"],
    "val": ["yunze"],
}


def validate_subjects(subjects: set[str]) -> None:
    expected = {subject for assigned in SPLIT_SUBJECTS.values() for subject in assigned}
    if subjects != expected:
        raise ValueError(
            f"Expected the 11 project subjects. Missing={sorted(expected - subjects)}, "
            f"unexpected={sorted(subjects - expected)}"
        )


def make_project_split(annotations: list[dict[str, Any]]) -> dict[str, list[str]]:
    validate_subjects({annotation["subject"] for annotation in annotations})
    ids = [annotation["frame_dir"] for annotation in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs in project annotations")
    return {
        split_name: [annotation["frame_dir"] for annotation in annotations if annotation["subject"] in subjects]
        for split_name, subjects in SPLIT_SUBJECTS.items()
    }


def save_project_protocol(output_root: Path, result: rerun.ProtocolResult) -> Path:
    annotations = result.annotations
    split = make_project_split(annotations)
    filename = f"{rerun.DATASET_ID}_{result.protocol_id}_val_yunze.pkl"
    pkl_path = output_root / "pyskl" / result.protocol_id / filename
    rerun.save_pkl(pkl_path, annotations, split)

    stats_dir = output_root / "stats" / result.protocol_id
    subjects = sorted({annotation["subject"] for annotation in annotations})
    labels = rerun.FINAL_LABELS
    subject_counts = Counter(annotation["subject"] for annotation in annotations)
    class_counts = Counter(annotation["label_name"] for annotation in annotations)
    subject_class_counts = Counter((a["subject"], a["label_name"]) for a in annotations)
    split_classes = {}
    split_subjects = {}
    for split_name, assigned_subjects in SPLIT_SUBJECTS.items():
        selected = [a for a in annotations if a["subject"] in assigned_subjects]
        counts = Counter(a["label_name"] for a in selected)
        split_classes[split_name] = {label: counts[label] for label in labels}
        split_subjects[split_name] = {subject: subject_counts[subject] for subject in assigned_subjects}
    summary = {
        "dataset_id": rerun.DATASET_ID,
        "protocol_id": result.protocol_id,
        "split_policy": "yunze_validation_other_10_subjects_training",
        "subjects": SPLIT_SUBJECTS,
        "label_to_id": rerun.LABEL_TO_ID,
        "num_annotations": len(annotations),
        "num_samples_by_split": {name: len(ids) for name, ids in split.items()},
        "samples_per_class": {label: class_counts[label] for label in labels},
        "samples_per_subject": dict(sorted(subject_counts.items())),
        "samples_per_split_class": split_classes,
        "samples_per_split_subject": split_subjects,
        "protocol_stats": result.stats,
        "pkl_path": str(pkl_path),
    }
    rerun.write_json(pkl_path.with_name(f"{pkl_path.stem}_summary.json"), summary)
    rerun.write_json(stats_dir / "summary.json", summary)
    rerun.write_csv(stats_dir / "samples_by_subject.csv", ["subject", "num_samples"], [
        {"subject": subject, "num_samples": subject_counts[subject]} for subject in subjects
    ])
    rerun.write_csv(stats_dir / "samples_by_class.csv", ["label_id", "label_name", "num_samples"], [
        {"label_id": index, "label_name": label, "num_samples": class_counts[label]}
        for index, label in enumerate(labels)
    ])
    rerun.write_csv(stats_dir / "samples_by_subject_class.csv",
                    ["subject", "label_id", "label_name", "num_samples"], [
        {"subject": subject, "label_id": index, "label_name": label,
         "num_samples": subject_class_counts[subject, label]}
        for subject in subjects for index, label in enumerate(labels)
    ])
    rerun.write_csv(stats_dir / "samples_by_split.csv", ["split", "num_samples", "subjects"], [
        {"split": name, "num_samples": len(ids), "subjects": ",".join(SPLIT_SUBJECTS[name])}
        for name, ids in split.items()
    ])
    rerun.write_csv(stats_dir / "samples_by_split_class.csv",
                    ["split", "label_id", "label_name", "num_samples"], [
        {"split": name, "label_id": index, "label_name": label, "num_samples": split_classes[name][label]}
        for name in split for index, label in enumerate(labels)
    ])
    rerun.write_csv(stats_dir / "samples_by_split_subject.csv", ["split", "subject", "num_samples"], [
        {"split": name, "subject": subject, "num_samples": count}
        for name, counts in split_subjects.items() for subject, count in counts.items()
    ])
    return pkl_path


def read_timestamps(path: Path) -> dict[int, float]:
    """Read explicit Frame keys, rejecting ambiguous or unusable camera times."""
    timestamps: dict[int, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"Frame", "Timestamp"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected Frame,Timestamp columns in {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_idx = int(row["Frame"])
                timestamp = float(row["Timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timestamp CSV row {path}:{line_number}") from exc
            if frame_idx < 0 or frame_idx in timestamps:
                raise ValueError(f"Negative or duplicate Frame {frame_idx} in {path}")
            if not math.isfinite(timestamp):
                raise ValueError(f"Non-finite timestamp for Frame {frame_idx} in {path}")
            timestamps[frame_idx] = timestamp
    if 0 not in timestamps:
        raise ValueError(f"Missing recording-start Frame 0 in {path}")
    ordered = [timestamps[index] for index in sorted(timestamps)]
    if any(later <= earlier for earlier, later in zip(ordered, ordered[1:])):
        raise ValueError(f"Timestamps must increase strictly with Frame in {path}")
    return timestamps


def find_timestamps(origin_root: Path, session_name: str) -> Path:
    candidates = [origin_root / session_name / name for name in ("timestamps.csv", "timestamp.csv")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one timestamps.csv or timestamp.csv in {origin_root / session_name}")
    return found[0]


def preprocess_session(
    source_jsonl: Path,
    timestamps_csv: Path,
    processed_jsonl: Path,
    phase: int = 0,
) -> dict[str, Any]:
    """Select the camera-frame phase first, then remove detected=false rows."""
    if phase not in range(DOWNSAMPLE_FACTOR):
        raise ValueError("Phase must be 0, 1, or 2")
    if source_jsonl.resolve() == processed_jsonl.resolve():
        raise ValueError("Source and processed JSONL must be different files")
    identity = rerun.parse_session_dir_name(source_jsonl.parent.name)
    timestamps = read_timestamps(timestamps_csv)
    metadata = None
    source_process_summary = None
    frames = []
    counts = dict(
        source_frame_rows=0,
        source_false_detection_rows=0,
        removed_other_phase_rows=0,
        phase_frame_rows=0,
        removed_false_detection_rows=0,
        kept_frame_rows=0,
    )
    previous_frame = -1
    with source_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "metadata":
                if metadata is not None:
                    raise ValueError(f"Multiple metadata rows in {source_jsonl}")
                metadata = record
                continue
            if record.get("type") == "process_summary":
                if source_process_summary is not None:
                    raise ValueError(f"Multiple process summaries in {source_jsonl}")
                source_process_summary = record
                continue
            if record.get("type") != "frame":
                raise ValueError(f"Unknown JSONL row type at {source_jsonl}:{line_number}")
            frame_idx = record.get("frame_idx")
            if type(frame_idx) is not int or frame_idx <= previous_frame:
                raise ValueError(f"Frame indices must be increasing integers at {source_jsonl}:{line_number}")
            if frame_idx not in timestamps:
                raise ValueError(f"Missing timestamp for Frame {frame_idx} in {timestamps_csv}")
            previous_frame = frame_idx
            counts["source_frame_rows"] += 1
            counts["source_false_detection_rows"] += int(record.get("detected") is False)
            if frame_idx % DOWNSAMPLE_FACTOR != phase:
                counts["removed_other_phase_rows"] += 1
                continue
            counts["phase_frame_rows"] += 1
            if record.get("detected") is False:
                counts["removed_false_detection_rows"] += 1
                continue
            record["source_timestamp_sec"] = record.get("timestamp_sec")
            record["timestamp_sec"] = timestamps[frame_idx]
            record["elapsed_timestamp_sec"] = timestamps[frame_idx] - timestamps[0]
            frames.append(record)
    if metadata is None:
        raise ValueError(f"Missing metadata in {source_jsonl}")
    if not frames:
        raise ValueError(f"No detected frames for phase {phase} in {source_jsonl}")
    dataset_info = metadata.get("dataset_info", {})
    for field, expected in (("subject", identity.subject), ("session_name", identity.directory_name.lower())):
        value = str(dataset_info.get(field, "")).strip().lower()
        if value and value != expected:
            raise ValueError(f"Metadata {field} mismatch in {source_jsonl}: {value!r} != {expected!r}")
    counts["kept_frame_rows"] = len(frames)
    indices = sorted(timestamps)
    duration = timestamps[indices[-1]] - timestamps[0]
    source_fps = indices[-1] / duration if duration > 0 else None
    stats = {
        "session_dir": identity.directory_name,
        "subject": identity.subject,
        "session": identity.session,
        "session_family": identity.session_family,
        "source_jsonl": str(source_jsonl),
        "processed_jsonl": str(processed_jsonl),
        "timestamps_csv": str(timestamps_csv),
        "timestamp_csv_rows": len(timestamps),
        "timestamp_csv_rows_without_source_frame": len(timestamps) - counts["source_frame_rows"],
        "recording_start_timestamp_sec": timestamps[0],
        "recording_span_sec": duration,
        "measured_source_fps": source_fps,
        "nominal_fps": NOMINAL_FPS,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "downsample_phase": phase,
        **counts,
    }
    metadata["preprocess_info"] = {
        **stats,
        "operation": "select_original_frame_phase_then_remove_detected_false",
        "frame_idx_policy": "preserve original camera frame indices and annotation segments",
        "timestamp_policy": TIMESTAMP_POLICY,
        "phase_rule": f"frame_idx % {DOWNSAMPLE_FACTOR} == {phase}",
    }
    if source_process_summary is not None:
        metadata["source_process_summary"] = source_process_summary
    video_info = metadata.setdefault("video_info", {})
    video_info["source_assumed_fps_used_for_timestamp"] = video_info.pop("assumed_fps_used_for_timestamp", None)
    metadata["timestamp_info"] = {
        "source_csv": str(timestamps_csv),
        "frame_column": "Frame",
        "timestamp_column": "Timestamp",
        "unit": "seconds",
        "timebase": "unix_epoch",
        "recording_start_timestamp_sec": timestamps[0],
    }
    metadata.setdefault("format_note", {}).update({
        "timestamp_sec": "Real Unix timestamp from the CSV Timestamp column, joined on Frame = frame_idx.",
        "source_timestamp_sec": "Original timestamp_sec from the source pose JSONL (typically frame_idx / 30).",
        "elapsed_timestamp_sec": "timestamp_sec minus the CSV timestamp for original camera Frame 0.",
    })
    processed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = processed_jsonl.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rerun.json_dump_line(metadata))
            for frame in frames:
                handle.write(rerun.json_dump_line(frame))
        temporary.replace(processed_jsonl)
    finally:
        temporary.unlink(missing_ok=True)
    return stats


def build_project_windows(sessions: list[rerun.SessionData], phase: int) -> rerun.ProtocolResult:
    for session in sessions:
        if np.any(session.frame_indices % DOWNSAMPLE_FACTOR != phase):
            raise ValueError(f"Wrong sampling phase in {session.jsonl_path}")
        if not np.all(np.isfinite(session.timestamps_sec)) or np.any(np.diff(session.timestamps_sec) <= 0):
            raise ValueError(f"Invalid camera timestamp sequence in {session.jsonl_path}")
    result = rerun.build_continuous_window(
        sessions,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_adjacent_gap_sec=MAX_ADJACENT_GAP_SEC,
        max_window_span_sec=MAX_WINDOW_SPAN_SEC,
    )
    provenance = {
        "nominal_fps": NOMINAL_FPS,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "downsample_phase": phase,
        "timestamp_policy": TIMESTAMP_POLICY,
    }
    result.stats.update(provenance)
    for annotation in result.annotations:
        annotation["frame_dir"] = f"project_fps10_phase{phase}__{annotation['frame_dir']}"
        annotation.update(provenance)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl-root", type=Path, default=rerun.RAW_JSONL_ROOT)
    parser.add_argument("--origin-root", type=Path, default=Path("data/radar_v4/origin"))
    parser.add_argument("--phase", type=int, choices=range(DOWNSAMPLE_FACTOR), default=0)
    parser.add_argument(
        "--output-root", type=Path,
        help="Defaults to data/project/yolo26xpose/fps10_phase<phase>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or Path(f"data/project/yolo26xpose/fps10_phase{args.phase}")
    # Never allow a project build to overwrite source data or thesis artifacts.
    for protected in (args.raw_jsonl_root, args.origin_root, Path("data/radar_v4/rerun")):
        if output_root.resolve() == protected.resolve() or protected.resolve() in output_root.resolve().parents:
            raise ValueError(f"Project output root must be outside {protected}")
    selected = sorted(rerun.selected_session_dirs(args.raw_jsonl_root, "included"),
                      key=lambda item: item[0].recording_index)
    if not selected:
        raise ValueError(f"No included recordings in {args.raw_jsonl_root}")
    validate_subjects({identity.subject for identity, _ in selected})
    inputs = [
        (identity, rerun.find_single_jsonl(directory),
         find_timestamps(args.origin_root, identity.directory_name))
        for identity, directory in selected
    ]
    records = []
    sessions = []
    for index, (identity, source_jsonl, timestamps_csv) in enumerate(inputs, start=1):
        target = (output_root / "detected_jsonl" / identity.directory_name /
                  f"{source_jsonl.stem}__fps10_phase{args.phase}__detected-only.jsonl")
        record = preprocess_session(source_jsonl, timestamps_csv, target, args.phase)
        records.append(record)
        session = rerun.load_session(target)
        if session is None:
            raise ValueError(f"Unexpected excluded recording: {target}")
        sessions.append(session)
        print(f"[{index}/{len(inputs)}] {identity.directory_name}: "
              f"{record['source_frame_rows']} source -> {record['kept_frame_rows']} detected phase-{args.phase} rows",
              flush=True)
    result = build_project_windows(sessions, args.phase)
    timestamp_sources = {record["session_dir"]: record for record in records}
    for annotation in result.annotations:
        record = timestamp_sources[annotation["session_name"]]
        annotation["timestamps_csv_path"] = record["timestamps_csv"]
        annotation["recording_start_timestamp_sec"] = record["recording_start_timestamp_sec"]
    rerun.save_label_map(output_root)
    rerun.write_csv(output_root / "stats" / "preprocess_jsonl.csv", list(records[0]), records)
    totals = {key: sum(record[key] for record in records) for key in (
        "source_frame_rows", "source_false_detection_rows", "removed_other_phase_rows",
        "phase_frame_rows", "removed_false_detection_rows", "kept_frame_rows",
    )}
    rerun.write_json(output_root / "stats" / "preprocess_summary.json", {
        "processed_recordings": len(records),
        "raw_jsonl_root": str(args.raw_jsonl_root),
        "origin_root": str(args.origin_root),
        "output_root": str(output_root),
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "downsample_phase": args.phase,
        "nominal_fps": NOMINAL_FPS,
        "timestamp_policy": TIMESTAMP_POLICY,
        "split_subjects": SPLIT_SUBJECTS,
        **totals,
    })
    pkl_path = save_project_protocol(output_root, result)
    print(f"[DONE] {len(result.annotations)} windows; train/val dataset saved to {pkl_path}", flush=True)
    print(json.dumps(rerun.counter_to_regular(result.stats["dropped_windows_by_reason"]), indent=2))


if __name__ == "__main__":
    main()
