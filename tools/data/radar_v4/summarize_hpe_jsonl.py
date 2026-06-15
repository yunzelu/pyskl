"""
Summarize YOLO pose detection and keypoint confidence from radar_v4 HPE JSONL.

Example:
    python tools/data/radar_v4/summarize_hpe_jsonl.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from stats_common import (
    label_to_group,
    normalize_label,
    read_frame_label_rows,
    rows_to_segments,
    safe_float,
    write_csv,
)


BODY_GROUPS = {
    "head": list(range(0, 5)),
    "upper_body": list(range(5, 11)),
    "lower_body": list(range(11, 17)),
}


class SegmentLabelLookup:
    def __init__(self, segments: list[dict[str, Any]]):
        self.segments = sorted(
            segments,
            key=lambda seg: int(seg["start_frame"]),
        )
        self.pos = 0

    def get(self, frame_idx: int) -> tuple[str, str] | None:
        while self.pos + 1 < len(self.segments):
            next_start = int(self.segments[self.pos + 1]["start_frame"])
            if frame_idx < next_start:
                break
            self.pos += 1

        if self.pos >= len(self.segments):
            return None

        seg = self.segments[self.pos]
        start = int(seg["start_frame"])
        end = seg["end_frame"]

        if frame_idx < start:
            return None

        if end is not None and frame_idx > int(end):
            return None

        return seg["label"], seg["label_group"]


def make_label_stats() -> dict[str, Any]:
    return {
        "label_groups": set(),
        "sessions": set(),
        "jsonl_files": set(),
        "frame_count": 0,
        "detected_frame_count": 0,
        "bbox_conf_sum": 0.0,
        "bbox_conf_count": 0,
        "frames_with_keypoints_conf": 0,
        "keypoint_conf_sum": 0.0,
        "keypoint_conf_count": 0,
        "body_group_conf_sum": {name: 0.0 for name in BODY_GROUPS},
        "body_group_conf_count": {name: 0 for name in BODY_GROUPS},
    }


def clean_conf(value: Any) -> float | None:
    conf = safe_float(value)
    if conf is None or math.isnan(conf):
        return None
    return conf


def frame_label_from_record(
    record: dict[str, Any],
    include_ignore: bool,
    include_unlabeled: bool,
) -> tuple[str, str] | None:
    label = normalize_label(record.get("label"))

    if label is None:
        if not include_unlabeled:
            return None
        return "unlabeled", "unlabeled"

    label_group = normalize_label(record.get("label_group")) or label_to_group(label)
    if label_group == "ignore" and not include_ignore:
        return None

    return label, label_group


def filter_label(
    label_info: tuple[str, str] | None,
    include_ignore: bool,
    include_unlabeled: bool,
) -> tuple[str, str] | None:
    if label_info is None:
        if not include_unlabeled:
            return None
        return "unlabeled", "unlabeled"

    label, label_group = label_info
    if label_group == "ignore" and not include_ignore:
        return None

    return label, label_group


def update_conf_stats(stats: dict[str, Any], keypoints_conf: Any) -> None:
    if not isinstance(keypoints_conf, list):
        return

    cleaned = [clean_conf(value) for value in keypoints_conf]
    valid_values = [value for value in cleaned if value is not None]

    if not valid_values:
        return

    stats["frames_with_keypoints_conf"] += 1
    stats["keypoint_conf_sum"] += sum(valid_values)
    stats["keypoint_conf_count"] += len(valid_values)

    for group_name, indices in BODY_GROUPS.items():
        group_values = [
            cleaned[idx]
            for idx in indices
            if idx < len(cleaned) and cleaned[idx] is not None
        ]

        if not group_values:
            continue

        stats["body_group_conf_sum"][group_name] += sum(group_values)
        stats["body_group_conf_count"][group_name] += len(group_values)


def read_jsonl_stats(
    jsonl_root: Path,
    origin_root: Path | None,
    output_dir: Path,
    include_ignore: bool,
    include_unlabeled: bool,
    label_source: str,
) -> tuple[Path, Path]:
    jsonl_paths = sorted(jsonl_root.rglob("*.jsonl"))
    if not jsonl_paths:
        raise RuntimeError(f"No JSONL files found under {jsonl_root}.")

    by_label: dict[str, dict[str, Any]] = defaultdict(make_label_stats)
    file_rows: list[dict[str, Any]] = []

    bad_json_lines = 0
    missing_origin_labels = 0

    for jsonl_path in jsonl_paths:
        session_name = jsonl_path.parent.name
        origin_lookup: SegmentLabelLookup | None = None
        origin_label_path: Path | None = None
        origin_label_missing = False

        if label_source == "origin":
            if origin_root is None:
                raise RuntimeError("--origin-root is required when --label-source origin.")

            origin_label_path = origin_root / session_name / "frame_labels.csv"
            if origin_label_path.exists():
                rows = read_frame_label_rows(origin_label_path)
                origin_lookup = SegmentLabelLookup(rows_to_segments(rows))
            else:
                origin_label_missing = True
                missing_origin_labels += 1

        file_frame_count = 0
        file_used_frame_count = 0
        file_detected_used_frame_count = 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    bad_json_lines += 1
                    continue

                record_type = record.get("type")

                if record_type == "metadata":
                    session_name = (
                        record.get("dataset_info", {}).get("session_name")
                        or session_name
                    )
                    continue

                if record_type != "frame":
                    continue

                file_frame_count += 1
                if origin_lookup is not None:
                    frame_idx = int(record.get("frame_idx", file_frame_count - 1))
                    label_info = filter_label(
                        origin_lookup.get(frame_idx),
                        include_ignore=include_ignore,
                        include_unlabeled=include_unlabeled,
                    )
                elif label_source == "origin" and origin_label_missing:
                    label_info = None
                else:
                    label_info = frame_label_from_record(
                        record,
                        include_ignore=include_ignore,
                        include_unlabeled=include_unlabeled,
                    )

                if label_info is None:
                    continue

                label, group = label_info
                stats = by_label[label]
                stats["label_groups"].add(group)
                stats["sessions"].add(session_name)
                stats["jsonl_files"].add(str(jsonl_path))
                stats["frame_count"] += 1
                file_used_frame_count += 1

                detected = bool(record.get("detected"))
                if detected:
                    stats["detected_frame_count"] += 1
                    file_detected_used_frame_count += 1

                bbox_conf = clean_conf(record.get("bbox_conf"))
                if bbox_conf is not None:
                    stats["bbox_conf_sum"] += bbox_conf
                    stats["bbox_conf_count"] += 1

                update_conf_stats(stats, record.get("keypoints_conf"))

        file_rows.append(
            {
                "jsonl_path": str(jsonl_path),
                "session_name": session_name,
                "label_source": (
                    str(origin_label_path)
                    if origin_label_path is not None and origin_label_path.exists()
                    else label_source
                ),
                "frame_count": file_frame_count,
                "included_labeled_frame_count": file_used_frame_count,
                "included_detected_frame_count": file_detected_used_frame_count,
                "included_detection_rate": (
                    file_detected_used_frame_count / file_used_frame_count
                    if file_used_frame_count
                    else None
                ),
            }
        )

    summary_rows: list[dict[str, Any]] = []

    for label, stats in sorted(by_label.items()):
        frame_count = stats["frame_count"]
        detected_count = stats["detected_frame_count"]
        bbox_count = stats["bbox_conf_count"]
        kpt_count = stats["keypoint_conf_count"]

        row = {
            "label": label,
            "label_group": ";".join(sorted(stats["label_groups"])),
            "frame_count": frame_count,
            "detected_frame_count": detected_count,
            "undetected_frame_count": frame_count - detected_count,
            "detection_rate": detected_count / frame_count if frame_count else None,
            "frames_with_keypoints_conf": stats["frames_with_keypoints_conf"],
            "avg_keypoint_conf": (
                stats["keypoint_conf_sum"] / kpt_count if kpt_count else None
            ),
            "avg_bbox_conf": (
                stats["bbox_conf_sum"] / bbox_count if bbox_count else None
            ),
            "keypoint_conf_value_count": kpt_count,
            "session_count": len(stats["sessions"]),
            "jsonl_file_count": len(stats["jsonl_files"]),
        }

        for group_name in BODY_GROUPS:
            group_count = stats["body_group_conf_count"][group_name]
            row[f"avg_{group_name}_conf"] = (
                stats["body_group_conf_sum"][group_name] / group_count
                if group_count
                else None
            )
            row[f"{group_name}_conf_value_count"] = group_count

        summary_rows.append(row)

    summary_csv = output_dir / "hpe_jsonl_class_summary.csv"
    file_csv = output_dir / "hpe_jsonl_file_summary.csv"

    write_csv(
        summary_csv,
        [
            "label",
            "label_group",
            "frame_count",
            "detected_frame_count",
            "undetected_frame_count",
            "detection_rate",
            "frames_with_keypoints_conf",
            "avg_keypoint_conf",
            "avg_head_conf",
            "avg_upper_body_conf",
            "avg_lower_body_conf",
            "avg_bbox_conf",
            "keypoint_conf_value_count",
            "head_conf_value_count",
            "upper_body_conf_value_count",
            "lower_body_conf_value_count",
            "session_count",
            "jsonl_file_count",
        ],
        summary_rows,
    )

    write_csv(
        file_csv,
        [
            "jsonl_path",
            "session_name",
            "label_source",
            "frame_count",
            "included_labeled_frame_count",
            "included_detected_frame_count",
            "included_detection_rate",
        ],
        file_rows,
    )

    print(f"[INFO] Processed {len(jsonl_paths)} JSONL files.")
    print(f"[INFO] Wrote class summary: {summary_csv}")
    print(f"[INFO] Wrote file summary: {file_csv}")

    if bad_json_lines:
        print(f"[WARN] Skipped {bad_json_lines} malformed JSON lines.")

    if missing_origin_labels:
        print(f"[WARN] Missing origin annotation for {missing_origin_labels} JSONL files.")

    return summary_csv, file_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl-root",
        type=Path,
        default=Path("data/radar_v4/raw_jsonl/yolo26xpose"),
        help="Root folder containing HPE JSONL files.",
    )
    parser.add_argument(
        "--origin-root",
        type=Path,
        default=Path("data/radar_v4/origin"),
        help="Root folder containing authoritative frame_labels.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/radar_v4/stats"),
        help="Directory where CSV statistics will be written.",
    )
    parser.add_argument(
        "--label-source",
        choices=["origin", "jsonl"],
        default="origin",
        help="Use current origin annotations or labels embedded in JSONL.",
    )
    parser.add_argument(
        "--include-ignore",
        action="store_true",
        help="Include DELETE frames.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include unlabeled frames before/after annotated periods.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_jsonl_stats(
        jsonl_root=args.jsonl_root,
        origin_root=args.origin_root,
        output_dir=args.output_dir,
        include_ignore=args.include_ignore,
        include_unlabeled=args.include_unlabeled,
        label_source=args.label_source,
    )


if __name__ == "__main__":
    main()
