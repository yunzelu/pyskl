"""
Summarize radar_v4 activity segments from the original annotation CSV files.

Example:
    python tools/data/radar_v4/summarize_segments.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from stats_common import (
    discover_annotation_sessions,
    mean,
    median,
    read_frame_label_rows,
    rows_to_segments,
    write_csv,
)


def read_avi_final_frame(video_path: Path) -> int | None:
    """
    Read dwTotalFrames from the AVI main header without requiring OpenCV.
    """
    try:
        with video_path.open("rb") as f:
            header = f.read(1024 * 1024)
    except OSError:
        return None

    avih_pos = header.find(b"avih")
    if avih_pos < 0:
        return None

    chunk_data_pos = avih_pos + 8
    total_frames_pos = chunk_data_pos + 16
    total_frames_end = total_frames_pos + 4
    if total_frames_end > len(header):
        return None

    total_frames = int.from_bytes(
        header[total_frames_pos:total_frames_end],
        byteorder="little",
        signed=False,
    )

    if total_frames <= 0:
        return None

    return total_frames - 1


def read_video_final_frame(video_path: Path) -> int | None:
    if not video_path.exists():
        return None

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return read_avi_final_frame(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return read_avi_final_frame(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        return read_avi_final_frame(video_path)

    return frame_count - 1


def summarize_segments(
    origin_root: Path,
    output_dir: Path,
    assumed_fps: float,
    include_ignore: bool,
    clip_open_ended_to_video: bool,
) -> tuple[Path, Path]:
    sessions = discover_annotation_sessions(origin_root)
    if not sessions:
        raise RuntimeError(
            f"No annotation files found under {origin_root}. "
            "Expected session folders containing frame_labels.csv."
        )

    detail_rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "label_groups": set(),
            "lengths": [],
            "sessions": set(),
            "subjects": set(),
        }
    )

    clipped_open_ended = 0
    skipped_open_ended = 0
    skipped_invalid_length = 0

    for session in sessions:
        rows = read_frame_label_rows(session["label_path"])
        final_frame = None
        if clip_open_ended_to_video:
            final_frame = read_video_final_frame(session["folder"] / "output_video.avi")

        segments = rows_to_segments(rows, final_frame=final_frame)

        for seg in segments:
            if seg["label_group"] == "ignore" and not include_ignore:
                continue

            length = seg["length_frames"]
            if length is None:
                skipped_open_ended += 1
                continue

            if length <= 0:
                skipped_invalid_length += 1
                continue

            if seg["open_ended"]:
                clipped_open_ended += 1

            label = seg["label"]
            length_sec = length / assumed_fps if assumed_fps > 0 else None

            detail_rows.append(
                {
                    "session_name": session["session_name"],
                    "subject": session["subject"],
                    "label": label,
                    "label_group": seg["label_group"],
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "length_frames": length,
                    "length_sec": length_sec,
                    "open_ended": seg["open_ended"],
                    "use_for_audit": seg["use_for_audit"],
                    "annotation_path": str(session["label_path"]),
                }
            )

            item = by_label[label]
            item["label_groups"].add(seg["label_group"])
            item["lengths"].append(float(length))
            item["sessions"].add(session["session_name"])
            item["subjects"].add(session["subject"])

    summary_rows: list[dict[str, Any]] = []
    for label, item in sorted(by_label.items()):
        lengths = item["lengths"]
        total_frames = sum(lengths)
        avg_frames = mean(lengths)
        median_frames = median(lengths)
        min_frames = min(lengths) if lengths else None
        max_frames = max(lengths) if lengths else None

        summary_rows.append(
            {
                "label": label,
                "label_group": ";".join(sorted(item["label_groups"])),
                "segment_count": len(lengths),
                "session_count": len(item["sessions"]),
                "subject_count": len(item["subjects"]),
                "total_frames": total_frames,
                "avg_length_frames": avg_frames,
                "median_length_frames": median_frames,
                "min_length_frames": min_frames,
                "max_length_frames": max_frames,
                "total_sec": total_frames / assumed_fps if assumed_fps > 0 else None,
                "avg_length_sec": avg_frames / assumed_fps
                if avg_frames is not None and assumed_fps > 0
                else None,
            }
        )

    detail_rows = sorted(
        detail_rows,
        key=lambda row: (
            row["label"],
            row["session_name"],
            int(row["start_frame"]),
        ),
    )

    summary_csv = output_dir / "segment_class_summary.csv"
    detail_csv = output_dir / "segment_detail.csv"

    write_csv(
        summary_csv,
        [
            "label",
            "label_group",
            "segment_count",
            "session_count",
            "subject_count",
            "total_frames",
            "avg_length_frames",
            "median_length_frames",
            "min_length_frames",
            "max_length_frames",
            "total_sec",
            "avg_length_sec",
        ],
        summary_rows,
    )

    write_csv(
        detail_csv,
        [
            "session_name",
            "subject",
            "label",
            "label_group",
            "start_frame",
            "end_frame",
            "length_frames",
            "length_sec",
            "open_ended",
            "use_for_audit",
            "annotation_path",
        ],
        detail_rows,
    )

    print(f"[INFO] Processed {len(sessions)} annotation files.")
    print(f"[INFO] Wrote class summary: {summary_csv}")
    print(f"[INFO] Wrote segment detail: {detail_csv}")

    if clipped_open_ended:
        print(f"[INFO] Clipped {clipped_open_ended} open-ended segments to video length.")

    if skipped_open_ended:
        print(f"[WARN] Skipped {skipped_open_ended} open-ended segments.")

    if skipped_invalid_length:
        print(f"[WARN] Skipped {skipped_invalid_length} invalid-length segments.")

    return summary_csv, detail_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin-root",
        type=Path,
        default=Path("data/radar_v4/origin"),
        help="Root folder containing radar_v4 original session folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/radar_v4/stats"),
        help="Directory where CSV statistics will be written.",
    )
    parser.add_argument(
        "--assumed-fps",
        type=float,
        default=30.0,
        help="FPS used to convert segment lengths from frames to seconds.",
    )
    parser.add_argument(
        "--include-ignore",
        action="store_true",
        help="Include DELETE segments. END rows are always treated as boundaries only.",
    )
    parser.add_argument(
        "--no-clip-open-ended-to-video",
        action="store_false",
        dest="clip_open_ended_to_video",
        default=True,
        help="Do not use output_video.avi frame count for final annotations without END.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_segments(
        origin_root=args.origin_root,
        output_dir=args.output_dir,
        assumed_fps=args.assumed_fps,
        include_ignore=args.include_ignore,
        clip_open_ended_to_video=args.clip_open_ended_to_video,
    )


if __name__ == "__main__":
    main()
