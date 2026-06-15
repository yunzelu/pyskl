from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_subject_from_folder(folder_name: str) -> str:
    parts = folder_name.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None

    text = str(label).strip()
    if not text:
        return None

    return text


def label_to_group(label: str | None) -> str:
    """
    Keep this grouping aligned with hpe_jsonl.py.
    """
    if label is None:
        return "unlabeled"

    text = label.strip().lower()

    if text in {"delete", "end"}:
        return "ignore"

    if "stationary" in text:
        return "stationary"

    if "transition" in text:
        return "transition"

    if "fall" in text:
        return "transition"

    if "walk" in text:
        return "walking"

    return "other"


def use_for_audit(label_group: str) -> bool:
    return label_group in {"stationary", "transition"}


def read_frame_label_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            frame = safe_int(row.get("Frame"))
            label = normalize_label(row.get("Label"))

            if frame is None or label is None:
                continue

            rows.append(
                {
                    "row_idx": row_idx,
                    "frame": frame,
                    "label": label,
                }
            )

    return sorted(rows, key=lambda x: (x["frame"], x["row_idx"]))


def rows_to_segments(
    rows: list[dict[str, Any]],
    final_frame: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convert frame_labels.csv rows into label segments.

    Each row marks the first frame of a label. The next row marks the exclusive
    end boundary, so the segment end frame is next_frame - 1. END rows are
    boundaries only and are not emitted as segments.
    """
    segments: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        start = int(row["frame"])
        label = normalize_label(row["label"])

        if label is None or label.upper() == "END":
            continue

        open_ended = i + 1 >= len(rows)

        if i + 1 < len(rows):
            end = int(rows[i + 1]["frame"]) - 1
        elif final_frame is not None:
            end = final_frame
        else:
            end = None

        label_group = label_to_group(label)
        length = None if end is None else end - start + 1

        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "label": label,
                "label_group": label_group,
                "use_for_audit": use_for_audit(label_group),
                "length_frames": length,
                "open_ended": open_ended,
            }
        )

    return segments


def discover_annotation_sessions(root: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []

    for label_path in sorted(root.rglob("frame_labels.csv")):
        folder = label_path.parent
        sessions.append(
            {
                "folder": folder,
                "session_name": folder.name,
                "subject": parse_subject_from_folder(folder.name),
                "label_path": label_path,
            }
        )

    return sessions


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def median(values: Iterable[float]) -> float | None:
    items = sorted(values)
    if not items:
        return None

    mid = len(items) // 2
    if len(items) % 2:
        return items[mid]

    return (items[mid - 1] + items[mid]) / 2.0


def fmt(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({name: fmt(row.get(name)) for name in fieldnames})
