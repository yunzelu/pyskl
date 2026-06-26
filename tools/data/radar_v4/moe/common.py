"""Utilities for radar v4 stream-level mixture-of-experts fusion."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LABEL_MAP = Path("tools/data/label_map/radarv4.txt")
DEFAULT_STREAMS = ("j", "jm", "b", "bm")
DROP_LABELS = {
    "DELETE",
    "END",
    "Kneeling-Stationary",
    "Transition-Kneeling-to-Stand",
}
RENAME_LABELS = {
    "Transition-Sit-to-Laybed": "Transition-Sit-to-LayBed",
    "LayBed-Stationary": "Lying-Stationary",
    "LayFloor-Stationary": "Lying-Stationary",
}


@dataclass(frozen=True)
class SessionSpec:
    session: str
    origin_session: Path | None
    stream_paths: dict[str, Path]


@dataclass(frozen=True)
class SessionExamples:
    features: list[list[float]]
    labels: list[int]
    frame_indices: list[int]
    sessions: list[str]
    skipped: dict[str, int]


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.replace(":", ",").split(",") if item.strip()]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_label_map(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"{path} has no labels")
    return labels


def read_manifest(path: Path, streams: list[str], require_origin: bool) -> list[SessionSpec]:
    rows: list[SessionSpec] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        missing = [stream for stream in streams if stream not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing stream columns: {', '.join(missing)}")
        if require_origin and "origin_session" not in reader.fieldnames:
            raise ValueError(f"{path} must contain origin_session")

        for row_index, row in enumerate(reader, start=2):
            session = str(row.get("session") or f"row_{row_index}").strip()
            origin_text = str(row.get("origin_session") or "").strip()
            origin_session = Path(origin_text) if origin_text else None
            if require_origin and origin_session is None:
                raise ValueError(f"{path}:{row_index} has no origin_session")

            stream_paths = {}
            for stream in streams:
                value = str(row.get(stream) or "").strip()
                if not value:
                    raise ValueError(f"{path}:{row_index} has no path for stream {stream}")
                stream_paths[stream] = Path(value)

            rows.append(
                SessionSpec(
                    session=session,
                    origin_session=origin_session,
                    stream_paths=stream_paths,
                )
            )

    if not rows:
        raise ValueError(f"{path} has no sessions")
    return rows


def read_prediction_rows(path: Path, labels: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        required = {"frame_index", "prediction_id", "contributing_windows"}
        missing_required = sorted(required.difference(reader.fieldnames))
        if missing_required:
            raise ValueError(f"{path} is missing columns: {', '.join(missing_required)}")

        score_columns = [f"score_{label}" for label in labels]
        missing_scores = [column for column in score_columns if column not in reader.fieldnames]
        if missing_scores:
            raise ValueError(f"{path} is missing score columns: {', '.join(missing_scores[:4])}")

        return [row for row in reader]


def clean_label(label: Any, valid_labels: set[str]) -> str | None:
    if label is None:
        return None

    text = str(label).strip()
    if not text or text in DROP_LABELS:
        return None

    text = RENAME_LABELS.get(text, text)
    return text if text in valid_labels else None


def read_frame_label_rows(label_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Frame" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise ValueError(f"{label_path} must contain Frame and Label columns")

        for row_index, row in enumerate(reader):
            rows.append(
                {
                    "row_index": row_index,
                    "frame": safe_int(row.get("Frame"), default=-1),
                    "label": row.get("Label"),
                }
            )

    return sorted(
        (row for row in rows if row["frame"] >= 0),
        key=lambda row: (row["frame"], row["row_index"]),
    )


def build_gt_timeline(origin_session: Path, total_frames: int, labels: list[str]) -> list[int | None]:
    label_path = origin_session / "frame_labels.csv"
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    valid_labels = set(labels)
    label_to_id = {label: index for index, label in enumerate(labels)}
    timeline: list[int | None] = [None] * total_frames
    rows = read_frame_label_rows(label_path)

    for index, row in enumerate(rows):
        start = int(row["frame"])
        if start >= total_frames:
            continue

        end = rows[index + 1]["frame"] - 1 if index + 1 < len(rows) else total_frames - 1
        end = min(end, total_frames - 1)
        if end < start:
            continue

        label = clean_label(row["label"], valid_labels)
        if label is None:
            continue

        label_id = label_to_id[label]
        for frame_index in range(max(0, start), end + 1):
            timeline[frame_index] = label_id

    return timeline


def stream_scores(row: dict[str, str], labels: list[str]) -> list[float] | None:
    values = [safe_float(row.get(f"score_{label}"), default=math.nan) for label in labels]
    if any(not math.isfinite(value) for value in values):
        return None
    return values


def row_has_model_prediction(row: dict[str, str]) -> bool:
    return safe_int(row.get("contributing_windows"), default=0) > 0 and safe_int(
        row.get("prediction_id"), default=-1
    ) >= 0


def aligned_stream_rows(spec: SessionSpec, labels: list[str], streams: list[str]) -> dict[str, list[dict[str, str]]]:
    rows_by_stream = {
        stream: read_prediction_rows(spec.stream_paths[stream], labels) for stream in streams
    }
    lengths = {stream: len(rows) for stream, rows in rows_by_stream.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{spec.session} stream prediction lengths differ: {lengths}")

    reference_stream = streams[0]
    reference_indices = [
        safe_int(row.get("frame_index"), default=-1) for row in rows_by_stream[reference_stream]
    ]
    for stream in streams[1:]:
        indices = [safe_int(row.get("frame_index"), default=-1) for row in rows_by_stream[stream]]
        if indices != reference_indices:
            raise ValueError(f"{spec.session} frame_index values differ for stream {stream}")

    return rows_by_stream


def load_session_examples(
    spec: SessionSpec,
    labels: list[str],
    streams: list[str],
    require_detection: bool,
    require_prediction: bool,
) -> SessionExamples:
    if spec.origin_session is None:
        raise ValueError(f"{spec.session} has no origin_session")

    rows_by_stream = aligned_stream_rows(spec, labels, streams)
    total_frames = len(rows_by_stream[streams[0]])
    gt_timeline = build_gt_timeline(spec.origin_session, total_frames, labels)

    features: list[list[float]] = []
    targets: list[int] = []
    frame_indices: list[int] = []
    sessions: list[str] = []
    skipped = {
        "no_gt": 0,
        "no_detection": 0,
        "no_prediction": 0,
        "bad_scores": 0,
    }

    for frame_index in range(total_frames):
        target = gt_timeline[frame_index]
        if target is None:
            skipped["no_gt"] += 1
            continue

        reference_row = rows_by_stream[streams[0]][frame_index]
        if require_detection and not truthy(reference_row.get("selected_detection")):
            skipped["no_detection"] += 1
            continue

        vector: list[float] = []
        valid = True
        for stream in streams:
            row = rows_by_stream[stream][frame_index]
            if require_prediction and not row_has_model_prediction(row):
                skipped["no_prediction"] += 1
                valid = False
                break

            scores = stream_scores(row, labels)
            if scores is None:
                skipped["bad_scores"] += 1
                valid = False
                break
            vector.extend(scores)

        if not valid:
            continue

        features.append(vector)
        targets.append(target)
        frame_indices.append(frame_index)
        sessions.append(spec.session)

    return SessionExamples(
        features=features,
        labels=targets,
        frame_indices=frame_indices,
        sessions=sessions,
        skipped=skipped,
    )


def make_gate_model(input_dim: int, hidden_dim: int, num_streams: int, dropout: float):
    import torch.nn as nn

    class GateMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_streams),
            )

        def forward(self, features):
            return self.net(features)

    return GateMLP()


def load_gate_checkpoint(path: Path, device: str):
    import torch

    checkpoint = torch.load(path, map_location=device)
    required = ["input_dim", "hidden_dim", "num_streams", "dropout", "state_dict"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"{path} is missing checkpoint keys: {', '.join(missing)}")

    gate = make_gate_model(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_streams=int(checkpoint["num_streams"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    gate.load_state_dict(checkpoint["state_dict"])
    gate.eval()
    return gate, checkpoint


def fuse_expert_probs(features, gate_logits, num_streams: int, num_classes: int):
    import torch
    import torch.nn.functional as F

    expert_probs = features.reshape(-1, num_streams, num_classes)
    alpha = F.softmax(gate_logits, dim=1)
    fused = torch.sum(alpha.unsqueeze(-1) * expert_probs, dim=1)
    return fused, alpha


def metrics_from_predictions(predictions: list[int], targets: list[int], num_classes: int) -> dict[str, float]:
    if not targets:
        return {"accuracy": 0.0, "mean_class_accuracy": 0.0}

    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    per_class = []
    for class_id in range(num_classes):
        total = sum(int(target == class_id) for target in targets)
        if total == 0:
            continue
        class_correct = sum(
            int(pred == target == class_id) for pred, target in zip(predictions, targets)
        )
        per_class.append(class_correct / total)

    return {
        "accuracy": correct / len(targets),
        "mean_class_accuracy": sum(per_class) / len(per_class) if per_class else 0.0,
    }


def json_safe(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [json_safe(value) for value in data]
    return data


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")
