"""Shared E2 protocol, discovery, and score-file utilities."""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RADAR_TOOLS = REPO_ROOT / "tools" / "data" / "radar_v4"
for path in (REPO_ROOT, RADAR_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FPS = 30.0
WINDOW_SIZE = 60
STRIDE = 10
CENTER_OFFSET = 30

DEFAULT_JSONL_ROOT = Path("data/radar_v4/raw_jsonl/yolo26xpose")
DEFAULT_CONFIG_ROOT = Path("configs/posec3d/slowonly_r50_radarv4/8111")
DEFAULT_WORK_ROOT = Path("work_dirs/posec3d/8111")
DEFAULT_OUTPUT_DIR = Path("thesis/e2/results")
DEFAULT_SCORE_DIR = DEFAULT_OUTPUT_DIR / "scores"

# Keep this order aligned with tools/data/radar_v4/build_pyskl_pkl.py.
LABELS = [
    "Falling",
    "Lying-Stationary",
    "Sit-Stationary",
    "Transition-LayBed-to-Sit",
    "Transition-LayFloor-to-Stand",
    "Transition-Sit-to-LayBed",
    "Transition-Sit-to-Stand",
    "Transition-Stand-to-Sit",
    "Walking",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

DROP_LABELS = {
    "DELETE",
    "END",
    "Kneeling-Stationary",
    "Transition-Kneeling-to-Stand",
}
RENAME_LABELS = {
    "Lie-Stationary": "Lying-Stationary",
    "LayBed-Stationary": "Lying-Stationary",
    "LayFloor-Stationary": "Lying-Stationary",
    "Transition-Lie-to-Sit": "Transition-LayBed-to-Sit",
    "Transition-Lie-to-Stand": "Transition-LayFloor-to-Stand",
    "Transition-Sit-to-Lie": "Transition-Sit-to-LayBed",
    "Transition-Sit-to-Laybed": "Transition-Sit-to-LayBed",
}


@dataclass(frozen=True)
class FoldSpec:
    fold: str
    test_subject: str
    work_dir: Path
    config_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class ScoreRow:
    source: str
    score_type: str
    fold: str
    test_subject: str
    session: str
    jsonl_path: Path
    window_start: int
    window_end: int
    center_frame: int
    center_time_sec: float
    raw_gt_label: str
    gt_label: str
    gt_group: str
    valid_detection_frames: int
    selected_detection_center: bool
    scores: np.ndarray

    @property
    def pred_id(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def pred_label(self) -> str:
        return LABELS[self.pred_id]

    @property
    def pred_group(self) -> str:
        return label_to_group(self.pred_label)

    @property
    def confidence(self) -> float:
        return float(self.scores[self.pred_id])

    @property
    def valid_gt(self) -> bool:
        return bool(self.gt_label)

    @property
    def correct(self) -> bool:
        return self.valid_gt and self.gt_label == self.pred_label


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int


def protocol_metadata() -> dict[str, Any]:
    return {
        "fps": FPS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "center_offset": CENTER_OFFSET,
        "tail_window": False,
        "walk_folders": "excluded",
        "invalid_gt_labels": sorted(DROP_LABELS),
        "classification_scope": "window center samples with valid ground truth",
        "sequence_metric_scope": (
            "center-sampled sequences; invalid ground-truth centers break segments "
            "and are excluded"
        ),
    }


def clean_label(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text in DROP_LABELS:
        return None

    text = RENAME_LABELS.get(text, text)
    if text not in LABEL_TO_ID:
        return None
    return text


def label_to_group(label: str | None) -> str:
    if label is None:
        return "unlabeled"

    text = label.strip().lower()
    if not text or text in {"delete", "end"}:
        return "ignore"
    if "stationary" in text:
        return "stationary"
    if text == "falling" or "transition" in text:
        return "transition"
    if "walk" in text:
        return "walking"
    return "other"


def clean_group(raw_group: Any, cleaned_label: str | None) -> str:
    text = "" if raw_group is None else str(raw_group).strip().lower()
    if text in {"stationary", "transition", "walking", "ignore", "unlabeled", "other"}:
        if text not in {"ignore", "unlabeled"} and cleaned_label is not None:
            return text
        if cleaned_label is None:
            return text
    return label_to_group(cleaned_label)


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def percent(value: float) -> float:
    return 100.0 * value


def score_column_name(label: str) -> str:
    return "score_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def score_columns() -> list[str]:
    return [score_column_name(label) for label in LABELS]


def parse_fold_name(name: str) -> tuple[str, str] | None:
    match = re.search(r"_fold_([^_]+)_.*_test_([^_]+)$", name)
    if not match:
        return None
    return match.group(1).lower(), match.group(2).lower()


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(stream_dir: Path) -> Path:
    best = sorted(
        stream_dir.glob("best_macro_f1_epoch_*.pth"),
        key=lambda path: (checkpoint_epoch(path), path.name),
    )
    if best:
        epoch = checkpoint_epoch(best[-1])
        epoch_checkpoint = stream_dir / f"epoch_{epoch}.pth"
        if not epoch_checkpoint.exists():
            raise FileNotFoundError(
                f"Best checkpoint marker exists but matching epoch checkpoint is missing: "
                f"{best[-1]} -> {epoch_checkpoint}"
            )
        return epoch_checkpoint

    latest = stream_dir / "latest.pth"
    if latest.exists():
        return latest

    raise FileNotFoundError(f"No best_macro_f1_epoch_*.pth or latest.pth under {stream_dir}")


def discover_folds(config_root: Path, work_root: Path, stream: str) -> list[FoldSpec]:
    folds: list[FoldSpec] = []

    for work_dir in sorted(path for path in work_root.iterdir() if path.is_dir()):
        parsed = parse_fold_name(work_dir.name)
        if parsed is None:
            continue
        fold, test_subject = parsed
        config_path = config_root / f"fold_{fold}" / f"{stream}.py"
        checkpoint_path = find_checkpoint(work_dir / stream)

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config for fold {fold}: {config_path}")

        folds.append(
            FoldSpec(
                fold=fold,
                test_subject=test_subject,
                work_dir=work_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )

    if not folds:
        raise ValueError(f"No 8111 fold work directories found under {work_root}")
    return folds


def subject_from_session_name(session_name: str) -> str:
    parts = session_name.split("-")
    return parts[1].lower() if len(parts) >= 2 else ""


def is_walk_session(session_name: str) -> bool:
    return session_name.lower().endswith("-walk")


def discover_subject_sessions(jsonl_root: Path, subject: str) -> list[Path]:
    sessions: list[Path] = []
    for session_dir in sorted(path for path in jsonl_root.iterdir() if path.is_dir()):
        if subject_from_session_name(session_dir.name) != subject:
            continue
        if is_walk_session(session_dir.name):
            continue

        jsonl_paths = sorted(session_dir.glob("*.jsonl"))
        if len(jsonl_paths) != 1:
            raise ValueError(
                f"Expected exactly one JSONL under {session_dir}, found {len(jsonl_paths)}"
            )
        sessions.append(jsonl_paths[0])

    if not sessions:
        raise ValueError(f"No non-walk JSONL sessions found for subject {subject}")
    return sessions


def score_key(row: ScoreRow) -> tuple[str, str, str, int, int, int]:
    return (
        row.fold,
        row.test_subject,
        row.session,
        row.window_start,
        row.window_end,
        row.center_frame,
    )


def score_csv_columns() -> list[str]:
    return [
        "source",
        "score_type",
        "fold",
        "test_subject",
        "session",
        "jsonl_path",
        "window_start",
        "window_end",
        "center_frame",
        "center_time_sec",
        "raw_gt_label",
        "gt_label",
        "gt_group",
        "pred_label",
        "pred_group",
        "pred_id",
        "confidence",
        "correct",
        "valid_detection_frames",
        "selected_detection_center",
        *score_columns(),
    ]


def write_score_csv(path: Path, rows: list[ScoreRow], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_csv_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.source,
                    "score_type": row.score_type,
                    "fold": row.fold,
                    "test_subject": row.test_subject,
                    "session": row.session,
                    "jsonl_path": str(row.jsonl_path),
                    "window_start": row.window_start,
                    "window_end": row.window_end,
                    "center_frame": row.center_frame,
                    "center_time_sec": f"{row.center_time_sec:.6f}",
                    "raw_gt_label": row.raw_gt_label,
                    "gt_label": row.gt_label,
                    "gt_group": row.gt_group,
                    "pred_label": row.pred_label,
                    "pred_group": row.pred_group,
                    "pred_id": row.pred_id,
                    "confidence": f"{row.confidence:.8f}",
                    "correct": int(row.correct),
                    "valid_detection_frames": row.valid_detection_frames,
                    "selected_detection_center": int(row.selected_detection_center),
                    **{
                        score_column_name(label): f"{float(score):.8f}"
                        for label, score in zip(LABELS, row.scores)
                    },
                }
            )


def read_score_csv(path: Path) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        missing = [column for column in score_csv_columns() if column not in reader.fieldnames]
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            raise ValueError(f"{path} is missing score columns: {preview}")

        for row in reader:
            scores = np.asarray(
                [safe_float(row.get(score_column_name(label))) for label in LABELS],
                dtype=np.float32,
            )
            rows.append(
                ScoreRow(
                    source=str(row.get("source") or ""),
                    score_type=str(row.get("score_type") or "prob"),
                    fold=str(row.get("fold") or ""),
                    test_subject=str(row.get("test_subject") or ""),
                    session=str(row.get("session") or ""),
                    jsonl_path=Path(str(row.get("jsonl_path") or "")),
                    window_start=safe_int(row.get("window_start"), -1),
                    window_end=safe_int(row.get("window_end"), -1),
                    center_frame=safe_int(row.get("center_frame"), -1),
                    center_time_sec=safe_float(row.get("center_time_sec")),
                    raw_gt_label=str(row.get("raw_gt_label") or ""),
                    gt_label=str(row.get("gt_label") or ""),
                    gt_group=str(row.get("gt_group") or ""),
                    valid_detection_frames=safe_int(row.get("valid_detection_frames")),
                    selected_detection_center=truthy(row.get("selected_detection_center")),
                    scores=scores,
                )
            )

    if not rows:
        raise ValueError(f"{path} has no score rows")
    return rows


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    import json

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
