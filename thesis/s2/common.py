"""Shared Study 2 protocol, fold discovery, and output helpers."""

from __future__ import annotations

import csv
import json
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

from thesis.e2.common import (  # noqa: E402
    CENTER_OFFSET,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_JSONL_ROOT,
    DEFAULT_WORK_ROOT,
    FPS,
    LABELS,
    LABEL_TO_ID,
    STRIDE,
    WINDOW_SIZE,
    checkpoint_epoch,
    clean_group,
    clean_label,
    find_checkpoint,
    label_to_group,
    percent,
    safe_float,
    safe_int,
    score_column_name,
    score_columns,
    write_json,
)

DEFAULT_STAGE1_PKL_DIR = Path("data/radar_v4/pyskl/8111")
DEFAULT_PKL_DIR = Path("data/radar_v4/pyskl/s2")
DEFAULT_S2_PKL = DEFAULT_PKL_DIR / "radarv4_yolo26xpose_clip60_s2_continuous.pkl"
DEFAULT_CONFIG_DIR = Path("thesis/s2/configs")
DEFAULT_OUTPUT_DIR = Path("work_dirs/thesis/s2")
DEFAULT_PREDICTION_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "eval"
DEFAULT_SELECTION_DIR = DEFAULT_OUTPUT_DIR / "selection"
DEFAULT_STAGE1_REPORT = DEFAULT_OUTPUT_DIR / "stage1_checkpoints.json"

METHOD_A = "A"
METHOD_B = "B"
METHOD_C = "C"
METHODS = (METHOD_A, METHOD_B, METHOD_C)
DEFAULT_ETAS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class S2FoldSpec:
    fold: str
    train_subjects: tuple[str, ...]
    val_subject: str
    calibration_subject: str | None
    test_subject: str
    stage1_name: str
    stage1_pkl_path: Path
    stage1_work_dir: Path
    stage1_config_dir: Path

    def split_key(self, split: str) -> str:
        if split not in {"train", "val", "calib", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        return f"fold_{self.fold}_{split}"

    def subject_for_split(self, split: str) -> str:
        if split == "val":
            return self.val_subject
        if split == "test":
            return self.test_subject
        if split == "calib" and self.calibration_subject is not None:
            return self.calibration_subject
        raise ValueError(f"Split {split!r} does not map to one subject")


def protocol_metadata(recording_scope: str | None = None) -> dict[str, Any]:
    if recording_scope is None:
        recording_scope = "non-walk sessions"
    return {
        "experiment": "S2",
        "fps": FPS,
        "window_size": WINDOW_SIZE,
        "stride_train_candidates": STRIDE,
        "stride_validation": STRIDE,
        "stride_test": STRIDE,
        "center_offset": CENTER_OFFSET,
        "center_convention": "center_frame = start_frame + window_size // 2",
        "tail_window": False,
        "temporal_padding": False,
        "default_recording_scope": recording_scope,
        "segmental_metric_resolution_seconds": STRIDE / FPS,
        "model_selection": {
            "stage1": "trimmed validation macro_f1 from existing E2 checkpoint",
            "stage2_primary": "continuous validation macro_f1",
            "stage2_tiebreaker": "continuous validation transition_macro_f1",
        },
        "excluded_methods": [
            "MC dropout",
            "temperature scaling",
            "Viterbi",
            "radar training",
            "pseudo-label selection",
        ],
    }


def eta_slug(eta: float) -> str:
    return f"eta{int(round(float(eta) * 100)):03d}"


def parse_eta_slug(slug: str) -> float:
    match = re.fullmatch(r"eta(\d{3})", slug)
    if not match:
        raise ValueError(f"Invalid eta slug: {slug}")
    return int(match.group(1)) / 100.0


def parse_stage1_fold_name(name: str) -> tuple[str, str, str, str] | None:
    match = re.search(
        r"_fold_([^_]+)_val_([^_]+)_calib_([^_]+)_test_([^_]+)$",
        name,
    )
    if not match:
        return None
    fold, val_subject, calibration_subject, test_subject = match.groups()
    return (
        fold.lower(),
        val_subject.lower(),
        calibration_subject.lower(),
        test_subject.lower(),
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_s2_folds(
    config_root: Path = DEFAULT_CONFIG_ROOT,
    work_root: Path = DEFAULT_WORK_ROOT,
    stage1_pkl_dir: Path = DEFAULT_STAGE1_PKL_DIR,
) -> list[S2FoldSpec]:
    folds: list[S2FoldSpec] = []
    for work_dir in sorted(path for path in work_root.iterdir() if path.is_dir()):
        parsed = parse_stage1_fold_name(work_dir.name)
        if parsed is None:
            continue

        fold, val_subject, calibration_subject, test_subject = parsed
        summary_path = stage1_pkl_dir / f"{work_dir.name}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing Stage-1 split summary: {summary_path}")
        summary = read_json(summary_path)
        split_subjects = summary.get("samples_per_subject_per_split", {})
        train_subjects = tuple(sorted(split_subjects.get("train", {}).keys()))
        if not train_subjects:
            raise ValueError(f"{summary_path} does not list train subjects")

        stage1_pkl = stage1_pkl_dir / f"{work_dir.name}.pkl"
        stage1_config_dir = config_root / f"fold_{fold}"
        if not stage1_pkl.exists():
            raise FileNotFoundError(f"Missing Stage-1 pkl: {stage1_pkl}")
        if not stage1_config_dir.exists():
            raise FileNotFoundError(f"Missing Stage-1 config dir: {stage1_config_dir}")

        folds.append(
            S2FoldSpec(
                fold=fold,
                train_subjects=train_subjects,
                val_subject=val_subject,
                calibration_subject=calibration_subject,
                test_subject=test_subject,
                stage1_name=work_dir.name,
                stage1_pkl_path=stage1_pkl,
                stage1_work_dir=work_dir,
                stage1_config_dir=stage1_config_dir,
            )
        )

    if not folds:
        raise ValueError(f"No Stage-1 folds found under {work_root}")
    return folds


def fold_by_name(folds: list[S2FoldSpec], fold_name: str) -> S2FoldSpec:
    fold_name = fold_name.lower().replace("fold_", "")
    for fold in folds:
        if fold.fold == fold_name:
            return fold
    raise ValueError(f"Unknown fold {fold_name!r}; available: {[fold.fold for fold in folds]}")


def stage1_config_path(fold: S2FoldSpec, stream: str) -> Path:
    path = fold.stage1_config_dir / f"{stream}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage-1 config: {path}")
    return path


def stage1_checkpoint_path(fold: S2FoldSpec, stream: str) -> Path:
    return find_checkpoint(fold.stage1_work_dir / stream)


def latest_stage1_log_json(fold: S2FoldSpec, stream: str) -> Path | None:
    log_paths = sorted((fold.stage1_work_dir / stream).glob("*.log.json"))
    return log_paths[-1] if log_paths else None


def read_log_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_epoch_from_checkpoint(path: Path) -> int:
    epoch = checkpoint_epoch(path)
    if epoch >= 0:
        return epoch
    match = re.search(r"(?:^|_)epoch_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def stage1_checkpoint_metadata(fold: S2FoldSpec, stream: str) -> dict[str, Any]:
    checkpoint = stage1_checkpoint_path(fold, stream)
    epoch = parse_epoch_from_checkpoint(checkpoint)
    record: dict[str, Any] | None = None

    log_path = latest_stage1_log_json(fold, stream)
    if log_path is not None:
        val_records = [
            item
            for item in read_log_json_records(log_path)
            if item.get("mode") == "val"
        ]
        for item in val_records:
            if int(item.get("epoch", -1)) == epoch:
                record = item
                break
        if record is None and val_records:
            record = max(
                val_records,
                key=lambda item: (
                    safe_float(item.get("macro_f1"), -1.0),
                    safe_float(item.get("top1_acc"), -1.0),
                    safe_int(item.get("epoch"), -1),
                ),
            )

    return {
        "fold": fold.fold,
        "stream": stream,
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "validation_accuracy": None if record is None else safe_float(record.get("top1_acc")),
        "validation_macro_f1": None if record is None else safe_float(record.get("macro_f1")),
        "validation_log_json": None if log_path is None else str(log_path),
        "stage1_config": str(stage1_config_path(fold, stream)),
        "stage1_pkl": str(fold.stage1_pkl_path),
    }


def stage2_work_dir(method: str, fold: str, stream: str, eta: float | None = None) -> Path:
    method = method.upper()
    if method == METHOD_C:
        if eta is None:
            raise ValueError("Method C requires eta")
        name = f"C_{eta_slug(eta)}"
    elif method == METHOD_B:
        name = "B_hard"
    else:
        raise ValueError(f"Unsupported Stage-2 method: {method}")
    return DEFAULT_OUTPUT_DIR / "train" / name / f"fold_{fold}" / stream


def s2_config_path(method: str, fold: str, stream: str, eta: float | None = None) -> Path:
    method = method.upper()
    if method == METHOD_A:
        filename = "posec3d_trimmed_baseline_A.py"
    elif method == METHOD_B:
        filename = "posec3d_continuous_hard_B.py"
    elif method == METHOD_C:
        if eta is None:
            raise ValueError("Method C requires eta")
        filename = f"posec3d_continuous_soft_C_{eta_slug(eta)}.py"
    else:
        raise ValueError(f"Unsupported method: {method}")
    return DEFAULT_CONFIG_DIR / f"fold_{fold}" / stream / filename


def default_prediction_path(method: str, stream: str = "joint") -> Path:
    suffix = f"_{stream}" if stream != "joint" else ""
    return DEFAULT_PREDICTION_DIR / f"predictions_{method.upper()}{suffix}.csv"


def default_metrics_path(method: str, stream: str = "joint") -> Path:
    suffix = f"_{stream}" if stream != "joint" else ""
    return DEFAULT_EVAL_DIR / f"metrics_{method.upper()}{suffix}.json"


def default_per_class_path(method: str, stream: str = "joint") -> Path:
    suffix = f"_{stream}" if stream != "joint" else ""
    return DEFAULT_EVAL_DIR / f"per_class_{method.upper()}{suffix}.csv"


def default_confusion_path(method: str, stream: str = "joint") -> Path:
    suffix = f"_{stream}" if stream != "joint" else ""
    return DEFAULT_EVAL_DIR / f"confusion_matrix_{method.upper()}{suffix}.csv"


def default_summary_table_path(stream: str = "joint") -> Path:
    suffix = f"_{stream}" if stream != "joint" else ""
    return DEFAULT_EVAL_DIR / f"summary_table{suffix}.csv"


def selection_path(method: str, stream: str = "joint") -> Path:
    return DEFAULT_SELECTION_DIR / f"selected_{method.upper()}_{stream}.json"


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64, copy=False)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / np.sum(exp_values, axis=-1, keepdims=True)).astype(np.float32)


def logit_column_name(label: str) -> str:
    return "logit_" + score_column_name(label).removeprefix("score_")


def prob_column_name(label: str) -> str:
    return "prob_" + score_column_name(label).removeprefix("score_")


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "CENTER_OFFSET",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_ROOT",
    "DEFAULT_ETAS",
    "DEFAULT_EVAL_DIR",
    "DEFAULT_JSONL_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PKL_DIR",
    "DEFAULT_PREDICTION_DIR",
    "DEFAULT_S2_PKL",
    "DEFAULT_SELECTION_DIR",
    "DEFAULT_STAGE1_PKL_DIR",
    "DEFAULT_STAGE1_REPORT",
    "DEFAULT_WORK_ROOT",
    "FPS",
    "LABELS",
    "LABEL_TO_ID",
    "METHOD_A",
    "METHOD_B",
    "METHOD_C",
    "METHODS",
    "S2FoldSpec",
    "STRIDE",
    "WINDOW_SIZE",
    "clean_group",
    "clean_label",
    "default_confusion_path",
    "default_metrics_path",
    "default_per_class_path",
    "default_prediction_path",
    "default_summary_table_path",
    "discover_s2_folds",
    "eta_slug",
    "fold_by_name",
    "label_to_group",
    "logit_column_name",
    "parse_eta_slug",
    "percent",
    "prob_column_name",
    "protocol_metadata",
    "read_json",
    "read_log_json_records",
    "safe_float",
    "safe_int",
    "s2_config_path",
    "score_column_name",
    "score_columns",
    "selection_path",
    "softmax",
    "stage1_checkpoint_metadata",
    "stage1_checkpoint_path",
    "stage1_config_path",
    "stage2_work_dir",
    "write_json",
    "write_rows_csv",
]
