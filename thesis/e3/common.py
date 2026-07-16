"""Shared helpers for E3 temperature calibration."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.e2.common import (  # noqa: E402
    CENTER_OFFSET,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_JSONL_ROOT,
    DEFAULT_WORK_ROOT,
    FPS,
    LABELS,
    STRIDE,
    WINDOW_SIZE,
    ScoreRow,
    clean_group,
    clean_label,
    discover_subject_sessions,
    find_checkpoint,
    label_to_group,
    percent,
    protocol_metadata as e2_protocol_metadata,
    read_score_csv,
    safe_float,
    safe_int,
    score_column_name,
    score_columns,
    write_json,
    write_score_csv,
)

DEFAULT_OUTPUT_DIR = Path("work_dirs/thesis/e3")
DEFAULT_LOGIT_DIR = DEFAULT_OUTPUT_DIR / "logits"
DEFAULT_TEMPERATURE_DIR = DEFAULT_OUTPUT_DIR / "temperatures"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUT_DIR / "analysis"


@dataclass(frozen=True)
class E3FoldSpec:
    fold: str
    val_subject: str
    calib_subject: str
    test_subject: str
    work_dir: Path
    config_path: Path
    checkpoint_path: Path


def protocol_metadata() -> dict[str, Any]:
    data = e2_protocol_metadata()
    data.update(
        {
            "experiment": "E3",
            "calibration": "scalar temperature per fold and stream",
            "temperature_formula": "softmax(logits / T)",
        }
    )
    return data


def parse_e3_fold_name(name: str) -> tuple[str, str, str, str] | None:
    match = re.search(
        r"_fold_([^_]+)_val_([^_]+)_calib_([^_]+)_test_([^_]+)$",
        name,
    )
    if not match:
        return None
    return tuple(item.lower() for item in match.groups())


def discover_e3_folds(config_root: Path, work_root: Path, stream: str) -> list[E3FoldSpec]:
    folds: list[E3FoldSpec] = []

    for work_dir in sorted(path for path in work_root.iterdir() if path.is_dir()):
        parsed = parse_e3_fold_name(work_dir.name)
        if parsed is None:
            continue
        fold, val_subject, calib_subject, test_subject = parsed
        config_path = config_root / f"fold_{fold}" / f"{stream}.py"
        checkpoint_path = find_checkpoint(work_dir / stream)

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config for fold {fold}: {config_path}")

        folds.append(
            E3FoldSpec(
                fold=fold,
                val_subject=val_subject,
                calib_subject=calib_subject,
                test_subject=test_subject,
                work_dir=work_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )

    if not folds:
        raise ValueError(f"No E3 fold work directories found under {work_root}")
    return folds


def subject_for_split(fold: E3FoldSpec, split: str) -> str:
    if split == "calib":
        return fold.calib_subject
    if split == "test":
        return fold.test_subject
    raise ValueError(f"Unsupported E3 split: {split}")


def default_logits_path(logit_dir: Path, stream: str, split: str) -> Path:
    return logit_dir / f"e3_{stream}_{split}_logits.csv"


def default_temperature_path(temperature_dir: Path, stream: str) -> Path:
    return temperature_dir / f"e3_{stream}_temperatures.json"


def state_or_transition(group: str) -> str:
    if group == "transition":
        return "transition"
    if group in {"stationary", "walking"}:
        return "state"
    return group or "unknown"


def softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

