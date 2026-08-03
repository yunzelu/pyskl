"""Shared Study 6 teacher split metadata and path helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

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

CLIP_LEN = 60
S6_TAG = "8111teacher4_s6"
DEFAULT_TRIMMED_PKL_DIR = Path("data/radar_v4/pyskl") / S6_TAG
DEFAULT_CONFIG_DIR = Path("thesis/s6/configs")
DEFAULT_TRIMMED_CONFIG_DIR = DEFAULT_CONFIG_DIR / "trimmed"
DEFAULT_CONTINUOUS_CONFIG_DIR = DEFAULT_CONFIG_DIR / "continuous"
DEFAULT_CONTINUOUS_SOURCE_PKL = Path("data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_continuous.pkl")
DEFAULT_CONTINUOUS_TEACHER_PKL = Path("data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl")
DEFAULT_TRIMMED_WORK_ROOT = Path("work_dirs/thesis/s6/trimmed")
DEFAULT_CONTINUOUS_WORK_ROOT = Path("work_dirs/thesis/s6/continuous_hard")


@dataclass(frozen=True)
class TeacherSpec:
    fold: str
    teacher: str
    val_subject: str
    calibration_subject: str
    original_test_subject: str
    train_subjects: tuple[str, ...]
    pseudo_subjects: tuple[str, ...]

    @property
    def fold_dir(self) -> str:
        return f"fold_{self.fold}"

    @property
    def pseudo_slug(self) -> str:
        return "_".join(self.pseudo_subjects)

    @property
    def pkl_stem(self) -> str:
        return (
            f"radarv4_yolo26xpose_clip{CLIP_LEN}_{S6_TAG}_"
            f"{self.fold_dir}_{self.teacher}_val_{self.val_subject}_"
            f"calib_{self.calibration_subject}_pseudo_{self.pseudo_slug}"
        )

    @property
    def train_split(self) -> str:
        return f"{self.fold_dir}_{self.teacher}_train"

    @property
    def val_split(self) -> str:
        return f"{self.fold_dir}_{self.teacher}_val"

    @property
    def calib_split(self) -> str:
        return f"{self.fold_dir}_{self.teacher}_calib"

    @property
    def pseudo_split(self) -> str:
        return f"{self.fold_dir}_{self.teacher}_pseudo"

    @property
    def test_split(self) -> str:
        return f"{self.fold_dir}_{self.teacher}_test"


TEACHER_SPECS = (
    TeacherSpec(
        fold="a",
        teacher="t1",
        val_subject="han",
        calibration_subject="dengdeng",
        original_test_subject="chenzhe",
        train_subjects=("li", "mia", "rose", "saad", "xilai", "yunze"),
        pseudo_subjects=("hui", "jiadi"),
    ),
    TeacherSpec(
        fold="a",
        teacher="t2",
        val_subject="han",
        calibration_subject="dengdeng",
        original_test_subject="chenzhe",
        train_subjects=("hui", "jiadi", "rose", "saad", "xilai", "yunze"),
        pseudo_subjects=("li", "mia"),
    ),
    TeacherSpec(
        fold="a",
        teacher="t3",
        val_subject="han",
        calibration_subject="dengdeng",
        original_test_subject="chenzhe",
        train_subjects=("hui", "jiadi", "li", "mia", "xilai", "yunze"),
        pseudo_subjects=("rose", "saad"),
    ),
    TeacherSpec(
        fold="a",
        teacher="t4",
        val_subject="han",
        calibration_subject="dengdeng",
        original_test_subject="chenzhe",
        train_subjects=("hui", "jiadi", "li", "mia", "rose", "saad"),
        pseudo_subjects=("xilai", "yunze"),
    ),
    TeacherSpec(
        fold="b",
        teacher="t1",
        val_subject="mia",
        calibration_subject="li",
        original_test_subject="jiadi",
        train_subjects=("han", "hui", "rose", "saad", "xilai", "yunze"),
        pseudo_subjects=("chenzhe", "dengdeng"),
    ),
    TeacherSpec(
        fold="b",
        teacher="t2",
        val_subject="mia",
        calibration_subject="li",
        original_test_subject="jiadi",
        train_subjects=("chenzhe", "dengdeng", "rose", "saad", "xilai", "yunze"),
        pseudo_subjects=("han", "hui"),
    ),
    TeacherSpec(
        fold="b",
        teacher="t3",
        val_subject="mia",
        calibration_subject="li",
        original_test_subject="jiadi",
        train_subjects=("chenzhe", "dengdeng", "han", "hui", "xilai", "yunze"),
        pseudo_subjects=("rose", "saad"),
    ),
    TeacherSpec(
        fold="b",
        teacher="t4",
        val_subject="mia",
        calibration_subject="li",
        original_test_subject="jiadi",
        train_subjects=("chenzhe", "dengdeng", "han", "hui", "rose", "saad"),
        pseudo_subjects=("xilai", "yunze"),
    ),
    TeacherSpec(
        fold="c",
        teacher="t1",
        val_subject="yunze",
        calibration_subject="xilai",
        original_test_subject="saad",
        train_subjects=("han", "hui", "jiadi", "li", "mia", "rose"),
        pseudo_subjects=("chenzhe", "dengdeng"),
    ),
    TeacherSpec(
        fold="c",
        teacher="t2",
        val_subject="yunze",
        calibration_subject="xilai",
        original_test_subject="saad",
        train_subjects=("chenzhe", "dengdeng", "jiadi", "li", "mia", "rose"),
        pseudo_subjects=("han", "hui"),
    ),
    TeacherSpec(
        fold="c",
        teacher="t3",
        val_subject="yunze",
        calibration_subject="xilai",
        original_test_subject="saad",
        train_subjects=("chenzhe", "dengdeng", "han", "hui", "mia", "rose"),
        pseudo_subjects=("jiadi", "li"),
    ),
    TeacherSpec(
        fold="c",
        teacher="t4",
        val_subject="yunze",
        calibration_subject="xilai",
        original_test_subject="saad",
        train_subjects=("chenzhe", "dengdeng", "han", "hui", "jiadi", "li"),
        pseudo_subjects=("mia", "rose"),
    ),
)


def normalize_fold(value: str) -> str:
    return value.lower().replace("fold_", "")


def normalize_teacher(value: str) -> str:
    value = value.lower()
    if not re.fullmatch(r"t[1-4]", value):
        raise ValueError(f"Unsupported teacher id: {value!r}")
    return value


def selected_specs(folds: list[str] | None = None, teachers: list[str] | None = None) -> list[TeacherSpec]:
    fold_filter = None if not folds else {normalize_fold(item) for item in folds}
    teacher_filter = None if not teachers else {normalize_teacher(item) for item in teachers}
    specs = [
        spec for spec in TEACHER_SPECS
        if (fold_filter is None or spec.fold in fold_filter)
        and (teacher_filter is None or spec.teacher in teacher_filter)
    ]
    found_folds = {spec.fold for spec in specs}
    found_teachers = {spec.teacher for spec in specs}
    if fold_filter:
        missing = sorted(fold_filter - found_folds)
        if missing:
            raise ValueError(f"Unknown fold(s): {missing}")
    if teacher_filter:
        missing = sorted(teacher_filter - found_teachers)
        if missing:
            raise ValueError(f"Unknown teacher(s): {missing}")
    return specs


def trimmed_pkl_path(spec: TeacherSpec, pkl_dir: Path = DEFAULT_TRIMMED_PKL_DIR) -> Path:
    return pkl_dir / f"{spec.pkl_stem}.pkl"


def trimmed_summary_path(spec: TeacherSpec, pkl_dir: Path = DEFAULT_TRIMMED_PKL_DIR) -> Path:
    path = trimmed_pkl_path(spec, pkl_dir)
    return path.with_name(f"{path.stem}_summary.json")


def trimmed_config_path(
    spec: TeacherSpec,
    stream: str,
    config_dir: Path = DEFAULT_TRIMMED_CONFIG_DIR,
) -> Path:
    return config_dir / spec.fold_dir / spec.teacher / f"{stream}.py"


def continuous_config_path(
    spec: TeacherSpec,
    stream: str,
    config_dir: Path = DEFAULT_CONTINUOUS_CONFIG_DIR,
) -> Path:
    return config_dir / spec.fold_dir / spec.teacher / f"{stream}.py"


def trimmed_work_dir(
    spec: TeacherSpec,
    stream: str,
    work_root: Path = DEFAULT_TRIMMED_WORK_ROOT,
) -> Path:
    return work_root / spec.fold_dir / spec.teacher / stream


def continuous_work_dir(
    spec: TeacherSpec,
    stream: str,
    work_root: Path = DEFAULT_CONTINUOUS_WORK_ROOT,
) -> Path:
    return work_root / spec.fold_dir / spec.teacher / stream


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def count_summary_labels(summary_path: Path, split: str = "train") -> list[int] | None:
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    by_split = summary.get("samples_per_class_per_split", {})
    counts = by_split.get(split)
    if not isinstance(counts, dict):
        return None
    return [int(counts.get(label, 0)) for label in LABELS]

