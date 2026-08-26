"""Study 3.5: no-training uncertainty evaluation on actual S6 pseudo-labels.

This analysis evaluates the actual out-of-fold pseudo-labels used by the radar
adaptation experiments. It does not fit teachers, temperatures, MI scales, or
radar models; manual labels are read only to score pseudo-label correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.s6.common import LABELS, TeacherSpec, selected_specs  # noqa: E402


DEFAULT_PSEUDO_ROOT = Path("work_dirs/thesis/s6/pseudo_labels")
DEFAULT_OUT_DIR = Path("work_dirs/thesis/s3/study35_actual_pseudo_uncertainty")
DEFAULT_FILENAME = "fusion_mc_calibrated_soft_probabilities_mi_weighted_gamma1.npz"

REQUIRED_ARRAYS = (
    "subject_id",
    "recording_id",
    "window_start_frame",
    "window_end_frame",
    "center_frame",
    "manual_label_id",
    "manual_label_name",
    "label_group",
    "pseudo_label_id",
    "mutual_information",
    "mi_weight",
)
SCOPES = (
    ("overall", "Overall"),
    ("states", "States"),
    ("transitions", "Transitions"),
)
QUINTILE_LABELS = (
    "lowest 20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "highest 20%",
)


@dataclass(frozen=True)
class PseudoData:
    fold: str
    teacher: np.ndarray
    subject_id: np.ndarray
    recording_id: np.ndarray
    window_start_frame: np.ndarray
    window_end_frame: np.ndarray
    center_frame: np.ndarray
    manual_label_id: np.ndarray
    manual_label_name: np.ndarray
    label_group: np.ndarray
    pseudo_label_id: np.ndarray
    mutual_information: np.ndarray
    mi_weight: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.manual_label_id.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate uncertainty on actual out-of-fold S6 pseudo-labels with "
            "one aggregated dataset per outer fold."
        )
    )
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--teachers", nargs="+", default=["t1", "t2", "t3", "t4"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    ""
                    if isinstance(value, float) and math.isnan(value)
                    else f"{value:.10g}"
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def write_json(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def binary_auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positives, dtype=bool)
    if values.shape[0] != labels.shape[0]:
        raise ValueError("scores and positives must have the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("AUROC scores contain non-finite values")
    n_pos = int(np.count_nonzero(labels))
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = float(np.sum(ranks[labels]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sample_sd(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    return float(np.std(np.asarray(finite, dtype=np.float64), ddof=1))


def mean_value(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def format_pm(mean: float, sd: float, decimals: int) -> str:
    if not math.isfinite(mean):
        return "NA"
    if not math.isfinite(sd):
        return f"{mean:.{decimals}f} +/- NA"
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def summarize_metric(
    rows: list[dict[str, Any]],
    group_key: str,
    value_keys: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    group_values = list(dict.fromkeys(str(row[group_key]) for row in rows))
    output: list[dict[str, Any]] = []
    for group_value in group_values:
        group_rows = [row for row in rows if str(row[group_key]) == group_value]
        item: dict[str, Any] = {group_key: group_value, "n_folds": len(group_rows)}
        for key, decimals in value_keys:
            values = [float(row[key]) for row in group_rows]
            mean = mean_value(values)
            sd = sample_sd(values)
            item[key] = format_pm(mean, sd, decimals)
            item[f"{key}_mean"] = mean
            item[f"{key}_sd"] = sd
        output.append(item)
    return output


def label_kind(groups: np.ndarray) -> np.ndarray:
    kinds = np.full(groups.shape[0], "other", dtype=object)
    normalized = np.char.lower(groups.astype(str))
    kinds[np.isin(normalized, ["stationary", "walking", "state"])] = "state"
    kinds[normalized == "transition"] = "transition"
    return kinds


def scope_mask(data: PseudoData, scope: str) -> np.ndarray:
    kinds = label_kind(data.label_group)
    if scope == "overall":
        return np.ones(data.n_samples, dtype=bool)
    if scope == "states":
        return kinds == "state"
    if scope == "transitions":
        return kinds == "transition"
    raise ValueError(f"Unsupported scope: {scope}")


def sample_keys(data: PseudoData) -> list[tuple[str, str, int, int, int]]:
    return [
        (
            str(subject),
            str(recording),
            int(start),
            int(end),
            int(center),
        )
        for subject, recording, start, end, center in zip(
            data.subject_id,
            data.recording_id,
            data.window_start_frame,
            data.window_end_frame,
            data.center_frame,
        )
    ]


def load_teacher_data(path: Path, fold: str, spec: TeacherSpec) -> PseudoData:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as loaded:
        missing = [key for key in REQUIRED_ARRAYS if key not in loaded.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        arrays = {key: loaded[key] for key in REQUIRED_ARRAYS}
        probabilities = loaded["probabilities"] if "probabilities" in loaded.files else None

    manual_label_id = np.asarray(arrays["manual_label_id"], dtype=np.int64)
    n_samples = int(manual_label_id.shape[0])
    for key, values in arrays.items():
        if np.asarray(values).shape[0] != n_samples:
            raise ValueError(f"{path} array {key!r} length does not match manual_label_id")

    pseudo_label_id = np.asarray(arrays["pseudo_label_id"], dtype=np.int64)
    mutual_information = np.asarray(arrays["mutual_information"], dtype=np.float64)
    mi_weight = np.asarray(arrays["mi_weight"], dtype=np.float64)
    if not np.all(np.isfinite(mutual_information)):
        raise ValueError(f"{path} mutual_information contains non-finite values")
    if not np.all(np.isfinite(mi_weight)):
        raise ValueError(f"{path} mi_weight contains non-finite values")
    if np.any(mi_weight < 0.0):
        raise ValueError(f"{path} mi_weight contains negative values")
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape[0] != n_samples:
            raise ValueError(f"{path} probabilities length does not match manual_label_id")
        argmax = np.argmax(probabilities, axis=1).astype(np.int64)
        if not np.array_equal(argmax, pseudo_label_id):
            raise ValueError(f"{path} pseudo_label_id does not match argmax(probabilities)")

    subject_id = np.asarray(arrays["subject_id"]).astype(str)
    observed_subjects = set(subject_id.tolist())
    expected_subjects = set(spec.pseudo_subjects)
    if observed_subjects != expected_subjects:
        raise ValueError(
            f"{path} subjects {sorted(observed_subjects)} do not match expected "
            f"pseudo subjects {sorted(expected_subjects)}"
        )

    manual_label_name = np.asarray(arrays["manual_label_name"]).astype(str)
    invalid_manual = [
        (index, int(label_id), name)
        for index, (label_id, name) in enumerate(zip(manual_label_id, manual_label_name))
        if int(label_id) < 0 or int(label_id) >= len(LABELS) or LABELS[int(label_id)] != str(name)
    ]
    if invalid_manual:
        index, label_id, name = invalid_manual[0]
        raise ValueError(
            f"{path} manual label mismatch at row {index}: id={label_id}, name={name!r}"
        )

    return PseudoData(
        fold=fold,
        teacher=np.full(n_samples, spec.teacher, dtype=object),
        subject_id=subject_id,
        recording_id=np.asarray(arrays["recording_id"]).astype(str),
        window_start_frame=np.asarray(arrays["window_start_frame"], dtype=np.int64),
        window_end_frame=np.asarray(arrays["window_end_frame"], dtype=np.int64),
        center_frame=np.asarray(arrays["center_frame"], dtype=np.int64),
        manual_label_id=manual_label_id,
        manual_label_name=manual_label_name,
        label_group=np.asarray(arrays["label_group"]).astype(str),
        pseudo_label_id=pseudo_label_id,
        mutual_information=mutual_information,
        mi_weight=mi_weight,
    )


def concatenate_fold(fold: str, parts: list[PseudoData]) -> PseudoData:
    if not parts:
        raise ValueError(f"No teacher data for fold {fold}")
    return PseudoData(
        fold=fold,
        teacher=np.concatenate([part.teacher for part in parts]),
        subject_id=np.concatenate([part.subject_id for part in parts]),
        recording_id=np.concatenate([part.recording_id for part in parts]),
        window_start_frame=np.concatenate([part.window_start_frame for part in parts]),
        window_end_frame=np.concatenate([part.window_end_frame for part in parts]),
        center_frame=np.concatenate([part.center_frame for part in parts]),
        manual_label_id=np.concatenate([part.manual_label_id for part in parts]),
        manual_label_name=np.concatenate([part.manual_label_name for part in parts]),
        label_group=np.concatenate([part.label_group for part in parts]),
        pseudo_label_id=np.concatenate([part.pseudo_label_id for part in parts]),
        mutual_information=np.concatenate([part.mutual_information for part in parts]),
        mi_weight=np.concatenate([part.mi_weight for part in parts]),
    )


def validate_fold_dataset(fold: str, data: PseudoData, specs: list[TeacherSpec]) -> dict[str, Any]:
    expected_subjects = sorted({subject for spec in specs for subject in spec.pseudo_subjects})
    observed_subjects = sorted(set(data.subject_id.tolist()))
    if observed_subjects != expected_subjects:
        raise ValueError(
            f"fold_{fold} subjects {observed_subjects} do not match expected "
            f"outer-training pseudo-label subjects {expected_subjects}"
        )

    seen: dict[tuple[str, str, int, int, int], str] = {}
    duplicates = []
    for key, teacher in zip(sample_keys(data), data.teacher):
        if key in seen:
            duplicates.append({"key": key, "first_teacher": seen[key], "second_teacher": str(teacher)})
        else:
            seen[key] = str(teacher)
    if duplicates:
        preview = duplicates[:3]
        raise ValueError(f"fold_{fold} has duplicate pseudo-label sample keys: {preview}")

    kinds = label_kind(data.label_group)
    teacher_counts = {
        str(teacher): int(np.count_nonzero(data.teacher == teacher))
        for teacher in sorted(set(data.teacher.tolist()))
    }
    subject_counts = {
        str(subject): int(np.count_nonzero(data.subject_id == subject))
        for subject in observed_subjects
    }
    return {
        "fold": f"fold_{fold}",
        "n_samples": data.n_samples,
        "n_teachers": len(teacher_counts),
        "teacher_counts": teacher_counts,
        "expected_outer_training_subjects": expected_subjects,
        "observed_subjects": observed_subjects,
        "subject_counts": subject_counts,
        "state_windows": int(np.count_nonzero(kinds == "state")),
        "transition_windows": int(np.count_nonzero(kinds == "transition")),
        "other_windows": int(np.count_nonzero(kinds == "other")),
    }


def load_fold_datasets(
    pseudo_root: Path,
    filename: str,
    folds: list[str],
    teachers: list[str],
) -> tuple[list[PseudoData], dict[str, Any]]:
    specs = selected_specs(folds, teachers)
    by_fold: dict[str, list[TeacherSpec]] = defaultdict(list)
    for spec in specs:
        by_fold[spec.fold].append(spec)

    datasets: list[PseudoData] = []
    diagnostics = {
        "source_files": [],
        "folds": {},
    }
    for fold in sorted(by_fold):
        parts = []
        for spec in sorted(by_fold[fold], key=lambda item: item.teacher):
            path = pseudo_root / spec.fold_dir / spec.teacher / "fusion_1to1" / filename
            part = load_teacher_data(path, fold, spec)
            parts.append(part)
            diagnostics["source_files"].append(str(path))
        data = concatenate_fold(fold, parts)
        diagnostics["folds"][f"fold_{fold}"] = validate_fold_dataset(fold, data, by_fold[fold])
        datasets.append(data)
    return datasets, diagnostics


def effective_error_rows(folds: list[PseudoData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in folds:
        errors = (data.pseudo_label_id != data.manual_label_id).astype(np.float64)
        for scope, label in SCOPES:
            mask = scope_mask(data, scope)
            n_samples = int(np.count_nonzero(mask))
            if n_samples == 0:
                error_rate = float("nan")
                weighted_error_rate = float("nan")
                sum_weights = float("nan")
            else:
                weights = data.mi_weight[mask]
                error_rate = float(np.mean(errors[mask]))
                sum_weights = float(np.sum(weights))
                weighted_error_rate = (
                    float(np.sum(weights * errors[mask]) / sum_weights)
                    if sum_weights > 0.0
                    else float("nan")
                )
            rows.append(
                {
                    "fold": data.fold,
                    "scope": label,
                    "n_samples": n_samples,
                    "pseudo_label_error_rate": error_rate,
                    "pseudo_label_error_percent": 100.0 * error_rate,
                    "effective_weighted_pseudo_label_error_rate": weighted_error_rate,
                    "effective_weighted_pseudo_label_error_percent": 100.0 * weighted_error_rate,
                    "weighted_minus_unweighted_error_rate": weighted_error_rate - error_rate,
                    "weighted_minus_unweighted_error_percent_points": 100.0
                    * (weighted_error_rate - error_rate),
                    "sum_weights": sum_weights,
                }
            )
    return rows


def auroc_rows(folds: list[PseudoData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in folds:
        errors = data.pseudo_label_id != data.manual_label_id
        for scope, label in SCOPES:
            mask = scope_mask(data, scope)
            n_samples = int(np.count_nonzero(mask))
            n_errors = int(np.count_nonzero(errors[mask]))
            n_correct = int(n_samples - n_errors)
            rows.append(
                {
                    "fold": data.fold,
                    "scope": label,
                    "n_samples": n_samples,
                    "n_errors": n_errors,
                    "n_correct": n_correct,
                    "error_auroc_mi_score": binary_auroc(data.mutual_information[mask], errors[mask])
                    if n_samples
                    else float("nan"),
                }
            )
    return rows


def correctness_weight_rows(folds: list[PseudoData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in folds:
        correct = data.pseudo_label_id == data.manual_label_id
        for scope, label in SCOPES:
            mask = scope_mask(data, scope)
            correct_mask = mask & correct
            incorrect_mask = mask & ~correct
            mean_correct = (
                float(np.mean(data.mi_weight[correct_mask]))
                if np.any(correct_mask)
                else float("nan")
            )
            mean_incorrect = (
                float(np.mean(data.mi_weight[incorrect_mask]))
                if np.any(incorrect_mask)
                else float("nan")
            )
            rows.append(
                {
                    "fold": data.fold,
                    "group": label,
                    "n_correct": int(np.count_nonzero(correct_mask)),
                    "n_incorrect": int(np.count_nonzero(incorrect_mask)),
                    "mean_weight_correct": mean_correct,
                    "mean_weight_incorrect": mean_incorrect,
                    "incorrect_minus_correct_weight": mean_incorrect - mean_correct,
                }
            )
    return rows


def quintile_rows(folds: list[PseudoData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data in folds:
        correct = data.pseudo_label_id == data.manual_label_id
        for scope, label in (("overall", "All windows"), ("transitions", "Transition windows")):
            mask = scope_mask(data, scope)
            indices = np.flatnonzero(mask)
            order = indices[np.argsort(data.mi_weight[indices], kind="mergesort")]
            for quintile_index, group_indices in enumerate(np.array_split(order, 5), start=1):
                if group_indices.size == 0:
                    accuracy = float("nan")
                    mean_weight = float("nan")
                    min_weight = float("nan")
                    max_weight = float("nan")
                else:
                    accuracy = float(np.mean(correct[group_indices]))
                    mean_weight = float(np.mean(data.mi_weight[group_indices]))
                    min_weight = float(np.min(data.mi_weight[group_indices]))
                    max_weight = float(np.max(data.mi_weight[group_indices]))
                rows.append(
                    {
                        "fold": data.fold,
                        "scope": label,
                        "quintile": quintile_index,
                        "quintile_label": QUINTILE_LABELS[quintile_index - 1],
                        "n_samples": int(group_indices.size),
                        "mean_weight": mean_weight,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                        "pseudo_label_accuracy": accuracy,
                        "pseudo_label_accuracy_percent": 100.0 * accuracy,
                    }
                )
    return rows


def summarize_error_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = summarize_metric(
        rows,
        "scope",
        [
            ("pseudo_label_error_percent", 2),
            ("effective_weighted_pseudo_label_error_percent", 2),
            ("weighted_minus_unweighted_error_percent_points", 2),
        ],
    )
    for row in summary:
        row["effective_weighted_error_lower_than_unweighted"] = (
            float(row["weighted_minus_unweighted_error_percent_points_mean"]) < 0.0
            if math.isfinite(float(row["weighted_minus_unweighted_error_percent_points_mean"]))
            else ""
        )
    return summary


def summarize_quintiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = list(dict.fromkeys(str(row["scope"]) for row in rows))
    for scope in scopes:
        for quintile in range(1, 6):
            group_rows = [
                row for row in rows
                if str(row["scope"]) == scope and int(row["quintile"]) == quintile
            ]
            accuracies = [float(row["pseudo_label_accuracy_percent"]) for row in group_rows]
            weights = [float(row["mean_weight"]) for row in group_rows]
            accuracy_mean = mean_value(accuracies)
            accuracy_sd = sample_sd(accuracies)
            weight_mean = mean_value(weights)
            weight_sd = sample_sd(weights)
            output.append(
                {
                    "scope": scope,
                    "quintile": quintile,
                    "quintile_label": QUINTILE_LABELS[quintile - 1],
                    "n_folds": len(group_rows),
                    "mean_weight": format_pm(weight_mean, weight_sd, 4),
                    "mean_weight_mean": weight_mean,
                    "mean_weight_sd": weight_sd,
                    "pseudo_label_accuracy_percent": format_pm(accuracy_mean, accuracy_sd, 2),
                    "pseudo_label_accuracy_percent_mean": accuracy_mean,
                    "pseudo_label_accuracy_percent_sd": accuracy_sd,
                }
            )
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(name for name, _key in columns) + " |"
    divider = "| " + " | ".join("---" for _name, _key in columns) + " |"
    body = [
        "| " + " | ".join(str(row[key]) for _name, key in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def make_quintile_plot(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")

    width = 900
    height = 540
    left = 82
    right = 210
    top = 38
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"All windows": "#246b8f", "Transition windows": "#b65f2a"}

    y_values: list[float] = []
    series = []
    for scope in ("All windows", "Transition windows"):
        scope_rows = sorted(
            [row for row in rows if str(row["scope"]) == scope],
            key=lambda row: int(row["quintile"]),
        )
        x = np.asarray([int(row["quintile"]) for row in scope_rows], dtype=np.float64)
        y = np.asarray([float(row["pseudo_label_accuracy_percent_mean"]) for row in scope_rows], dtype=np.float64)
        sd = np.asarray([float(row["pseudo_label_accuracy_percent_sd"]) for row in scope_rows], dtype=np.float64)
        y_values.extend((y - np.nan_to_num(sd, nan=0.0)).tolist())
        y_values.extend((y + np.nan_to_num(sd, nan=0.0)).tolist())
        series.append((scope, x, y, sd))

    y_min = max(0.0, math.floor(min(y_values) - 2.0))
    y_max = min(100.0, math.ceil(max(y_values) + 2.0))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - 1.0) / 4.0 * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
            '.axis{stroke:#222;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}'
            '.label{font-size:16px}.tick{font-size:12px}.legend{font-size:13px}</style>'
        ),
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y_pos = sy(float(tick))
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    for quintile, label in enumerate(QUINTILE_LABELS, start=1):
        x_pos = sx(float(quintile))
        parts.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{x_pos:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{quintile}</text>')
        parts.append(f'<text class="tick" x="{x_pos:.2f}" y="{top + plot_h + 43}" text-anchor="middle">{label}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<text class="label" x="{left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle">'
        "Reliability-weight quintile</text>"
    )
    parts.append(
        f'<text class="label" x="24" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {top + plot_h / 2:.2f})">Pseudo-label accuracy (%)</text>'
    )

    for scope, x, y, sd in series:
        color = colors[scope]
        line = " ".join(f"{sx(float(xi)):.2f},{sy(float(yi)):.2f}" for xi, yi in zip(x, y))
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xi, yi, sdi in zip(x, y, sd):
            px = sx(float(xi))
            py = sy(float(yi))
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{color}"/>')
            if math.isfinite(float(sdi)):
                err_top = sy(min(y_max, float(yi + sdi)))
                err_bottom = sy(max(y_min, float(yi - sdi)))
                parts.append(f'<line x1="{px:.2f}" y1="{err_top:.2f}" x2="{px:.2f}" y2="{err_bottom:.2f}" stroke="{color}" stroke-width="1.4"/>')
                parts.append(f'<line x1="{px - 8:.2f}" y1="{err_top:.2f}" x2="{px + 8:.2f}" y2="{err_top:.2f}" stroke="{color}" stroke-width="1.4"/>')
                parts.append(f'<line x1="{px - 8:.2f}" y1="{err_bottom:.2f}" x2="{px + 8:.2f}" y2="{err_bottom:.2f}" stroke="{color}" stroke-width="1.4"/>')

    legend_x = left + plot_w + 28
    legend_y = top + 20
    for index, scope in enumerate(("All windows", "Transition windows")):
        y_pos = legend_y + index * 30
        color = colors[scope]
        parts.append(f'<line x1="{legend_x}" y1="{y_pos}" x2="{legend_x + 26}" y2="{y_pos}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 13}" cy="{y_pos}" r="4.5" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 36}" y="{y_pos + 5}">{scope}</text>')
    parts.append("</svg>")
    write_text(path, "\n".join(parts), overwrite)


def write_report(
    path: Path,
    auroc_summary: list[dict[str, Any]],
    error_summary: list[dict[str, Any]],
    weight_summary: list[dict[str, Any]],
    quintile_summary: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    overwrite: bool,
) -> None:
    text = [
        "# Study 3.5: Actual Out-of-Fold Pseudo-Label Uncertainty",
        "",
        "Source: fused S6 inner-teacher pseudo-label files under "
        "`work_dirs/thesis/s6/pseudo_labels/fold_*/t*/fusion_1to1/`.",
        "",
        "Protocol: for each outer split, the four inner-teacher out-of-fold prediction "
        "files were concatenated into one complete pseudo-label dataset. Summary cells "
        "are mean +/- sample SD across outer folds A, B, and C. The 12 inner teachers "
        "are not treated as independent evaluation folds.",
        "",
        "No fitting was performed in this script. Manual labels are used only to evaluate "
        "pseudo-label correctness; they are not used to fit teachers, temperatures, MI "
        "scales, weights, or radar models.",
        "",
        "## Error AUROC",
        "",
        "Error score: calibrated MC mutual information. Positive label: "
        "`1[pseudo_label_id != manual_label_id]`.",
        "",
        markdown_table(
            auroc_summary,
            [
                ("Group", "scope"),
                ("Error AUROC higher", "error_auroc_mi_score"),
            ],
        ),
        "",
        "## Effective Weighted Pseudo-Label Error Rate",
        "",
        "The ordinary pseudo-label error rate is unchanged. The weighted value reports "
        "`sum_i w_i e_i / sum_i w_i`, i.e. the effective training mass assigned to "
        "incorrect pseudo-labels.",
        "",
        markdown_table(
            error_summary,
            [
                ("Group", "scope"),
                ("Unweighted R (%)", "pseudo_label_error_percent"),
                ("Effective weighted pseudo-label error rate Rw (%)", "effective_weighted_pseudo_label_error_percent"),
                ("Rw - R (pp)", "weighted_minus_unweighted_error_percent_points"),
            ],
        ),
        "",
        "## Correct-Versus-Incorrect Weight",
        "",
        markdown_table(
            weight_summary,
            [
                ("Group", "group"),
                ("Mean weight for correct labels", "mean_weight_correct"),
                ("Mean weight for incorrect labels", "mean_weight_incorrect"),
            ],
        ),
        "",
        "## Accuracy By Reliability-Weight Quintile",
        "",
        "Figure: `study35_accuracy_by_weight_quintile.svg`.",
        "",
        markdown_table(
            quintile_summary,
            [
                ("Scope", "scope"),
                ("Quintile", "quintile_label"),
                ("Mean weight", "mean_weight"),
                ("Pseudo-label accuracy (%)", "pseudo_label_accuracy_percent"),
            ],
        ),
        "",
        "## Diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2),
        "```",
        "",
    ]
    write_text(path, "\n".join(text), overwrite)


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds, diagnostics = load_fold_datasets(
        pseudo_root=args.pseudo_root,
        filename=args.filename,
        folds=args.folds,
        teachers=args.teachers,
    )

    auroc_by_fold = auroc_rows(folds)
    error_by_fold = effective_error_rows(folds)
    weight_by_fold = correctness_weight_rows(folds)
    quintile_by_fold = quintile_rows(folds)

    auroc_summary = summarize_metric(
        auroc_by_fold,
        "scope",
        [("error_auroc_mi_score", 4)],
    )
    error_summary = summarize_error_rates(error_by_fold)
    weight_summary = summarize_metric(
        weight_by_fold,
        "group",
        [
            ("mean_weight_correct", 4),
            ("mean_weight_incorrect", 4),
            ("incorrect_minus_correct_weight", 4),
        ],
    )
    quintile_summary = summarize_quintiles(quintile_by_fold)

    diagnostics.update(
        {
            "study": "Study 3.5",
            "analysis": "actual_out_of_fold_pseudo_label_uncertainty",
            "pseudo_root": str(args.pseudo_root),
            "source_filename": args.filename,
            "aggregation_unit": "outer fold",
            "outer_folds": [data.fold.upper() for data in folds],
            "inner_teacher_outputs_per_outer_fold": sorted(set(args.teachers)),
            "fold_summary_rule": "compute metrics after concatenating t1-t4 within each outer fold; report mean +/- sample SD across folds A, B, and C",
            "manual_label_use": "evaluation only",
            "training_or_fitting_performed": False,
            "teacher_temperature_mi_scale_or_radar_model_fit_by_this_script": False,
            "error_definition": "e_i = 1[pseudo_label_id_i != manual_label_id_i]",
            "error_score": "mutual_information",
            "weight_definition": "precomputed mi_weight from S6 pseudo-label export",
            "state_definition": "manual label_group in {stationary, walking}",
            "transition_definition": "manual label_group == transition, including Falling",
            "quintile_policy": "within each outer fold and scope, sort by mi_weight ascending and split into five equal-count groups",
        }
    )

    outputs = {
        "auroc_by_fold": args.out_dir / "study35_error_auroc_by_fold.csv",
        "auroc_summary": args.out_dir / "study35_error_auroc_summary.csv",
        "effective_error_by_fold": args.out_dir / "study35_effective_weighted_error_by_fold.csv",
        "effective_error_summary": args.out_dir / "study35_effective_weighted_error_summary.csv",
        "correctness_weight_by_fold": args.out_dir / "study35_correct_vs_incorrect_weight_by_fold.csv",
        "correctness_weight_summary": args.out_dir / "study35_correct_vs_incorrect_weight_summary.csv",
        "quintile_by_fold": args.out_dir / "study35_accuracy_by_weight_quintile_by_fold.csv",
        "quintile_summary": args.out_dir / "study35_accuracy_by_weight_quintile_summary.csv",
        "quintile_plot": args.out_dir / "study35_accuracy_by_weight_quintile.svg",
        "diagnostics": args.out_dir / "study35_diagnostics.json",
        "report": args.out_dir / "study35_report.md",
    }

    write_csv(outputs["auroc_by_fold"], auroc_by_fold, args.overwrite)
    write_csv(outputs["auroc_summary"], auroc_summary, args.overwrite)
    write_csv(outputs["effective_error_by_fold"], error_by_fold, args.overwrite)
    write_csv(outputs["effective_error_summary"], error_summary, args.overwrite)
    write_csv(outputs["correctness_weight_by_fold"], weight_by_fold, args.overwrite)
    write_csv(outputs["correctness_weight_summary"], weight_summary, args.overwrite)
    write_csv(outputs["quintile_by_fold"], quintile_by_fold, args.overwrite)
    write_csv(outputs["quintile_summary"], quintile_summary, args.overwrite)
    make_quintile_plot(outputs["quintile_plot"], quintile_summary, args.overwrite)
    write_json(outputs["diagnostics"], diagnostics, args.overwrite)
    write_report(
        outputs["report"],
        auroc_summary,
        error_summary,
        weight_summary,
        quintile_summary,
        diagnostics,
        args.overwrite,
    )

    return {
        "outputs": {key: str(value) for key, value in outputs.items()},
        "auroc_summary": auroc_summary,
        "error_summary": error_summary,
        "weight_summary": weight_summary,
        "quintile_summary": quintile_summary,
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    result = build_outputs(args)
    print(f"[DONE] wrote Study 3.5 outputs to {args.out_dir}")
    for row in result["auroc_summary"]:
        print(f"{row['scope']} Error AUROC: {row['error_auroc_mi_score']}")
    for row in result["error_summary"]:
        print(
            f"{row['scope']} R={row['pseudo_label_error_percent']}, "
            f"Rw={row['effective_weighted_pseudo_label_error_percent']}"
        )


if __name__ == "__main__":
    main()
