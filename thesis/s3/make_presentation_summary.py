"""Build presentation-ready Study 3 summary tables and charts from saved S3 outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.e2.common import LABELS, label_to_group


FOLDS = ("a", "b", "c")
STREAMS = ("joint", "limb")
COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
SYSTEM_NAMES = {
    "deterministic_raw": "Deterministic raw",
    "deterministic_calibrated": "Deterministic calibrated",
    "mc_raw": "MC-dropout raw",
    "mc_calibrated": "MC-dropout calibrated",
}
SCORE_NAMES = {
    "one_minus_confidence": "1 - calibrated confidence",
    "predictive_entropy": "Calibrated predictive entropy",
    "mutual_information": "Calibrated MC mutual information",
}


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def validate_probabilities(probs: np.ndarray, name: str = "probabilities", tolerance: float = 1e-5) -> None:
    values = np.asarray(probs)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError(f"{name} values must be in [0, 1]")
    sums = values.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=tolerance, rtol=0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to one; max error={max_error}")


def mean_mc_probabilities(mc_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(mc_logits)
    if values.ndim != 3:
        raise ValueError(f"mc_logits must have shape [N, K, C], got {values.shape}")
    return softmax_np(values, temperature=temperature).mean(axis=1)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    picked = probabilities[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == targets).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(np.count_nonzero(mask))
        if count:
            acc = float(np.mean(correct[mask]))
            conf = float(np.mean(confidences[mask]))
            ece += count / len(targets) * abs(acc - conf)
    return float(ece)


def fit_temperature(
    mc_logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    max_iter: int = 80,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    logits = np.asarray(mc_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError(f"mc_logits must have shape [N, K, C], got {logits.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("mc_logits and labels have different lengths")

    def metrics_at(temperature: float) -> dict[str, float]:
        probs = mean_mc_probabilities(logits, temperature=temperature)
        return {
            "nll": negative_log_likelihood(probs, targets),
            "ece": expected_calibration_error(probs, targets, num_bins=num_bins),
            "brier": brier_score(probs, targets),
        }

    before = metrics_at(1.0)
    lower = math.log(float(min_temperature))
    upper = math.log(float(max_temperature))
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0
    left = lower + inv_phi_sq * (upper - lower)
    right = lower + inv_phi * (upper - lower)

    def objective(log_temperature: float) -> float:
        return metrics_at(math.exp(log_temperature))["nll"]

    left_value = objective(left)
    right_value = objective(right)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        if abs(upper - lower) <= tolerance:
            converged = True
            break
        if left_value < right_value:
            upper = right
            right = left
            right_value = left_value
            left = lower + inv_phi_sq * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + inv_phi * (upper - lower)
            right_value = objective(right)

    fitted = math.exp((lower + upper) / 2.0)
    after = metrics_at(fitted)
    status = "converged" if converged else "max_iter"
    warning = ""
    if after["nll"] > before["nll"] + 1e-8:
        warning = "Optimized temperature produced worse calibration NLL than T=1; falling back to T=1."
        fitted = 1.0
        after = before
        status = "fallback_identity_worse_than_identity"
    return {
        "temperature": float(fitted),
        "optimization_method": "golden_section_log_temperature",
        "convergence_status": status,
        "iterations": int(iterations),
        "bounds": [float(min_temperature), float(max_temperature)],
        "calibration_nll_before": float(before["nll"]),
        "calibration_nll_after": float(after["nll"]),
        "calibration_ece_before": float(before["ece"]),
        "calibration_ece_after": float(after["ece"]),
        "calibration_brier_before": float(before["brier"]),
        "calibration_brier_after": float(after["brier"]),
        "warning": warning,
    }


def predictive_quantities(prob_passes: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    values = np.asarray(prob_passes, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"prob_passes must have shape [N, K, C], got {values.shape}")
    validate_probabilities(values, name="prob_passes")
    p_bar = values.mean(axis=1)
    validate_probabilities(p_bar, name="predictive_mean")
    predictive_entropy = -np.sum(p_bar * np.log(np.clip(p_bar, eps, 1.0)), axis=1)
    per_pass_entropy = -np.sum(values * np.log(np.clip(values, eps, 1.0)), axis=2)
    expected_entropy = per_pass_entropy.mean(axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    pass_predictions = np.argmax(values, axis=2)
    modal_counts = np.zeros(values.shape[0], dtype=np.int64)
    for index, predictions in enumerate(pass_predictions):
        modal_counts[index] = int(np.max(np.bincount(predictions, minlength=values.shape[2])))
    variation_ratio = 1.0 - modal_counts.astype(np.float64) / values.shape[1]
    class_variance = np.var(values, axis=1)
    top2 = np.sort(p_bar, axis=1)[:, -2:]
    return {
        "probabilities": p_bar,
        "prediction": np.argmax(p_bar, axis=1).astype(np.int64),
        "confidence": np.max(p_bar, axis=1),
        "margin": top2[:, 1] - top2[:, 0],
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
        "class_variance": class_variance,
        "mean_probability_variance": class_variance.mean(axis=1),
    }


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    ece_bins: int = 15,
) -> dict[str, float]:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    validate_probabilities(probabilities)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    correct = predictions == targets
    return {
        "accuracy": float(np.mean(correct)) if len(targets) else 0.0,
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets, num_bins=ece_bins),
    }


def binary_auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positives, dtype=bool)
    n_pos = int(np.count_nonzero(labels))
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    pos_rank_sum = float(np.sum(ranks[labels]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class SplitArtifacts:
    keys: list[tuple[str, str, int, int, int, int]]
    labels: np.ndarray
    label_names: list[str]
    det_logits: np.ndarray
    mc_logits: np.ndarray | None


@dataclass(frozen=True)
class FoldFusion:
    fold: str
    calibration: SplitArtifacts
    test: SplitArtifacts
    det_temperature: dict[str, Any]
    mc_temperature: dict[str, Any]
    probabilities: dict[str, np.ndarray]
    mc_quantities: dict[str, np.ndarray]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sample_key(row: dict[str, str]) -> tuple[str, str, int, int, int, int]:
    return (
        row["subject_id"],
        row["recording_id"],
        int(row["window_start_frame"]),
        int(row["window_end_frame"]),
        int(row["center_frame"]),
        int(row["ground_truth_id"]),
    )


def labels_from_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    ids = np.asarray([int(row["ground_truth_id"]) for row in rows], dtype=np.int64)
    names = [row["ground_truth_label"] for row in rows]
    return ids, names


def load_stream_split(base_dir: Path, split: str) -> SplitArtifacts:
    det_path = base_dir / "deterministic_predictions.npz"
    mc_path = base_dir / f"{split}_mc_logits.npz"
    sample_path = base_dir / f"{split}_samples.csv"
    if not det_path.exists():
        raise FileNotFoundError(det_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    rows = read_csv_rows(sample_path)
    keys = [sample_key(row) for row in rows]
    labels, label_names = labels_from_rows(rows)

    with np.load(det_path, allow_pickle=False) as det:
        det_logits = np.asarray(det[f"{split}_logits"], dtype=np.float64)
        det_labels = np.asarray(det[f"{split}_labels"], dtype=np.int64)
    if det_logits.shape[0] != len(rows):
        raise ValueError(f"{det_path} {split}_logits length does not match {sample_path}")
    if not np.array_equal(labels, det_labels):
        raise ValueError(f"{det_path} {split}_labels do not match {sample_path}")

    mc_logits = None
    if mc_path.exists():
        with np.load(mc_path, allow_pickle=False) as mc:
            mc_logits = np.asarray(mc["mc_logits"], dtype=np.float64)
            mc_labels = np.asarray(mc["labels"], dtype=np.int64)
        if mc_logits.shape[0] != len(rows):
            raise ValueError(f"{mc_path} mc_logits length does not match {sample_path}")
        if not np.array_equal(labels, mc_labels):
            raise ValueError(f"{mc_path} labels do not match {sample_path}")

    return SplitArtifacts(
        keys=keys,
        labels=labels,
        label_names=label_names,
        det_logits=det_logits,
        mc_logits=mc_logits,
    )


def assert_aligned(fold: str, split: str, left: SplitArtifacts, right: SplitArtifacts) -> None:
    if left.keys != right.keys:
        raise ValueError(f"fold_{fold} {split} joint/limb sample keys are not aligned")
    if not np.array_equal(left.labels, right.labels):
        raise ValueError(f"fold_{fold} {split} joint/limb labels are not aligned")
    if left.label_names != right.label_names:
        raise ValueError(f"fold_{fold} {split} joint/limb label names are not aligned")


def fuse_split(joint: SplitArtifacts, limb: SplitArtifacts) -> SplitArtifacts:
    if joint.mc_logits is None or limb.mc_logits is None:
        mc_logits = None
    else:
        if joint.mc_logits.shape != limb.mc_logits.shape:
            raise ValueError("Joint and limb MC logits do not have the same shape")
        mc_logits = (joint.mc_logits + limb.mc_logits) / 2.0
    return SplitArtifacts(
        keys=joint.keys,
        labels=joint.labels,
        label_names=joint.label_names,
        det_logits=(joint.det_logits + limb.det_logits) / 2.0,
        mc_logits=mc_logits,
    )


def load_fold_fusion(root: Path, fold: str, ece_bins: int) -> FoldFusion:
    stream_data: dict[str, dict[str, SplitArtifacts]] = {}
    for stream in STREAMS:
        base_dir = root / f"fold_{fold}" / stream
        if not base_dir.exists():
            raise FileNotFoundError(base_dir)
        stream_data[stream] = {
            "calibration": load_stream_split(base_dir, "calibration"),
            "test": load_stream_split(base_dir, "test"),
        }

    assert_aligned(fold, "calibration", stream_data["joint"]["calibration"], stream_data["limb"]["calibration"])
    assert_aligned(fold, "test", stream_data["joint"]["test"], stream_data["limb"]["test"])

    calibration = fuse_split(stream_data["joint"]["calibration"], stream_data["limb"]["calibration"])
    test = fuse_split(stream_data["joint"]["test"], stream_data["limb"]["test"])
    if calibration.mc_logits is None or test.mc_logits is None:
        raise ValueError(f"fold_{fold} is missing calibration or test MC logits")

    det_temperature = fit_temperature(
        calibration.det_logits[:, None, :],
        calibration.labels,
        num_bins=ece_bins,
    )
    mc_temperature = fit_temperature(
        calibration.mc_logits,
        calibration.labels,
        num_bins=ece_bins,
    )

    det_t = float(det_temperature["temperature"])
    mc_t = float(mc_temperature["temperature"])
    deterministic_raw = softmax_np(test.det_logits)
    deterministic_calibrated = softmax_np(test.det_logits, temperature=det_t)
    mc_raw_passes = softmax_np(test.mc_logits)
    mc_cal_passes = softmax_np(test.mc_logits, temperature=mc_t)
    mc_raw = predictive_quantities(mc_raw_passes)
    mc_calibrated = predictive_quantities(mc_cal_passes)

    probabilities = {
        "deterministic_raw": deterministic_raw,
        "deterministic_calibrated": deterministic_calibrated,
        "mc_raw": mc_raw["probabilities"],
        "mc_calibrated": mc_calibrated["probabilities"],
    }
    for name, probs in probabilities.items():
        validate_probabilities(probs, name=f"fold_{fold}_{name}")

    return FoldFusion(
        fold=fold,
        calibration=calibration,
        test=test,
        det_temperature=det_temperature,
        mc_temperature=mc_temperature,
        probabilities=probabilities,
        mc_quantities=mc_calibrated,
    )


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def mean_value(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def format_pm(mean: float, sd: float, decimals: int) -> str:
    if math.isnan(sd):
        return f"{mean:.{decimals}f} +/- NA"
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def retained_indices(scores: np.ndarray, coverage: float) -> np.ndarray:
    n_samples = int(scores.shape[0])
    keep = max(1, int(round(n_samples * coverage)))
    order = np.argsort(scores, kind="mergesort")
    return np.sort(order[:keep])


def metric_rows(folds: list[FoldFusion], ece_bins: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_fold = []
    for fold in folds:
        labels = fold.test.labels
        for system_key, probs in fold.probabilities.items():
            metrics = classification_metrics(probs, labels, LABELS, ece_bins=ece_bins)
            by_fold.append(
                {
                    "fold": fold.fold,
                    "prediction_system": SYSTEM_NAMES[system_key],
                    "center_accuracy_percent": 100.0 * float(metrics["accuracy"]),
                    "nll": float(metrics["nll"]),
                    "ece_percent": 100.0 * float(metrics["ece"]),
                    "brier": float(metrics["brier"]),
                    "temperature": (
                        float(fold.det_temperature["temperature"])
                        if system_key == "deterministic_calibrated"
                        else float(fold.mc_temperature["temperature"])
                        if system_key == "mc_calibrated"
                        else 1.0
                    ),
                    "num_test_samples": int(labels.shape[0]),
                }
            )

    summary = []
    for system_key in SYSTEM_NAMES:
        name = SYSTEM_NAMES[system_key]
        rows = [row for row in by_fold if row["prediction_system"] == name]
        acc = [float(row["center_accuracy_percent"]) for row in rows]
        nll = [float(row["nll"]) for row in rows]
        ece = [float(row["ece_percent"]) for row in rows]
        brier = [float(row["brier"]) for row in rows]
        summary.append(
            {
                "prediction_system": name,
                "center_accuracy_percent": format_pm(mean_value(acc), sample_sd(acc), 2),
                "nll": format_pm(mean_value(nll), sample_sd(nll), 4),
                "ece_percent": format_pm(mean_value(ece), sample_sd(ece), 2),
                "center_accuracy_mean": mean_value(acc),
                "center_accuracy_sd": sample_sd(acc),
                "nll_mean": mean_value(nll),
                "nll_sd": sample_sd(nll),
                "ece_mean": mean_value(ece),
                "ece_sd": sample_sd(ece),
                "brier_mean": mean_value(brier),
                "brier_sd": sample_sd(brier),
            }
        )
    return by_fold, summary


def score_arrays(fold: FoldFusion) -> dict[str, np.ndarray]:
    quantities = fold.mc_quantities
    return {
        "one_minus_confidence": 1.0 - quantities["confidence"],
        "predictive_entropy": quantities["predictive_entropy"],
        "mutual_information": quantities["mutual_information"],
    }


def coverage_and_score_rows(
    folds: list[FoldFusion],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    coverage_by_fold = []
    score_by_fold = []
    retention_by_fold = []
    for fold in folds:
        labels = fold.test.labels
        predictions = fold.mc_quantities["prediction"]
        errors = predictions != labels
        groups = np.asarray([group_for_label_name(name) for name in fold.test.label_names])
        for score_key, scores in score_arrays(fold).items():
            error_auroc = binary_auroc(scores, errors)
            acc80 = None
            for coverage in COVERAGES:
                indices = retained_indices(scores, coverage)
                accuracy = float(np.mean(predictions[indices] == labels[indices]))
                if abs(coverage - 0.80) < 1e-12:
                    acc80 = accuracy
                    state_mask = groups == "state"
                    transition_mask = groups == "transition"
                    retained_mask = np.zeros(labels.shape[0], dtype=bool)
                    retained_mask[indices] = True
                    state_total = int(np.count_nonzero(state_mask))
                    transition_total = int(np.count_nonzero(transition_mask))
                    other_total = int(np.count_nonzero(~(state_mask | transition_mask)))
                    retention_by_fold.append(
                        {
                            "fold": fold.fold,
                            "score": SCORE_NAMES[score_key],
                            "state_retention_percent": (
                                100.0 * float(np.count_nonzero(retained_mask & state_mask)) / state_total
                                if state_total
                                else float("nan")
                            ),
                            "transition_retention_percent": (
                                100.0 * float(np.count_nonzero(retained_mask & transition_mask)) / transition_total
                                if transition_total
                                else float("nan")
                            ),
                            "state_total": state_total,
                            "transition_total": transition_total,
                            "other_total": other_total,
                        }
                    )
                coverage_by_fold.append(
                    {
                        "fold": fold.fold,
                        "score": SCORE_NAMES[score_key],
                        "coverage": coverage,
                        "coverage_percent": 100.0 * coverage,
                        "retained_samples": int(indices.shape[0]),
                        "accuracy_percent": 100.0 * accuracy,
                    }
                )
            if acc80 is None:
                raise RuntimeError("Coverage 0.80 was not evaluated")
            score_by_fold.append(
                {
                    "fold": fold.fold,
                    "score": SCORE_NAMES[score_key],
                    "error_auroc": error_auroc,
                    "accuracy_at_80_percent": 100.0 * acc80,
                }
            )
    return coverage_by_fold, score_by_fold, retention_by_fold


def group_for_label_name(label: str) -> str:
    group = label_to_group(label)
    if group in {"stationary", "walking"}:
        return "state"
    if group == "transition":
        return "transition"
    return group


def summarize_rows(
    rows: list[dict[str, Any]],
    group_key: str,
    value_keys: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    output = []
    for group_value in dict.fromkeys(row[group_key] for row in rows):
        group_rows = [row for row in rows if row[group_key] == group_value]
        item: dict[str, Any] = {group_key: group_value}
        for key, decimals in value_keys:
            values = [float(row[key]) for row in group_rows]
            item[key] = format_pm(mean_value(values), sample_sd(values), decimals)
            item[f"{key}_mean"] = mean_value(values)
            item[f"{key}_sd"] = sample_sd(values)
        output.append(item)
    return output


def summarize_curve(coverage_by_fold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for score in SCORE_NAMES.values():
        for coverage in COVERAGES:
            rows = [
                row
                for row in coverage_by_fold
                if row["score"] == score and abs(float(row["coverage"]) - coverage) < 1e-12
            ]
            values = [float(row["accuracy_percent"]) for row in rows]
            output.append(
                {
                    "score": score,
                    "coverage": coverage,
                    "coverage_percent": 100.0 * coverage,
                    "accuracy_percent": format_pm(mean_value(values), sample_sd(values), 2),
                    "accuracy_mean": mean_value(values),
                    "accuracy_sd": sample_sd(values),
                }
            )
    return output


def make_accuracy_coverage_plot(path: Path, curve_summary: list[dict[str, Any]]) -> None:
    width = 900
    height = 540
    left = 82
    right = 230
    top = 36
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        SCORE_NAMES["one_minus_confidence"]: "#2f6f9f",
        SCORE_NAMES["predictive_entropy"]: "#b45f06",
        SCORE_NAMES["mutual_information"]: "#38761d",
    }
    all_lower = []
    all_upper = []
    series: list[tuple[str, list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]] = []
    for score in SCORE_NAMES.values():
        rows = [row for row in curve_summary if row["score"] == score]
        rows = sorted(rows, key=lambda row: float(row["coverage_percent"]))
        x = np.asarray([float(row["coverage_percent"]) for row in rows])
        y = np.asarray([float(row["accuracy_mean"]) for row in rows])
        sd = np.asarray([float(row["accuracy_sd"]) for row in rows])
        all_lower.extend((y - sd).tolist())
        all_upper.extend((y + sd).tolist())
        series.append((score, rows, x, y, sd))

    y_min = max(0.0, math.floor(min(all_lower) - 1.0))
    y_max = min(100.0, math.ceil(max(all_upper) + 1.0))
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - 50.0) / 50.0 * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#222;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.label{font-size:16px}.tick{font-size:12px}.legend{font-size:13px}</style>',
    ]
    for tick in np.linspace(y_min, y_max, 5):
        y_pos = sy(float(tick))
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    for tick in (50, 60, 70, 80, 90, 95, 100):
        x_pos = sx(float(tick))
        parts.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{x_pos:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<text class="label" x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle">'
        "Coverage: pseudo-labels retained (%)</text>"
    )
    parts.append(
        f'<text class="label" x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
        'transform="rotate(-90 22 '
        f'{top + plot_h / 2:.2f})">Accuracy of retained pseudo-labels (%)</text>'
    )

    for score, _rows, x, y, sd in series:
        color = colors[score]
        upper_points = [(sx(float(xi)), sy(float(yi + sdi))) for xi, yi, sdi in zip(x, y, sd)]
        lower_points = [(sx(float(xi)), sy(float(yi - sdi))) for xi, yi, sdi in zip(x, y, sd)]
        polygon = " ".join(f"{px:.2f},{py:.2f}" for px, py in [*upper_points, *reversed(lower_points)])
        line = " ".join(f"{sx(float(xi)):.2f},{sy(float(yi)):.2f}" for xi, yi in zip(x, y))
        parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.16"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xi, yi in zip(x, y):
            parts.append(f'<circle cx="{sx(float(xi)):.2f}" cy="{sy(float(yi)):.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 28
    legend_y = top + 18
    for index, score in enumerate(SCORE_NAMES.values()):
        y_pos = legend_y + index * 30
        color = colors[score]
        parts.append(f'<line x1="{legend_x}" y1="{y_pos}" x2="{legend_x + 24}" y2="{y_pos}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 12}" cy="{y_pos}" r="4" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 34}" y="{y_pos + 5}">{html.escape(score)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_retention_plot(path: Path, retention_summary: list[dict[str, Any]]) -> None:
    width = 900
    height = 520
    left = 82
    right = 40
    top = 36
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    scores = [SCORE_NAMES[key] for key in ("one_minus_confidence", "predictive_entropy", "mutual_information")]
    state_mean = np.asarray(
        [float(next(row for row in retention_summary if row["score"] == score)["state_retention_percent_mean"])
         for score in scores]
    )
    state_sd = np.asarray(
        [float(next(row for row in retention_summary if row["score"] == score)["state_retention_percent_sd"])
         for score in scores]
    )
    transition_mean = np.asarray(
        [float(next(row for row in retention_summary if row["score"] == score)["transition_retention_percent_mean"])
         for score in scores]
    )
    transition_sd = np.asarray(
        [float(next(row for row in retention_summary if row["score"] == score)["transition_retention_percent_sd"])
         for score in scores]
    )

    def sx(index: int, offset: float) -> float:
        group_w = plot_w / len(scores)
        return left + group_w * (index + 0.5) + offset

    def sy(value: float) -> float:
        return top + (105.0 - value) / 105.0 * plot_h

    bar_w = 74
    colors = {"State": "#2f6f9f", "Transition": "#b45f06"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#222;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.label{font-size:16px}.tick{font-size:12px}.legend{font-size:13px}</style>',
    ]
    for tick in (0, 20, 40, 60, 80, 100):
        y_pos = sy(float(tick))
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(
        f'<text class="label" x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
        'transform="rotate(-90 22 '
        f'{top + plot_h / 2:.2f})">Retained at 80% overall coverage (%)</text>'
    )

    for index, label in enumerate(["Confidence", "Predictive entropy", "Mutual information"]):
        parts.append(f'<text class="tick" x="{sx(index, 0):.2f}" y="{top + plot_h + 28}" text-anchor="middle">{label}</text>')
        for name, mean_values, sd_values, offset in (
            ("State", state_mean, state_sd, -bar_w * 0.58),
            ("Transition", transition_mean, transition_sd, bar_w * 0.58),
        ):
            mean = float(mean_values[index])
            sd = float(sd_values[index])
            center = sx(index, offset)
            y_top = sy(mean)
            bar_h = top + plot_h - y_top
            parts.append(
                f'<rect x="{center - bar_w / 2:.2f}" y="{y_top:.2f}" width="{bar_w}" height="{bar_h:.2f}" '
                f'fill="{colors[name]}"/>'
            )
            err_top = sy(min(105.0, mean + sd))
            err_bottom = sy(max(0.0, mean - sd))
            parts.append(f'<line x1="{center:.2f}" y1="{err_top:.2f}" x2="{center:.2f}" y2="{err_bottom:.2f}" stroke="#222" stroke-width="1.4"/>')
            parts.append(f'<line x1="{center - 9:.2f}" y1="{err_top:.2f}" x2="{center + 9:.2f}" y2="{err_top:.2f}" stroke="#222" stroke-width="1.4"/>')
            parts.append(f'<line x1="{center - 9:.2f}" y1="{err_bottom:.2f}" x2="{center + 9:.2f}" y2="{err_bottom:.2f}" stroke="#222" stroke-width="1.4"/>')

    legend_x = left + plot_w - 190
    legend_y = top + 18
    for idx, name in enumerate(("State", "Transition")):
        y_pos = legend_y + idx * 26
        parts.append(f'<rect x="{legend_x}" y="{y_pos - 12}" width="18" height="18" fill="{colors[name]}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 28}" y="{y_pos + 2}">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(name for name, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[key]) for _, key in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_markdown_report(
    path: Path,
    calibration_summary: list[dict[str, Any]],
    brier_summary: list[dict[str, Any]],
    score_summary: list[dict[str, Any]],
    retention_summary: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    text = [
        "# Study 3 Presentation Summary",
        "",
        "Source: saved S3 artifacts under `work_dirs/thesis/s3/B_mc_dropout`.",
        "",
        "Fusion policy: equal-weight joint/limb logit fusion, matching the S2 fusion convention. "
        "For MC dropout, joint and limb logits are fused pass-by-pass before softmax.",
        "",
        "All summary cells are mean +/- sample SD across folds a, b, and c. "
        "Each held-out fold contributes one value; test windows are not pooled across folds for summaries.",
        "",
        "## Probability Calibration on Held-Out Untrimmed Recordings",
        "",
        markdown_table(
            calibration_summary,
            [
                ("Prediction system", "prediction_system"),
                ("Center accuracy (%) higher", "center_accuracy_percent"),
                ("NLL lower", "nll"),
                ("ECE (%) lower", "ece_percent"),
            ],
        ),
        "",
        "Appendix Brier score:",
        "",
        markdown_table(
            brier_summary,
            [
                ("Prediction system", "prediction_system"),
                ("Brier score lower", "brier"),
            ],
        ),
        "",
        "## Study 3B: Uncertainty Supports Reliability Weighting",
        "",
        "Main chart: `accuracy_coverage_curve.svg`.",
        "",
        markdown_table(
            score_summary,
            [
                ("Candidate score", "score"),
                ("Error AUROC higher", "error_auroc"),
                ("Accuracy at 80% coverage higher", "accuracy_at_80_percent"),
            ],
        ),
        "",
        "## 80% Coverage Retention",
        "",
        "Grouped bar chart: `retention_80_grouped_bars.svg`.",
        "",
        markdown_table(
            retention_summary,
            [
                ("Score", "score"),
                ("State retention (%)", "state_retention_percent"),
                ("Transition retention (%)", "transition_retention_percent"),
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
    path.write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize S3 outputs for stage presentation.")
    parser.add_argument("--root", type=Path, default=Path("work_dirs/thesis/s3/B_mc_dropout"))
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/thesis/s3/stage_presentation_20260804"))
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds = [load_fold_fusion(args.root, fold, args.ece_bins) for fold in FOLDS]
    metric_by_fold, calibration_summary = metric_rows(folds, args.ece_bins)
    coverage_by_fold, score_by_fold, retention_by_fold = coverage_and_score_rows(folds)
    curve_summary = summarize_curve(coverage_by_fold)
    score_summary = summarize_rows(
        score_by_fold,
        "score",
        [("error_auroc", 4), ("accuracy_at_80_percent", 2)],
    )
    retention_summary = summarize_rows(
        retention_by_fold,
        "score",
        [("state_retention_percent", 2), ("transition_retention_percent", 2)],
    )
    brier_summary = [
        {
            "prediction_system": row["prediction_system"],
            "brier": format_pm(float(row["brier_mean"]), float(row["brier_sd"]), 4),
            "brier_mean": row["brier_mean"],
            "brier_sd": row["brier_sd"],
        }
        for row in calibration_summary
    ]

    write_csv(args.out_dir / "probability_calibration_fusion_by_fold.csv", metric_by_fold)
    write_csv(args.out_dir / "probability_calibration_fusion_table.csv", calibration_summary)
    write_csv(args.out_dir / "appendix_brier_fusion_table.csv", brier_summary)
    write_csv(args.out_dir / "accuracy_coverage_fusion_by_fold.csv", coverage_by_fold)
    write_csv(args.out_dir / "accuracy_coverage_fusion_summary.csv", curve_summary)
    write_csv(args.out_dir / "uncertainty_candidate_summary_by_fold.csv", score_by_fold)
    write_csv(args.out_dir / "uncertainty_candidate_summary_table.csv", score_summary)
    write_csv(args.out_dir / "retention_80_by_fold.csv", retention_by_fold)
    write_csv(args.out_dir / "retention_80_table.csv", retention_summary)

    make_accuracy_coverage_plot(args.out_dir / "accuracy_coverage_curve.svg", curve_summary)
    make_retention_plot(args.out_dir / "retention_80_grouped_bars.svg", retention_summary)

    diagnostics = {
        "folds": list(FOLDS),
        "streams": list(STREAMS),
        "coverages": list(COVERAGES),
        "ece_bins": args.ece_bins,
        "fusion_policy": "equal-weight joint/limb logit average",
        "mc_fusion_policy": "equal-weight passwise joint/limb MC logit average",
        "calibration_policy": "one fused scalar temperature per fold fitted on calibration split",
        "group_policy": {
            "state": "labels whose project label_to_group is stationary or walking",
            "transition": "labels whose project label_to_group is transition; this includes Falling",
        },
        "fold_sample_counts": {
            fold.fold: {
                "calibration": int(fold.calibration.labels.shape[0]),
                "test": int(fold.test.labels.shape[0]),
                "mc_passes": int(fold.test.mc_logits.shape[1]) if fold.test.mc_logits is not None else None,
                "deterministic_temperature": float(fold.det_temperature["temperature"]),
                "mc_temperature": float(fold.mc_temperature["temperature"]),
            }
            for fold in folds
        },
        "other_group_counts_at_test": {
            fold.fold: int(
                sum(
                    1
                    for label in fold.test.label_names
                    if group_for_label_name(label) not in {"state", "transition"}
                )
            )
            for fold in folds
        },
    }
    write_json(args.out_dir / "diagnostics.json", diagnostics)
    write_markdown_report(
        args.out_dir / "presentation_summary.md",
        calibration_summary,
        brier_summary,
        score_summary,
        retention_summary,
        diagnostics,
    )
    print(f"[DONE] wrote S3 presentation summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
