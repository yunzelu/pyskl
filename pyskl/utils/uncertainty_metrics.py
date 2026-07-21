"""Metrics for probabilistic and MC-dropout predictions."""

from __future__ import annotations

import numpy as np

from .temperature_scaling import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def validate_probabilities(
    probs: np.ndarray,
    tolerance: float = 1e-5,
    name: str = "probabilities",
) -> None:
    values = np.asarray(probs)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError(f"{name} values must be in [0, 1]")
    sums = values.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=tolerance, rtol=0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to one; max error={max_error}")


def predictive_quantities(prob_passes: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Compute MC predictive quantities from probabilities shaped ``[N, K, C]``."""
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
    mean_probability_variance = class_variance.mean(axis=1)
    top2 = np.sort(p_bar, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    confidence = np.max(p_bar, axis=1)

    if np.any(mutual_information < -1e-7):
        raise ValueError("mutual information contains negative values below tolerance")
    if np.any(mutual_information > predictive_entropy + 1e-7):
        raise ValueError("mutual information exceeds predictive entropy")

    return {
        "probabilities": p_bar,
        "prediction": np.argmax(p_bar, axis=1).astype(np.int64),
        "confidence": confidence,
        "margin": margin,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
        "class_variance": class_variance,
        "mean_probability_variance": mean_probability_variance,
    }


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(labels.astype(np.int64), predictions.astype(np.int64)):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            matrix[gt, pred] += 1
    return matrix


def per_class_precision_recall_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    num_classes = len(class_names)
    rows = []
    for class_id, label_name in enumerate(class_names):
        tp = int(np.count_nonzero((labels == class_id) & (predictions == class_id)))
        fp = int(np.count_nonzero((labels != class_id) & (predictions == class_id)))
        fn = int(np.count_nonzero((labels == class_id) & (predictions != class_id)))
        support = int(np.count_nonzero(labels == class_id))
        predicted = int(np.count_nonzero(predictions == class_id))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append(
            {
                "class_id": class_id,
                "label": label_name,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    ece_bins: int = 15,
) -> dict[str, object]:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    validate_probabilities(probabilities)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    correct = predictions == targets
    per_class = per_class_precision_recall_f1(predictions, targets, class_names)
    active_f1 = [
        float(row["f1"])
        for row in per_class
        if int(row["support"]) > 0 or int(row["predicted"]) > 0
    ]
    top2 = np.sort(probabilities, axis=1)[:, -2:]
    return {
        "accuracy": float(np.mean(correct)) if len(targets) else 0.0,
        "macro_f1": float(np.mean(active_f1)) if active_f1 else 0.0,
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets, num_bins=ece_bins),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))) if len(targets) else 0.0,
        "mean_top1_top2_margin": float(np.mean(top2[:, 1] - top2[:, 0])) if len(targets) else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(predictions, targets, len(class_names)),
    }


def binary_auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """AUROC where higher scores indicate the positive class."""
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


def binary_auprc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Average precision where higher scores indicate the positive class."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positives, dtype=bool)
    n_pos = int(np.count_nonzero(labels))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(values), dtype=np.float64) + 1.0)
    return float(np.sum(precision * sorted_labels) / n_pos)


def reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> list[dict[str, float | int]]:
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == targets
    rows = []
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": float(np.mean(correct[mask])) if count else 0.0,
                "confidence": float(np.mean(confidences[mask])) if count else 0.0,
            }
        )
    return rows
