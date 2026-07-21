"""Scalar temperature scaling for deterministic and MC predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax over the last axis."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    probs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probs.astype(np.float64, copy=False)


def mean_mc_probabilities(mc_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert ``[N, K, C]`` MC logits to predictive mean probabilities."""
    values = np.asarray(mc_logits)
    if values.ndim != 3:
        raise ValueError(f"mc_logits must have shape [N, K, C], got {values.shape}")
    return softmax_np(values, temperature=temperature).mean(axis=1)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean multiclass negative log likelihood from probabilities."""
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError(f"probs must be [N, C], got {probabilities.shape}")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probs and labels have different lengths")
    picked = probabilities[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean multiclass Brier score."""
    probabilities = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Top-label expected calibration error."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
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
        if count == 0:
            continue
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
    """Fit one scalar temperature by minimizing MC predictive NLL.

    Optimization is a deterministic golden-section search over ``log(T)``.
    This avoids adding a SciPy dependency and keeps the positivity constraint
    exact.
    """
    logits = np.asarray(mc_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError(f"mc_logits must have shape [N, K, C], got {logits.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("mc_logits and labels have different lengths")
    if min_temperature <= 0 or max_temperature <= min_temperature:
        raise ValueError("temperature bounds must satisfy 0 < min < max")

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
        warning = (
            "Optimized temperature produced worse calibration NLL than T=1; "
            "falling back to T=1."
        )
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
