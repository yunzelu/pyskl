"""Fit scalar temperature values on E3 calibration logits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        DEFAULT_LOGIT_DIR,
        DEFAULT_TEMPERATURE_DIR,
        LABELS,
        default_temperature_path,
        protocol_metadata,
        read_score_csv,
        softmax,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_LOGIT_DIR,
        DEFAULT_TEMPERATURE_DIR,
        LABELS,
        default_temperature_path,
        protocol_metadata,
        read_score_csv,
        softmax,
        write_json,
    )

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def ece_score(confidence: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
    if confidence.size == 0:
        return 0.0

    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
        if not np.any(in_bin):
            continue
        bin_conf = float(np.mean(confidence[in_bin]))
        bin_acc = float(np.mean(correct[in_bin]))
        ece += float(np.mean(in_bin)) * abs(bin_acc - bin_conf)
    return ece


def nll_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(scaled, axis=1)
    target_logits = scaled[np.arange(labels.shape[0]), labels]
    return float(np.mean(logsumexp - target_logits))


def probs_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float, num_bins: int) -> dict[str, float]:
    probs = softmax(logits / temperature)
    pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = pred == labels
    return {
        "accuracy": float(np.mean(correct)) if correct.size else 0.0,
        "mean_confidence": float(np.mean(confidence)) if confidence.size else 0.0,
        "nll": nll_from_logits(logits, labels, temperature),
        "ece": ece_score(confidence, correct, num_bins),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int) -> float:
    # Optimize log(T) with a derivative-free scalar search. The model is frozen;
    # only this one positive scalar is learned from calibration logits.
    lower = -5.0
    upper = 5.0
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - np.sqrt(5.0)) / 2.0

    h = upper - lower
    c = lower + inv_phi_sq * h
    d = lower + inv_phi * h
    yc = nll_from_logits(logits, labels, temperature=float(np.exp(c)))
    yd = nll_from_logits(logits, labels, temperature=float(np.exp(d)))

    for _ in range(max_iter):
        if yc < yd:
            upper = d
            d = c
            yd = yc
            h = inv_phi * h
            c = lower + inv_phi_sq * h
            yc = nll_from_logits(logits, labels, temperature=float(np.exp(c)))
        else:
            lower = c
            c = d
            yc = yd
            h = inv_phi * h
            d = lower + inv_phi * h
            yd = nll_from_logits(logits, labels, temperature=float(np.exp(d)))

    log_temperature = (lower + upper) / 2.0
    return float(np.exp(log_temperature))


def fold_arrays(rows) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        if row.score_type != "logit":
            raise ValueError(f"Temperature fitting requires logits, got score_type={row.score_type!r}")
        if row.valid_gt:
            grouped[row.fold].append(row)

    output: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for fold, fold_rows in grouped.items():
        logits = np.stack([row.scores for row in fold_rows]).astype(np.float32, copy=False)
        labels = np.asarray([LABEL_TO_ID[row.gt_label] for row in fold_rows], dtype=np.int64)
        subject = fold_rows[0].test_subject if fold_rows else ""
        output[fold] = (logits, labels, subject)
    return output


def fit_all(calib_logits: Path, output: Path, num_bins: int, max_iter: int, overwrite: bool) -> dict[str, Any]:
    rows = read_score_csv(calib_logits)
    folds = fold_arrays(rows)
    if not folds:
        raise ValueError(f"{calib_logits} has no valid calibration rows")

    temperatures: dict[str, Any] = {}
    for fold in sorted(folds):
        logits, labels, subject = folds[fold]
        before = probs_metrics(logits, labels, temperature=1.0, num_bins=num_bins)
        temperature = fit_temperature(logits, labels, max_iter=max_iter)
        after = probs_metrics(logits, labels, temperature=temperature, num_bins=num_bins)
        temperatures[fold] = {
            "fold": fold,
            "calib_subject": subject,
            "temperature": temperature,
            "num_samples": int(labels.shape[0]),
            "before": before,
            "after": after,
        }

    result = {
        "experiment": "E3",
        "stage": "temperature_fit",
        "calib_logits": str(calib_logits),
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "num_bins": num_bins,
        "temperatures": temperatures,
    }
    write_json(output, result, overwrite=overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit E3 temperature scaling on calibration logits.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--calib-logits", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")

    calib_logits = args.calib_logits or DEFAULT_LOGIT_DIR / f"e3_{args.stream}_calib_logits.csv"
    output = args.output or default_temperature_path(DEFAULT_TEMPERATURE_DIR, args.stream)
    result = fit_all(
        calib_logits=calib_logits,
        output=output,
        num_bins=args.num_bins,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote temperatures to {output}")
    for fold, item in result["temperatures"].items():
        print(
            f"fold {fold}: T={item['temperature']:.6f}, "
            f"ECE {item['before']['ece']:.6f} -> {item['after']['ece']:.6f}, "
            f"NLL {item['before']['nll']:.6f} -> {item['after']['nll']:.6f}"
        )


if __name__ == "__main__":
    main()
