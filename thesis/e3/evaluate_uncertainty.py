"""Evaluate E3 calibrated confidence and uncertainty on test logits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        DEFAULT_ANALYSIS_DIR,
        DEFAULT_LOGIT_DIR,
        DEFAULT_TEMPERATURE_DIR,
        FPS,
        LABELS,
        default_temperature_path,
        label_to_group,
        protocol_metadata,
        read_score_csv,
        safe_int,
        softmax,
        state_or_transition,
        write_json,
    )
except ImportError:
    from common import (
        DEFAULT_ANALYSIS_DIR,
        DEFAULT_LOGIT_DIR,
        DEFAULT_TEMPERATURE_DIR,
        FPS,
        LABELS,
        default_temperature_path,
        label_to_group,
        protocol_metadata,
        read_score_csv,
        safe_int,
        softmax,
        state_or_transition,
        write_json,
    )

from infer_hpe_jsonl_timeline import load_jsonl_records

DEFAULT_COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def entropy(probabilities: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probabilities * np.log(probabilities + eps)))


def top_margin(probabilities: np.ndarray) -> float:
    if probabilities.size < 2:
        return 0.0
    top2 = np.partition(probabilities, -2)[-2:]
    return float(top2[-1] - top2[-2])


def prediction_features(logits: np.ndarray, temperature: float) -> dict[str, Any]:
    raw_probs = softmax(logits)
    cal_probs = softmax(logits / temperature)
    pred_id = int(np.argmax(logits))
    return {
        "pred_id": pred_id,
        "pred_label": LABELS[pred_id],
        "pred_group": label_to_group(LABELS[pred_id]),
        "raw_confidence": float(np.max(raw_probs)),
        "raw_entropy": entropy(raw_probs),
        "raw_margin": top_margin(raw_probs),
        "calibrated_confidence": float(np.max(cal_probs)),
        "calibrated_entropy": entropy(cal_probs),
        "calibrated_margin": top_margin(cal_probs),
    }


def load_temperatures(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    temperatures = data.get("temperatures", {})
    if not isinstance(temperatures, dict) or not temperatures:
        raise ValueError(f"{path} has no temperatures")

    output = {}
    for fold, item in temperatures.items():
        value = item.get("temperature") if isinstance(item, dict) else None
        if value is None:
            raise ValueError(f"{path} has no temperature for fold {fold}")
        temperature = float(value)
        if temperature <= 0:
            raise ValueError(f"Temperature for fold {fold} must be positive")
        output[str(fold)] = temperature
    return output


def jsonl_boundaries(path: Path) -> list[int]:
    metadata, _frames = load_jsonl_records(path)
    segments = metadata.get("annotation_info", {}).get("segments", []) if metadata else []
    boundaries: set[int] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = safe_int(segment.get("start_frame"), -1)
        end = safe_int(segment.get("end_frame"), -1)
        if start >= 0:
            boundaries.add(start)
        if end >= 0:
            boundaries.add(end)
    return sorted(boundaries)


def nearest_boundary_distance(center_frame: int, boundaries: list[int]) -> int | None:
    if not boundaries:
        return None
    return min(abs(center_frame - boundary) for boundary in boundaries)


def build_prediction_rows(score_rows, temperatures: dict[str, float]) -> list[dict[str, Any]]:
    boundary_cache: dict[Path, list[int]] = {}
    output: list[dict[str, Any]] = []

    for row in score_rows:
        if row.score_type != "logit":
            raise ValueError(f"E3 uncertainty evaluation requires logits, got {row.score_type!r}")
        if row.fold not in temperatures:
            raise ValueError(f"No temperature found for fold {row.fold}")

        temperature = temperatures[row.fold]
        features = prediction_features(row.scores.astype(np.float64), temperature)
        if row.jsonl_path not in boundary_cache:
            boundary_cache[row.jsonl_path] = jsonl_boundaries(row.jsonl_path)
        boundaries = boundary_cache[row.jsonl_path]
        distance = nearest_boundary_distance(row.center_frame, boundaries)
        correct = row.valid_gt and row.gt_label == features["pred_label"]

        output.append(
            {
                "fold": row.fold,
                "subject": row.test_subject,
                "session": row.session,
                "jsonl_path": str(row.jsonl_path),
                "window_start": row.window_start,
                "window_end": row.window_end,
                "center_frame": row.center_frame,
                "center_time_sec": row.center_time_sec,
                "temperature": temperature,
                "valid_gt": int(row.valid_gt),
                "raw_gt_label": row.raw_gt_label,
                "gt_label": row.gt_label,
                "gt_group": row.gt_group,
                "gt_kind": state_or_transition(row.gt_group),
                "pred_label": features["pred_label"],
                "pred_group": features["pred_group"],
                "pred_kind": state_or_transition(features["pred_group"]),
                "pred_id": features["pred_id"],
                "correct": int(correct) if row.valid_gt else "",
                "outcome": "correct" if correct else ("incorrect" if row.valid_gt else "ignored"),
                "distance_to_boundary_frames": "" if distance is None else distance,
                "distance_to_boundary_sec": "" if distance is None else distance / FPS,
                "valid_detection_frames": row.valid_detection_frames,
                "selected_detection_center": int(row.selected_detection_center),
                "raw_confidence": features["raw_confidence"],
                "raw_entropy": features["raw_entropy"],
                "raw_margin": features["raw_margin"],
                "calibrated_confidence": features["calibrated_confidence"],
                "calibrated_entropy": features["calibrated_entropy"],
                "calibrated_margin": features["calibrated_margin"],
            }
        )

    return output


def valid_records(rows: list[dict[str, Any]], confidence_key: str) -> tuple[np.ndarray, np.ndarray]:
    valid = [row for row in rows if row["valid_gt"]]
    confidence = np.asarray([float(row[confidence_key]) for row in valid], dtype=np.float64)
    correct = np.asarray([bool(row["correct"]) for row in valid], dtype=bool)
    return confidence, correct


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
        ece += float(np.mean(in_bin)) * abs(float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin])))
    return ece


def confidence_accuracy_curve(
    rows: list[dict[str, Any]],
    scope: str,
    fold: str,
    score_name: str,
    confidence_key: str,
    num_bins: int,
) -> list[dict[str, Any]]:
    confidence, correct = valid_records(rows, confidence_key)
    curve: list[dict[str, Any]] = []
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)

        count = int(np.count_nonzero(in_bin))
        curve.append(
            {
                "scope": scope,
                "fold": fold,
                "score": score_name,
                "bin_index": bin_index,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "count": count,
                "accuracy": float(np.mean(correct[in_bin])) if count else "",
                "mean_confidence": float(np.mean(confidence[in_bin])) if count else "",
            }
        )
    return curve


def coverage_accuracy_curve(
    rows: list[dict[str, Any]],
    scope: str,
    fold: str,
    score_name: str,
    confidence_key: str,
    coverage_levels: list[float],
) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["valid_gt"]]
    valid.sort(key=lambda row: float(row[confidence_key]), reverse=True)
    total = len(valid)
    curve: list[dict[str, Any]] = []
    for coverage in coverage_levels:
        keep = int(np.ceil(total * coverage)) if total else 0
        keep = min(total, max(0, keep))
        kept_rows = valid[:keep]
        correct = [bool(row["correct"]) for row in kept_rows]
        confidence = [float(row[confidence_key]) for row in kept_rows]
        curve.append(
            {
                "scope": scope,
                "fold": fold,
                "score": score_name,
                "coverage": coverage,
                "kept": keep,
                "total": total,
                "accuracy": float(np.mean(correct)) if keep else "",
                "mean_confidence": float(np.mean(confidence)) if keep else "",
                "min_confidence": float(confidence[-1]) if keep else "",
            }
        )
    return curve


def scoped_prediction_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["fold"])].append(row)

    scopes = [("overall", "", rows)]
    for fold in sorted(grouped):
        scopes.append(("fold", fold, grouped[fold]))
    return scopes


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def build_reports(
    test_logits: Path,
    temperatures_path: Path,
    output_dir: Path,
    name: str,
    num_bins: int,
    overwrite: bool,
) -> dict[str, Any]:
    score_rows = read_score_csv(test_logits)
    temperatures = load_temperatures(temperatures_path)
    prediction_rows = build_prediction_rows(score_rows, temperatures)
    scopes = scoped_prediction_rows(prediction_rows)

    ece_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for scope, fold, rows in scopes:
        for score_name, confidence_key in (
            ("before", "raw_confidence"),
            ("after", "calibrated_confidence"),
        ):
            confidence, correct = valid_records(rows, confidence_key)
            ece = ece_score(confidence, correct, num_bins)
            ece_rows.append(
                {
                    "scope": scope,
                    "fold": fold,
                    "score": score_name,
                    "num_samples": int(confidence.size),
                    "accuracy": float(np.mean(correct)) if correct.size else 0.0,
                    "mean_confidence": float(np.mean(confidence)) if confidence.size else 0.0,
                    "ece": ece,
                    "ece_percent": ece * 100.0,
                }
            )
            confidence_rows.extend(
                confidence_accuracy_curve(
                    rows=rows,
                    scope=scope,
                    fold=fold,
                    score_name=score_name,
                    confidence_key=confidence_key,
                    num_bins=num_bins,
                )
            )
            coverage_rows.extend(
                coverage_accuracy_curve(
                    rows=rows,
                    scope=scope,
                    fold=fold,
                    score_name=score_name,
                    confidence_key=confidence_key,
                    coverage_levels=DEFAULT_COVERAGE_LEVELS,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{name}_predictions.csv", prediction_rows, overwrite)
    write_csv(output_dir / f"{name}_ece.csv", ece_rows, overwrite)
    write_csv(output_dir / f"{name}_confidence_accuracy_curve.csv", confidence_rows, overwrite)
    write_csv(output_dir / f"{name}_coverage_accuracy_curve.csv", coverage_rows, overwrite)

    result = {
        "experiment": "E3",
        "stage": "uncertainty_evaluation",
        "test_logits": str(test_logits),
        "temperatures": str(temperatures_path),
        "protocol": protocol_metadata(),
        "labels": LABELS,
        "num_bins": num_bins,
        "ece": ece_rows,
        "coverage_levels": DEFAULT_COVERAGE_LEVELS,
        "outputs": {
            "predictions": str(output_dir / f"{name}_predictions.csv"),
            "ece": str(output_dir / f"{name}_ece.csv"),
            "confidence_accuracy_curve": str(output_dir / f"{name}_confidence_accuracy_curve.csv"),
            "coverage_accuracy_curve": str(output_dir / f"{name}_coverage_accuracy_curve.csv"),
        },
    }
    write_json(output_dir / f"{name}_summary.json", result, overwrite)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E3 calibration and uncertainty on test logits.")
    parser.add_argument("--stream", choices=["joint", "limb"], default="joint")
    parser.add_argument("--test-logits", type=Path)
    parser.add_argument("--temperatures", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--name", help="Output filename prefix. Default: e3_<stream>.")
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive")

    test_logits = args.test_logits or DEFAULT_LOGIT_DIR / f"e3_{args.stream}_test_logits.csv"
    temperatures = args.temperatures or default_temperature_path(DEFAULT_TEMPERATURE_DIR, args.stream)
    name = args.name or f"e3_{args.stream}"

    result = build_reports(
        test_logits=test_logits,
        temperatures_path=temperatures,
        output_dir=args.output_dir,
        name=name,
        num_bins=args.num_bins,
        overwrite=args.overwrite,
    )
    print(f"[DONE] wrote E3 analysis reports to {args.output_dir}")
    for row in result["ece"]:
        if row["scope"] == "overall":
            print(
                f"{row['score']}: accuracy={row['accuracy']:.6f}, "
                f"mean_confidence={row['mean_confidence']:.6f}, "
                f"ECE={row['ece']:.6f}"
            )


if __name__ == "__main__":
    main()
