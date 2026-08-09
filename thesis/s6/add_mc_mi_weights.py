"""Normalize fused MC mutual information by calibration MI and export weights."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.s6.common import LABELS, selected_specs
from thesis.s6.fuse_pseudo_two_streams import (
    METADATA_KEYS,
    label_column,
    load_npz,
    predictive_quantities,
    softmax,
    validate_alignment,
    validate_probabilities,
)


DEFAULT_PSEUDO_ROOT = Path("work_dirs/thesis/s6/pseudo_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add calibration-normalized MC-MI weights to fused S6 pseudo labels."
    )
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--teachers", nargs="+", default=["t1", "t2", "t3", "t4"])
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--joint-weight", type=float, default=0.5)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_temperature(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["temperature"])


def gamma_tag(gamma: float) -> str:
    text = f"{gamma:g}".replace("-", "m").replace(".", "p")
    return f"gamma{text}"


def probability_rows(
    data: dict[str, np.ndarray],
    gamma: float,
    w_min: float,
    q95: float,
    normalized_mi: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    probabilities = np.asarray(data["probabilities"], dtype=np.float64)
    validate_probabilities(probabilities, "weighted_pseudo_probabilities")
    predictions = np.asarray(data["pseudo_label_id"], dtype=np.int64)
    rows = []
    for index, probs in enumerate(probabilities):
        pred_id = int(predictions[index])
        row = {
            "sample_index": index,
            "stream": "joint_limb_fusion_1to1",
            "pseudo_label_id": pred_id,
            "pseudo_label_name": LABELS[pred_id],
            "confidence": f"{float(np.max(probs)):.8f}",
            "mutual_information": f"{float(data['mutual_information'][index]):.8f}",
            "calibration_mi_q95": f"{float(q95):.8f}",
            "normalized_mutual_information": f"{float(normalized_mi[index]):.8f}",
            "mi_weight": f"{float(weights[index]):.8f}",
            "w_min": f"{float(w_min):.8f}",
            "gamma": f"{float(gamma):.8f}",
        }
        for key in METADATA_KEYS:
            value = data[key][index]
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = value
        for class_id, label in enumerate(LABELS):
            row[label_column(label)] = f"{float(probs[class_id]):.8f}"
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_npz(path: Path, overwrite: bool, **arrays: Any) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def compute_calibration_fused_mi(
    pseudo_root: Path,
    fold_dir: str,
    teacher: str,
    joint_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    limb_weight = 1.0 - joint_weight
    base = pseudo_root / fold_dir / teacher
    joint_logits_path = base / "joint" / "calibration_temperature_logits.npz"
    limb_logits_path = base / "limb" / "calibration_temperature_logits.npz"
    joint_temperature_path = base / "joint" / "temperature.json"
    limb_temperature_path = base / "limb" / "temperature.json"

    joint = load_npz(joint_logits_path)
    limb = load_npz(limb_logits_path)
    validate_alignment(joint, limb)
    joint_temperature = read_temperature(joint_temperature_path)
    limb_temperature = read_temperature(limb_temperature_path)

    joint_probs = softmax(joint["logits"] / joint_temperature)
    limb_probs = softmax(limb["logits"] / limb_temperature)
    fused_passes = joint_weight * joint_probs + limb_weight * limb_probs
    quantities = predictive_quantities(fused_passes)
    metadata = {
        "joint_calibration_logits": str(joint_logits_path),
        "limb_calibration_logits": str(limb_logits_path),
        "joint_temperature": joint_temperature,
        "limb_temperature": limb_temperature,
        "calibration_sample_count": int(quantities["mutual_information"].shape[0]),
    }
    return quantities["mutual_information"], metadata


def add_weights_one(
    pseudo_root: Path,
    fold_dir: str,
    teacher: str,
    gammas: list[float],
    w_min: float,
    quantile: float,
    joint_weight: float,
    overwrite: bool,
    skip_missing: bool,
) -> list[Path]:
    base = pseudo_root / fold_dir / teacher
    required = [
        base / "joint" / "calibration_temperature_logits.npz",
        base / "limb" / "calibration_temperature_logits.npz",
        base / "joint" / "temperature.json",
        base / "limb" / "temperature.json",
        base / "fusion_1to1" / "fusion_mc_calibrated_soft_probabilities.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = f"Missing MC-MI weighting source for {fold_dir}/{teacher}: {missing}"
        if skip_missing:
            print(f"[WARN] {message}")
            return []
        raise FileNotFoundError(message)

    print(f"[INFO] Adding MC-MI weights: {fold_dir} {teacher}")
    calibration_mi, calibration_meta = compute_calibration_fused_mi(
        pseudo_root,
        fold_dir,
        teacher,
        joint_weight,
    )
    pseudo_path = base / "fusion_1to1" / "fusion_mc_calibrated_soft_probabilities.npz"
    pseudo = load_npz(pseudo_path)
    pseudo_mi = np.asarray(pseudo["mutual_information"], dtype=np.float64)
    q = float(np.quantile(calibration_mi, quantile))
    warning = ""
    if q <= 1e-12:
        normalized = np.zeros_like(pseudo_mi)
        warning = "Calibration MI quantile is zero; normalized MI set to zero and all weights become 1."
    else:
        normalized = np.minimum(pseudo_mi / q, 1.0)
    written = []
    for gamma in gammas:
        weights = w_min + (1.0 - w_min) * np.power(1.0 - normalized, gamma)
        tag = gamma_tag(gamma)
        output_prefix = f"fusion_mc_calibrated_soft_probabilities_mi_weighted_{tag}"
        output_dir = base / "fusion_1to1"
        arrays = {
            **pseudo,
            "calibration_mutual_information": calibration_mi.astype(np.float32, copy=False),
            "calibration_mi_quantile": np.asarray([q], dtype=np.float32),
            "calibration_mi_quantile_level": np.asarray([quantile], dtype=np.float32),
            "normalized_mutual_information": normalized.astype(np.float32, copy=False),
            "mi_weight": weights.astype(np.float32, copy=False),
            "w_min": np.asarray([w_min], dtype=np.float32),
            "gamma": np.asarray([gamma], dtype=np.float32),
        }
        npz_path = output_dir / f"{output_prefix}.npz"
        save_npz(npz_path, overwrite, **arrays)
        csv_path = output_dir / f"{output_prefix}.csv"
        write_csv(
            csv_path,
            probability_rows(pseudo, gamma, w_min, q, normalized, weights),
            overwrite,
        )
        summary = {
            "fold": fold_dir,
            "teacher": teacher,
            "source": str(pseudo_path),
            "output_npz": str(npz_path),
            "output_csv": str(csv_path),
            "normalization": "min(pseudo_mi / calibration_mi_q, 1)",
            "weight_formula": "w_min + (1 - w_min) * (1 - normalized_mi) ** gamma",
            "calibration_distribution": "fused joint/limb calibration MC mutual information only",
            "calibration_mi_quantile_level": quantile,
            "calibration_mi_quantile": q,
            "w_min": w_min,
            "gamma": gamma,
            "pseudo_sample_count": int(pseudo_mi.shape[0]),
            "calibration_mi": {
                "mean": float(np.mean(calibration_mi)),
                "median": float(np.median(calibration_mi)),
                "q25": float(np.quantile(calibration_mi, 0.25)),
                "q75": float(np.quantile(calibration_mi, 0.75)),
                "q95": float(np.quantile(calibration_mi, 0.95)),
                "max": float(np.max(calibration_mi)),
            },
            "pseudo_mi": {
                "mean": float(np.mean(pseudo_mi)),
                "median": float(np.median(pseudo_mi)),
                "q25": float(np.quantile(pseudo_mi, 0.25)),
                "q75": float(np.quantile(pseudo_mi, 0.75)),
                "q95": float(np.quantile(pseudo_mi, 0.95)),
                "max": float(np.max(pseudo_mi)),
            },
            "weights": {
                "mean": float(np.mean(weights)),
                "median": float(np.median(weights)),
                "q25": float(np.quantile(weights, 0.25)),
                "q75": float(np.quantile(weights, 0.75)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            },
            "calibration_sources": calibration_meta,
            "warning": warning,
        }
        write_json(output_dir / f"{output_prefix}_summary.json", summary, overwrite)
        written.append(csv_path)
    return written


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.w_min <= 1.0:
        raise ValueError("--w-min must be in [0, 1]")
    if not 0.0 < args.quantile <= 1.0:
        raise ValueError("--quantile must be in (0, 1]")
    if not 0.0 <= args.joint_weight <= 1.0:
        raise ValueError("--joint-weight must be in [0, 1]")
    written = []
    for spec in selected_specs(args.folds, args.teachers):
        written.extend(
            add_weights_one(
                args.pseudo_root,
                spec.fold_dir,
                spec.teacher,
                args.gammas,
                args.w_min,
                args.quantile,
                args.joint_weight,
                args.overwrite,
                args.skip_missing,
            )
        )
    print(f"[INFO] Wrote {len(written)} weighted MC pseudo-label CSV files")


if __name__ == "__main__":
    main()
