"""Fuse S6 joint/limb pseudo-label outputs with 1:1 stream weights."""

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


DEFAULT_PSEUDO_ROOT = Path("work_dirs/thesis/s6/pseudo_labels")
METADATA_KEYS = (
    "subject_id",
    "recording_id",
    "sequence_id",
    "frame_dir",
    "window_start_frame",
    "window_end_frame",
    "center_frame",
    "center_timestamp",
    "manual_label_id",
    "manual_label_name",
    "label_group",
)
MODE_FILES = {
    "hard": "deterministic_hard_pseudo_labels.npz",
    "raw_soft": "raw_soft_probabilities.npz",
    "calibrated_soft": "calibrated_soft_probabilities.npz",
    "mc_calibrated_soft": "mc_calibrated_soft_probabilities.npz",
}
MODE_OUTPUTS = {
    "hard": "fusion_hard_pseudo_labels",
    "raw_soft": "fusion_raw_soft_probabilities",
    "calibrated_soft": "fusion_calibrated_soft_probabilities",
    "mc_calibrated_soft": "fusion_mc_calibrated_soft_probabilities",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse S6 joint and limb pseudo-label exports.")
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--folds", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--teachers", nargs="+", default=["t1", "t2", "t3", "t4"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hard"],
        choices=("hard", "raw_soft", "calibrated_soft", "mc_calibrated_soft", "all"),
    )
    parser.add_argument("--joint-weight", type=float, default=0.5)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expanded_modes(values: list[str]) -> list[str]:
    if "all" in values:
        return ["hard", "raw_soft", "calibrated_soft", "mc_calibrated_soft"]
    return values


def softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def validate_probabilities(values: np.ndarray, name: str) -> None:
    probs = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
        raise ValueError(f"{name} has values outside [0, 1]")
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{name} rows do not sum to one; max error={np.max(np.abs(sums - 1.0))}")


def predictive_quantities(probability_passes: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(probability_passes, dtype=np.float64)
    validate_probabilities(values, "probability_passes")
    mean_probs = values.mean(axis=1)
    validate_probabilities(mean_probs, "predictive_mean")
    eps = 1e-12
    predictive_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, eps, 1.0)), axis=1)
    pass_entropy = -np.sum(values * np.log(np.clip(values, eps, 1.0)), axis=2)
    expected_entropy = pass_entropy.mean(axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    pass_predictions = np.argmax(values, axis=2)
    modal_counts = np.zeros(values.shape[0], dtype=np.int64)
    for index, predictions in enumerate(pass_predictions):
        modal_counts[index] = int(np.max(np.bincount(predictions, minlength=values.shape[2])))
    variation_ratio = 1.0 - modal_counts.astype(np.float64) / values.shape[1]
    return {
        "probabilities": mean_probs,
        "pseudo_label_id": np.argmax(mean_probs, axis=1).astype(np.int64),
        "confidence": np.max(mean_probs, axis=1),
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
        "mean_probability_variance": np.var(values, axis=1).mean(axis=1),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def arrays_equal(left: np.ndarray, right: np.ndarray, key: str) -> bool:
    if left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        return bool(np.allclose(left, right, atol=1e-7, rtol=0))
    return bool(np.array_equal(left, right))


def validate_alignment(joint: dict[str, np.ndarray], limb: dict[str, np.ndarray]) -> None:
    for key in METADATA_KEYS:
        if key not in joint or key not in limb:
            raise KeyError(f"Missing metadata key {key!r}")
        if not arrays_equal(joint[key], limb[key], key):
            raise ValueError(f"Joint/limb metadata is not aligned for key {key!r}")


def label_column(label: str) -> str:
    slug = label.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    return f"prob_{slug}"


def base_row(data: dict[str, np.ndarray], index: int, fold: str, teacher: str) -> dict[str, Any]:
    manual_id = int(data["manual_label_id"][index])
    row = {
        "fold": fold,
        "teacher": teacher,
        "stream": "joint_limb_fusion_1to1",
        "sample_index": index,
    }
    for key in METADATA_KEYS:
        value = data[key][index]
        if isinstance(value, np.generic):
            value = value.item()
        row[key] = value
    row["manual_label_name"] = (
        str(data["manual_label_name"][index])
        if "manual_label_name" in data
        else LABELS[manual_id]
    )
    return row


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


def save_npz(path: Path, overwrite: bool, **arrays: Any) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_manifest(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metadata_arrays(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: data[key] for key in METADATA_KEYS}


def probability_rows(
    data: dict[str, np.ndarray],
    probabilities: np.ndarray,
    fold: str,
    teacher: str,
    include_probs: bool,
    extra: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    validate_probabilities(probabilities, "fused_probabilities")
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    rows = []
    for index, probs in enumerate(probabilities):
        pred_id = int(predictions[index])
        row = base_row(data, index, fold, teacher)
        row.update(
            {
                "pseudo_label_id": pred_id,
                "pseudo_label_name": LABELS[pred_id],
                "confidence": f"{float(confidences[index]):.8f}",
            }
        )
        if extra:
            for key, values in extra.items():
                row[key] = f"{float(values[index]):.8f}"
        if include_probs:
            for class_id, label in enumerate(LABELS):
                row[label_column(label)] = f"{float(probs[class_id]):.8f}"
        rows.append(row)
    return rows


def fuse_mode(
    joint: dict[str, np.ndarray],
    limb: dict[str, np.ndarray],
    mode: str,
    joint_weight: float,
) -> dict[str, np.ndarray]:
    limb_weight = 1.0 - joint_weight
    if mode in {"hard", "raw_soft"}:
        if "logits" not in joint or "logits" not in limb:
            raise KeyError(f"{mode} fusion requires logits in joint and limb NPZ files")
        logits = joint_weight * joint["logits"] + limb_weight * limb["logits"]
        probs = softmax(logits)
        return {
            "logits": logits.astype(np.float32, copy=False),
            "probabilities": probs.astype(np.float32, copy=False),
            "pseudo_label_id": np.argmax(probs, axis=1).astype(np.int64),
        }

    if mode == "calibrated_soft":
        if "probabilities" not in joint or "probabilities" not in limb:
            raise KeyError("calibrated_soft fusion requires probabilities in joint and limb NPZ files")
        probs = joint_weight * joint["probabilities"] + limb_weight * limb["probabilities"]
        validate_probabilities(probs, "calibrated_soft_fused_probabilities")
        result = {
            "probabilities": probs.astype(np.float32, copy=False),
            "pseudo_label_id": np.argmax(probs, axis=1).astype(np.int64),
        }
        if "temperature" in joint:
            result["joint_temperature"] = joint["temperature"]
        if "temperature" in limb:
            result["limb_temperature"] = limb["temperature"]
        return result

    if mode == "mc_calibrated_soft":
        if "probability_passes" not in joint or "probability_passes" not in limb:
            raise KeyError("mc_calibrated_soft fusion requires probability_passes in joint and limb NPZ files")
        passes = joint_weight * joint["probability_passes"] + limb_weight * limb["probability_passes"]
        quantities = predictive_quantities(passes)
        return {
            "probability_passes": passes.astype(np.float32, copy=False),
            "probabilities": quantities["probabilities"].astype(np.float32, copy=False),
            "pseudo_label_id": quantities["pseudo_label_id"],
            "predictive_entropy": quantities["predictive_entropy"].astype(np.float32, copy=False),
            "expected_entropy": quantities["expected_entropy"].astype(np.float32, copy=False),
            "mutual_information": quantities["mutual_information"].astype(np.float32, copy=False),
            "variation_ratio": quantities["variation_ratio"].astype(np.float32, copy=False),
            "mean_probability_variance": quantities["mean_probability_variance"].astype(np.float32, copy=False),
        }

    raise ValueError(f"Unsupported mode: {mode}")


def fuse_one(
    pseudo_root: Path,
    out_root: Path,
    fold_dir: str,
    teacher: str,
    mode: str,
    joint_weight: float,
    overwrite: bool,
    skip_missing: bool,
) -> Path | None:
    source_name = MODE_FILES[mode]
    joint_path = pseudo_root / fold_dir / teacher / "joint" / source_name
    limb_path = pseudo_root / fold_dir / teacher / "limb" / source_name
    missing = [str(path) for path in (joint_path, limb_path) if not path.exists()]
    if missing:
        message = f"Missing source for {fold_dir}/{teacher}/{mode}: {missing}"
        if skip_missing:
            print(f"[WARN] {message}")
            return None
        raise FileNotFoundError(message)

    print(f"[INFO] Fusing {fold_dir} {teacher} {mode}")
    joint = load_npz(joint_path)
    limb = load_npz(limb_path)
    validate_alignment(joint, limb)
    fused = fuse_mode(joint, limb, mode, joint_weight)
    output_dir = out_root / fold_dir / teacher / "fusion_1to1"
    output_prefix = MODE_OUTPUTS[mode]

    arrays = {
        **metadata_arrays(joint),
        **fused,
        "joint_weight": np.asarray([joint_weight], dtype=np.float32),
        "limb_weight": np.asarray([1.0 - joint_weight], dtype=np.float32),
    }
    save_npz(output_dir / f"{output_prefix}.npz", overwrite=overwrite, **arrays)

    extra = None
    if mode == "mc_calibrated_soft":
        extra = {
            "predictive_entropy": fused["predictive_entropy"],
            "expected_entropy": fused["expected_entropy"],
            "mutual_information": fused["mutual_information"],
            "variation_ratio": fused["variation_ratio"],
            "mean_probability_variance": fused["mean_probability_variance"],
        }
    include_probs = mode != "hard"
    rows = probability_rows(
        joint,
        fused["probabilities"],
        fold_dir,
        teacher,
        include_probs=include_probs,
        extra=extra,
    )
    write_csv(output_dir / f"{output_prefix}.csv", rows, overwrite=overwrite)
    write_manifest(
        output_dir / f"{output_prefix}_manifest.json",
        {
            "fold": fold_dir,
            "teacher": teacher,
            "mode": mode,
            "fusion": "joint_limb_1to1",
            "joint_weight": joint_weight,
            "limb_weight": 1.0 - joint_weight,
            "joint_source": str(joint_path),
            "limb_source": str(limb_path),
            "output_npz": str(output_dir / f"{output_prefix}.npz"),
            "output_csv": str(output_dir / f"{output_prefix}.csv"),
            "sample_count": int(fused["probabilities"].shape[0]),
            "class_order": LABELS,
        },
        overwrite=overwrite,
    )
    return output_dir / f"{output_prefix}.csv"


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.joint_weight <= 1.0):
        raise ValueError("--joint-weight must be in [0, 1]")
    modes = expanded_modes(args.modes)
    out_root = args.out_root or args.pseudo_root
    written = []
    for spec in selected_specs(args.folds, args.teachers):
        for mode in modes:
            path = fuse_one(
                args.pseudo_root,
                out_root,
                spec.fold_dir,
                spec.teacher,
                mode,
                args.joint_weight,
                args.overwrite,
                args.skip_missing,
            )
            if path is not None:
                written.append(path)
    print(f"[INFO] Wrote {len(written)} fused file groups")


if __name__ == "__main__":
    main()
