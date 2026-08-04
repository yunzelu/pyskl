"""Study 3: MC-dropout and temperature calibration for PoseC3D B checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyskl.utils.mc_dropout import enable_mc_dropout
from pyskl.utils.temperature_scaling import fit_temperature, softmax_np
from pyskl.utils.uncertainty_metrics import (
    binary_auprc,
    binary_auroc,
    classification_metrics,
    predictive_quantities,
    reliability_bins,
    validate_probabilities,
)
from thesis.s2.common import CENTER_OFFSET, FPS, LABELS, LABEL_TO_ID, STRIDE, WINDOW_SIZE
from thesis.s2.evaluate_predictions import EvalRow, segmental_metrics


COVERAGE_LEVELS = (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10)
UNCERTAINTY_SCORE_COLUMNS = (
    "one_minus_confidence",
    "predictive_entropy",
    "mutual_information",
    "variation_ratio",
    "mean_probability_variance",
)


@dataclass(frozen=True)
class SampleMeta:
    index: int
    subject_id: str
    recording_id: str
    sequence_id: str
    frame_dir: str
    window_start_frame: int
    window_end_frame: int
    center_frame: int
    center_timestamp: float
    label_id: int
    label_name: str
    label_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation-only Study 3 MC-dropout/temperature calibration for PoseC3D."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-ann", type=Path, required=True)
    parser.add_argument("--test-ann", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-passes", type=int, default=10)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-mc-logits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--calibration-split", help="Default: fold_<id>_calib inferred from config.")
    parser.add_argument("--test-split", help="Default: test_split from config.")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite to replace S3 outputs")
    path.mkdir(parents=True, exist_ok=True)
    (path / "sequences").mkdir(exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def git_commit_hash() -> str:
    return run_command(["git", "rev-parse", "HEAD"])


def package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def write_environment(path: Path) -> dict[str, Any]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "git_commit": git_commit_hash(),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "mmcv": package_version("mmcv"),
            "pyskl": "local",
            "matplotlib": package_version("matplotlib"),
        },
    }
    lines = [
        f"python: {env['python']}",
        f"platform: {env['platform']}",
        f"executable: {env['executable']}",
        f"cwd: {env['cwd']}",
        f"git_commit: {env['git_commit']}",
    ]
    for name, version in env["packages"].items():
        lines.append(f"{name}: {version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logit_test_cfg(config: Any) -> None:
    if config.model.get("test_cfg") is None:
        config.model["test_cfg"] = {}
    config.model.test_cfg["average_clips"] = "score"


def update_test_pipeline_clip_len(config: Any) -> None:
    for stage in config.data.test.pipeline:
        if stage.get("type") in {"UniformSample", "UniformSampleFrames"}:
            stage["clip_len"] = WINDOW_SIZE


def load_config(path: Path) -> Any:
    import mmcv

    cfg = mmcv.Config.fromfile(str(path))
    set_logit_test_cfg(cfg)
    update_test_pipeline_clip_len(cfg)
    return cfg


def split_fold_id(split_name: str) -> str | None:
    parts = str(split_name).split("_")
    if len(parts) >= 3 and parts[0] == "fold":
        return parts[1]
    return None


def infer_splits(config: Any, calibration_split: str | None, test_split: str | None) -> tuple[str, str]:
    test = test_split or str(config.get("test_split") or config.data.test.get("split") or "")
    if not test:
        raise ValueError("Could not infer test split; pass --test-split")
    fold = split_fold_id(test)
    if calibration_split is None:
        if fold is None:
            raise ValueError("Could not infer calibration split; pass --calibration-split")
        calibration_split = f"fold_{fold}_calib"
    return calibration_split, test


def build_dataset_and_loader(
    config: Any,
    ann_file: Path,
    split: str,
    batch_size: int | None,
    num_workers: int | None,
) -> tuple[Any, Any]:
    from pyskl.datasets import build_dataloader, build_dataset

    dataset_cfg = copy.deepcopy(config.data.test)
    dataset_cfg.ann_file = str(ann_file)
    dataset_cfg.split = split
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg, dict(test_mode=True))

    effective_batch_size = (
        batch_size
        if batch_size is not None
        else int(config.data.get("videos_per_gpu", 1))
    )
    effective_workers = (
        num_workers
        if num_workers is not None
        else int(config.data.get("workers_per_gpu", 1))
    )
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=effective_batch_size,
        workers_per_gpu=effective_workers,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )
    return dataset, data_loader


def sample_metadata(dataset: Any) -> list[SampleMeta]:
    samples: list[SampleMeta] = []
    for index, item in enumerate(dataset.video_infos):
        label_id = int(item.get("hard_label", item.get("label", -1)))
        label_name = str(item.get("label_name") or (LABELS[label_id] if 0 <= label_id < len(LABELS) else ""))
        subject = str(item.get("subject_id") or item.get("subject") or "").lower()
        recording = str(item.get("recording_id") or item.get("session_name") or "")
        start_frame = int(item.get("start_frame", 0))
        end_frame = int(item.get("end_frame", start_frame + WINDOW_SIZE - 1))
        center_frame = int(item.get("center_frame", start_frame + CENTER_OFFSET))
        center_timestamp = float(item.get("center_timestamp", center_frame / FPS))
        samples.append(
            SampleMeta(
                index=index,
                subject_id=subject,
                recording_id=recording,
                sequence_id=recording,
                frame_dir=str(item.get("frame_dir") or ""),
                window_start_frame=start_frame,
                window_end_frame=end_frame,
                center_frame=center_frame,
                center_timestamp=center_timestamp,
                label_id=label_id,
                label_name=label_name,
                label_group=str(item.get("gt_group") or item.get("label_group") or ""),
            )
        )
    return samples


def validate_split_safety(
    calibration_samples: list[SampleMeta],
    test_samples: list[SampleMeta],
    config: Any,
) -> dict[str, Any]:
    calibration_subjects = sorted({item.subject_id for item in calibration_samples})
    test_subjects = sorted({item.subject_id for item in test_samples})
    calibration_recordings = {item.recording_id for item in calibration_samples}
    test_recordings = {item.recording_id for item in test_samples}
    duplicate_recordings = sorted(calibration_recordings & test_recordings)

    if set(calibration_subjects) & set(test_subjects):
        raise ValueError(
            f"Calibration and test subjects overlap: {sorted(set(calibration_subjects) & set(test_subjects))}"
        )
    if duplicate_recordings:
        raise ValueError(f"Duplicate recording IDs across calibration/test: {duplicate_recordings}")

    val_split = str(config.get("val_split") or "")
    safety = {
        "calibration_subjects": calibration_subjects,
        "test_subjects": test_subjects,
        "duplicate_recordings": duplicate_recordings,
        "validation_split_from_config": val_split,
        "validation_subject_loaded": False,
    }
    fold = split_fold_id(val_split)
    if fold is not None:
        try:
            from thesis.s2.common import discover_s2_folds

            fold_spec = next(item for item in discover_s2_folds() if item.fold == fold)
            validation_subject = fold_spec.val_subject
            safety["validation_subject"] = validation_subject
            loaded_subjects = set(calibration_subjects) | set(test_subjects)
            safety["validation_subject_loaded"] = validation_subject in loaded_subjects
            if validation_subject in loaded_subjects:
                raise ValueError(
                    f"Validation subject {validation_subject!r} was loaded by S3; "
                    "temperature calibration must use the calibration subject only."
                )
        except StopIteration:
            pass
    return safety


def assert_valid_labels(samples: list[SampleMeta], split_name: str) -> np.ndarray:
    labels = np.asarray([item.label_id for item in samples], dtype=np.int64)
    invalid = [item for item in samples if item.label_id not in range(len(LABELS))]
    if invalid:
        preview = [
            {
                "recording_id": item.recording_id,
                "center_frame": item.center_frame,
                "label_id": item.label_id,
                "label_name": item.label_name,
            }
            for item in invalid[:5]
        ]
        raise ValueError(f"{split_name} contains invalid labels. First examples: {preview}")
    return labels


def assert_monotonic_by_sequence(samples: list[SampleMeta], split_name: str) -> None:
    by_sequence: dict[str, list[SampleMeta]] = {}
    for item in samples:
        by_sequence.setdefault(item.sequence_id, []).append(item)
    for sequence_id, rows in by_sequence.items():
        timestamps = [item.center_timestamp for item in rows]
        if any(second < first for first, second in zip(timestamps, timestamps[1:])):
            raise ValueError(f"{split_name} sequence {sequence_id!r} timestamps are not monotonic")


def scatter_if_needed(data_batch: dict[str, Any], model: Any) -> dict[str, Any]:
    from mmcv.parallel import scatter

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        return scatter(data_batch, [device])[0]
    return data_batch


def batch_logits(model: Any, data_batch: dict[str, Any]) -> np.ndarray:
    import torch

    data_batch = scatter_if_needed(data_batch, model)
    with torch.no_grad():
        scores = model(return_loss=False, **data_batch)
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return values


def looks_like_probabilities(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        return False
    return bool(
        np.all(np.isfinite(arr))
        and np.all(arr >= -1e-6)
        and np.all(arr <= 1.0 + 1e-6)
        and np.allclose(arr.sum(axis=1), 1.0, atol=1e-4, rtol=0)
    )


def deterministic_logits(model: Any, data_loader: Any, expected_samples: int) -> np.ndarray:
    model.eval()
    outputs = [batch_logits(model, batch) for batch in data_loader]
    logits = np.concatenate(outputs, axis=0)
    if logits.shape[0] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {logits.shape[0]}")
    if looks_like_probabilities(logits):
        raise RuntimeError(
            "Model output looks like probabilities, not raw logits. "
            "Temperature scaling requires raw logits; check test_cfg.average_clips."
        )
    return logits


def mc_logits(
    model: Any,
    data_loader: Any,
    expected_samples: int,
    num_passes: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    if num_passes < 2:
        raise ValueError("--num-passes must be at least 2 for MC-dropout stochasticity checks")
    dropout_info = enable_mc_dropout(model)
    batches = []
    first_batch_diff: float | None = None
    for batch in data_loader:
        pass_logits = []
        for _ in range(num_passes):
            pass_logits.append(batch_logits(model, batch))
        stacked = np.stack(pass_logits, axis=1)
        if first_batch_diff is None:
            first_batch_diff = float(np.mean(np.abs(stacked[:, 0, :] - stacked[:, 1, :])))
            if first_batch_diff <= 1e-8:
                raise RuntimeError(
                    "MC-dropout sanity check failed: pass 1 and pass 2 logits "
                    f"mean absolute difference is {first_batch_diff:.3g}."
                )
        batches.append(stacked)

    logits = np.concatenate(batches, axis=0)
    if logits.shape[0] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {logits.shape[0]}")
    return logits.astype(np.float32, copy=False), dropout_info, float(first_batch_diff or 0.0)


def eval_rows_for_sequence_metrics(
    samples: list[SampleMeta],
    probabilities: np.ndarray,
    method: str,
) -> list[EvalRow]:
    predictions = np.argmax(probabilities, axis=1)
    rows = []
    for item, pred_id, prob in zip(samples, predictions, probabilities):
        rows.append(
            EvalRow(
                method=method,
                stream="joint",
                eta="",
                fold="s3",
                subject_id=item.subject_id,
                recording_id=item.recording_id,
                start_frame=item.window_start_frame,
                end_frame=item.window_end_frame,
                center_frame=item.center_frame,
                center_timestamp=item.center_timestamp,
                gt_label=item.label_name,
                gt_group=item.label_group,
                pred_label=LABELS[int(pred_id)],
                pred_id=int(pred_id),
                confidence=float(np.max(prob)),
                logits=np.log(np.clip(prob, 1e-12, 1.0)).astype(np.float32),
                probabilities=prob.astype(np.float32),
            )
        )
    return rows


def metrics_with_sequence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    samples: list[SampleMeta],
    ece_bins: int,
    method: str,
) -> dict[str, Any]:
    metrics = classification_metrics(probabilities, labels, LABELS, ece_bins)
    sequence = segmental_metrics(eval_rows_for_sequence_metrics(samples, probabilities, method))
    metrics.update(sequence)
    matrix = metrics["confusion_matrix"]
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def metrics_table_rows(summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, name in (
        ("deterministic", "Deterministic"),
        ("deterministic_calibrated", "Deterministic calibrated"),
        ("mc_raw", "MC raw"),
        ("mc_calibrated", "MC calibrated"),
    ):
        item = summary[key]
        rows.append(
            {
                "method": name,
                "accuracy": f"{float(item['accuracy']):.8f}",
                "macro_f1": f"{float(item['macro_f1']):.8f}",
                "nll": f"{float(item['nll']):.8f}",
                "brier": f"{float(item['brier']):.8f}",
                "ece": f"{float(item['ece']):.8f}",
                "edit": f"{float(item.get('edit', 0.0)):.8f}",
                "f1_10": f"{float(item.get('f1_10', 0.0)):.8f}",
                "f1_25": f"{float(item.get('f1_25', 0.0)):.8f}",
                "f1_50": f"{float(item.get('f1_50', 0.0)):.8f}",
            }
        )
    return rows


def uncertainty_score_maps(
    deterministic_probs: np.ndarray,
    deterministic_calibrated_probs: np.ndarray,
    raw_quantities: dict[str, np.ndarray],
    calibrated_quantities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "deterministic_1_confidence": 1.0 - np.max(deterministic_probs, axis=1),
        "deterministic_calibrated_1_confidence": 1.0 - np.max(deterministic_calibrated_probs, axis=1),
        "mc_raw_1_confidence": 1.0 - raw_quantities["confidence"],
        "mc_raw_predictive_entropy": raw_quantities["predictive_entropy"],
        "mc_raw_mutual_information": raw_quantities["mutual_information"],
        "mc_raw_variation_ratio": raw_quantities["variation_ratio"],
        "mc_raw_mean_probability_variance": raw_quantities["mean_probability_variance"],
        "mc_calibrated_1_confidence": 1.0 - calibrated_quantities["confidence"],
        "mc_calibrated_predictive_entropy": calibrated_quantities["predictive_entropy"],
        "mc_calibrated_mutual_information": calibrated_quantities["mutual_information"],
        "mc_calibrated_variation_ratio": calibrated_quantities["variation_ratio"],
        "mc_calibrated_mean_probability_variance": calibrated_quantities["mean_probability_variance"],
    }


def uncertainty_quality_rows(
    scores: dict[str, np.ndarray],
    predictions_by_mode: dict[str, np.ndarray],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for name, values in scores.items():
        predictions = predictions_for_score(name, predictions_by_mode)
        errors = predictions != labels
        correct_values = values[~errors]
        error_values = values[errors]
        quantiles = np.percentile(values, [25, 50, 75]) if len(values) else [0, 0, 0]
        rows.append(
            {
                "uncertainty": name,
                "positive_class": "incorrect_prediction",
                "error_auroc": binary_auroc(values, errors),
                "error_auprc": binary_auprc(values, errors),
                "mean_uncertainty_correct": float(np.mean(correct_values)) if len(correct_values) else float("nan"),
                "mean_uncertainty_incorrect": float(np.mean(error_values)) if len(error_values) else float("nan"),
                "q25": float(quantiles[0]),
                "median": float(quantiles[1]),
                "q75": float(quantiles[2]),
            }
        )
    return rows


def predictions_for_score(name: str, predictions_by_mode: dict[str, np.ndarray]) -> np.ndarray:
    if name.startswith("deterministic_calibrated_"):
        return predictions_by_mode["deterministic_calibrated"]
    if name.startswith("deterministic_"):
        return predictions_by_mode["deterministic"]
    if name.startswith("mc_raw_"):
        return predictions_by_mode["mc_raw"]
    return predictions_by_mode["mc_calibrated"]


def probabilities_for_score(name: str, probabilities_by_mode: dict[str, np.ndarray]) -> np.ndarray:
    if name.startswith("deterministic_calibrated_"):
        return probabilities_by_mode["deterministic_calibrated"]
    if name.startswith("deterministic_"):
        return probabilities_by_mode["deterministic"]
    if name.startswith("mc_raw_"):
        return probabilities_by_mode["mc_raw"]
    return probabilities_by_mode["mc_calibrated"]


def retained_indices(uncertainty: np.ndarray, coverage: float) -> np.ndarray:
    n_samples = uncertainty.shape[0]
    keep = max(1, int(round(n_samples * coverage)))
    order = np.argsort(uncertainty, kind="mergesort")
    return np.sort(order[:keep])


def class_distribution_fields(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    retained = labels[indices]
    output = {}
    for class_id, label in enumerate(LABELS):
        output[f"retained_{label}"] = int(np.count_nonzero(retained == class_id))
    return output


def transition_retention_count(labels: np.ndarray, indices: np.ndarray) -> int:
    transition_ids = [
        LABEL_TO_ID[label]
        for label in LABELS
        if label == "Falling" or "Transition" in label
    ]
    retained = labels[indices]
    return int(np.count_nonzero(np.isin(retained, transition_ids)))


def coverage_rows(
    scores: dict[str, np.ndarray],
    probabilities_by_mode: dict[str, np.ndarray],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for score_name, uncertainty in scores.items():
        probabilities = probabilities_for_score(score_name, probabilities_by_mode)
        predictions = np.argmax(probabilities, axis=1)
        for coverage in COVERAGE_LEVELS:
            indices = retained_indices(uncertainty, coverage)
            retained_labels = labels[indices]
            retained_predictions = predictions[indices]
            if len(np.unique(retained_labels)) == 0:
                macro_f1 = float("nan")
            else:
                retained_probs = probabilities[indices]
                macro_f1 = float(
                    classification_metrics(retained_probs, retained_labels, LABELS, ece_bins=15)["macro_f1"]
                )
            accuracy = float(np.mean(retained_predictions == retained_labels))
            row = {
                "uncertainty": score_name,
                "coverage": f"{coverage:.2f}",
                "retained_samples": int(indices.shape[0]),
                "center_time_accuracy": f"{accuracy:.8f}",
                "selective_risk": f"{1.0 - accuracy:.8f}",
                "macro_f1": "" if np.isnan(macro_f1) else f"{macro_f1:.8f}",
                "transition_retained": transition_retention_count(labels, indices),
            }
            row.update(class_distribution_fields(labels, indices))
            rows.append(row)
    return rows


def per_class_uncertainty_rows(
    samples: list[SampleMeta],
    labels: np.ndarray,
    raw_quantities: dict[str, np.ndarray],
    calibrated_quantities: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for mode, quantities in (("mc_raw", raw_quantities), ("mc_calibrated", calibrated_quantities)):
        errors = quantities["prediction"] != labels
        for score in UNCERTAINTY_SCORE_COLUMNS:
            if score == "one_minus_confidence":
                values = 1.0 - quantities["confidence"]
            else:
                values = quantities[score]
            for class_id, label_name in enumerate(LABELS):
                mask = labels == class_id
                class_values = values[mask]
                error_values = values[mask & errors]
                correct_values = values[mask & ~errors]
                if len(class_values):
                    q25, q50, q75 = np.percentile(class_values, [25, 50, 75])
                else:
                    q25 = q50 = q75 = float("nan")
                rows.append(
                    {
                        "mode": mode,
                        "uncertainty": score,
                        "class_id": class_id,
                        "label": label_name,
                        "support": int(np.count_nonzero(mask)),
                        "errors": int(np.count_nonzero(mask & errors)),
                        "mean": float(np.mean(class_values)) if len(class_values) else float("nan"),
                        "mean_correct": float(np.mean(correct_values)) if len(correct_values) else float("nan"),
                        "mean_incorrect": float(np.mean(error_values)) if len(error_values) else float("nan"),
                        "q25": float(q25),
                        "median": float(q50),
                        "q75": float(q75),
                    }
                )
    return rows


def array_to_text(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.8g}" for value in values)


def sample_rows(
    samples: list[SampleMeta],
    det_logits: np.ndarray,
    det_probs: np.ndarray,
    det_cal_probs: np.ndarray,
    mc_raw_quantities: dict[str, np.ndarray],
    mc_cal_quantities: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    det_predictions = np.argmax(det_probs, axis=1)
    det_cal_predictions = np.argmax(det_cal_probs, axis=1)
    raw_predictions = mc_raw_quantities["prediction"]
    cal_predictions = mc_cal_quantities["prediction"]
    rows = []
    for index, item in enumerate(samples):
        rows.append(
            {
                "index": item.index,
                "subject_id": item.subject_id,
                "recording_id": item.recording_id,
                "sequence_id": item.sequence_id,
                "frame_dir": item.frame_dir,
                "window_start_frame": item.window_start_frame,
                "window_end_frame": item.window_end_frame,
                "center_frame": item.center_frame,
                "center_timestamp": f"{item.center_timestamp:.9f}",
                "ground_truth_id": item.label_id,
                "ground_truth_label": item.label_name,
                "deterministic_pred_id": int(det_predictions[index]),
                "deterministic_pred_label": LABELS[int(det_predictions[index])],
                "deterministic_calibrated_pred_id": int(det_cal_predictions[index]),
                "deterministic_calibrated_pred_label": LABELS[int(det_cal_predictions[index])],
                "mc_raw_pred_id": int(raw_predictions[index]),
                "mc_raw_pred_label": LABELS[int(raw_predictions[index])],
                "mc_calibrated_pred_id": int(cal_predictions[index]),
                "mc_calibrated_pred_label": LABELS[int(cal_predictions[index])],
                "deterministic_confidence": f"{float(np.max(det_probs[index])):.8f}",
                "deterministic_calibrated_confidence": f"{float(np.max(det_cal_probs[index])):.8f}",
                "mc_raw_confidence": f"{float(mc_raw_quantities['confidence'][index]):.8f}",
                "mc_calibrated_confidence": f"{float(mc_cal_quantities['confidence'][index]):.8f}",
                "mc_raw_predictive_entropy": f"{float(mc_raw_quantities['predictive_entropy'][index]):.8f}",
                "mc_raw_expected_entropy": f"{float(mc_raw_quantities['expected_entropy'][index]):.8f}",
                "mc_raw_mutual_information": f"{float(mc_raw_quantities['mutual_information'][index]):.8f}",
                "mc_raw_variation_ratio": f"{float(mc_raw_quantities['variation_ratio'][index]):.8f}",
                "mc_raw_mean_probability_variance": f"{float(mc_raw_quantities['mean_probability_variance'][index]):.8f}",
                "mc_calibrated_predictive_entropy": f"{float(mc_cal_quantities['predictive_entropy'][index]):.8f}",
                "mc_calibrated_expected_entropy": f"{float(mc_cal_quantities['expected_entropy'][index]):.8f}",
                "mc_calibrated_mutual_information": f"{float(mc_cal_quantities['mutual_information'][index]):.8f}",
                "mc_calibrated_variation_ratio": f"{float(mc_cal_quantities['variation_ratio'][index]):.8f}",
                "mc_calibrated_mean_probability_variance": f"{float(mc_cal_quantities['mean_probability_variance'][index]):.8f}",
                "deterministic_logits": array_to_text(det_logits[index]),
                "deterministic_probabilities": array_to_text(det_probs[index]),
                "deterministic_calibrated_probabilities": array_to_text(det_cal_probs[index]),
                "mc_mean_raw_probabilities": array_to_text(mc_raw_quantities["probabilities"][index]),
                "mc_mean_calibrated_probabilities": array_to_text(mc_cal_quantities["probabilities"][index]),
            }
        )
    return rows


def save_npz_outputs(
    out_dir: Path,
    calibration: dict[str, Any],
    test: dict[str, Any],
    deterministic_temperature: float,
    mc_temperature: float,
) -> None:
    np.savez_compressed(
        out_dir / "deterministic_predictions.npz",
        calibration_logits=calibration["det_logits"],
        calibration_probabilities=calibration["det_probs"],
        calibration_calibrated_probabilities=calibration["det_cal_probs"],
        calibration_labels=calibration["labels"],
        test_logits=test["det_logits"],
        test_probabilities=test["det_probs"],
        test_calibrated_probabilities=test["det_cal_probs"],
        test_labels=test["labels"],
        deterministic_temperature=np.asarray([deterministic_temperature], dtype=np.float64),
    )
    np.savez_compressed(
        out_dir / "calibration_mc_logits.npz",
        mc_logits=calibration["mc_logits"],
        labels=calibration["labels"],
        mean_raw_probabilities=calibration["mc_raw"]["probabilities"],
        mean_calibrated_probabilities=calibration["mc_calibrated"]["probabilities"],
        temperature=np.asarray([mc_temperature], dtype=np.float64),
    )
    np.savez_compressed(
        out_dir / "test_mc_logits.npz",
        mc_logits=test["mc_logits"],
        labels=test["labels"],
        mean_raw_probabilities=test["mc_raw"]["probabilities"],
        mean_calibrated_probabilities=test["mc_calibrated"]["probabilities"],
        temperature=np.asarray([mc_temperature], dtype=np.float64),
    )


def save_sequence_outputs(out_dir: Path, samples: list[SampleMeta], test: dict[str, Any]) -> None:
    by_sequence: dict[str, list[int]] = {}
    for index, item in enumerate(samples):
        by_sequence.setdefault(item.sequence_id, []).append(index)
    sequence_dir = out_dir / "sequences"
    for sequence_id, indices in by_sequence.items():
        indices = sorted(indices, key=lambda idx: (samples[idx].center_timestamp, samples[idx].window_start_frame))
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sequence_id)
        np.savez_compressed(
            sequence_dir / f"{safe_name}.npz",
            sequence_id=np.asarray([sequence_id]),
            subject_id=np.asarray([samples[indices[0]].subject_id]),
            recording_id=np.asarray([samples[indices[0]].recording_id]),
            center_frame=np.asarray([samples[idx].center_frame for idx in indices], dtype=np.int64),
            center_timestamp=np.asarray([samples[idx].center_timestamp for idx in indices], dtype=np.float64),
            ground_truth_label_id=np.asarray([samples[idx].label_id for idx in indices], dtype=np.int64),
            ground_truth_label=np.asarray([samples[idx].label_name for idx in indices]),
            deterministic_logits=test["det_logits"][indices],
            deterministic_probabilities=test["det_probs"][indices],
            deterministic_calibrated_probabilities=test["det_cal_probs"][indices],
            mc_mean_raw_probabilities=test["mc_raw"]["probabilities"][indices],
            mc_mean_calibrated_probabilities=test["mc_calibrated"]["probabilities"][indices],
            deterministic_pred_id=np.argmax(test["det_probs"][indices], axis=1).astype(np.int64),
            deterministic_calibrated_pred_id=np.argmax(test["det_cal_probs"][indices], axis=1).astype(np.int64),
            mc_raw_pred_id=test["mc_raw"]["prediction"][indices].astype(np.int64),
            mc_calibrated_pred_id=test["mc_calibrated"]["prediction"][indices].astype(np.int64),
            predictive_entropy=test["mc_calibrated"]["predictive_entropy"][indices],
            expected_entropy=test["mc_calibrated"]["expected_entropy"][indices],
            mutual_information=test["mc_calibrated"]["mutual_information"][indices],
            variation_ratio=test["mc_calibrated"]["variation_ratio"][indices],
            raw_mutual_information=test["mc_raw"]["mutual_information"][indices],
        )


def save_confusion_csv(path: Path, matrix: list[list[int]], method: str) -> None:
    rows = []
    for row_index, label in enumerate(LABELS):
        item = {"method": method, "true_label": label}
        for col_index, pred_label in enumerate(LABELS):
            item[f"pred_{pred_label}"] = int(matrix[row_index][col_index])
        rows.append(item)
    write_csv(path, rows)


def make_plots(
    out_dir: Path,
    labels: np.ndarray,
    metrics: dict[str, dict[str, Any]],
    det_probs: np.ndarray,
    det_cal_probs: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    coverage: list[dict[str, Any]],
    calibrated_quantities: dict[str, np.ndarray],
    per_class_uncertainty: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; plots were not written: {exc}")
        return warnings

    plot_dir = out_dir / "plots"

    def reliability(path: Path, probs: np.ndarray, title: str) -> None:
        bins = reliability_bins(probs, labels, num_bins=15)
        centers = [(float(row["lower"]) + float(row["upper"])) / 2.0 for row in bins]
        acc = [float(row["accuracy"]) for row in bins]
        conf = [float(row["confidence"]) for row in bins]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.bar(centers, acc, width=1 / 15, alpha=0.7, label="accuracy")
        ax.plot(centers, conf, marker="o", label="confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    reliability(plot_dir / "reliability_diagram_deterministic.png", det_probs, "Deterministic")
    reliability(
        plot_dir / "reliability_diagram_deterministic_calibrated.png",
        det_cal_probs,
        "Deterministic calibrated",
    )
    reliability(plot_dir / "reliability_diagram_mc_raw.png", raw_probs, "MC raw")
    reliability(plot_dir / "reliability_diagram_mc_calibrated.png", cal_probs, "MC calibrated")

    selected_scores = {
        "deterministic_1_confidence",
        "deterministic_calibrated_1_confidence",
        "mc_raw_predictive_entropy",
        "mc_raw_mutual_information",
        "mc_calibrated_mutual_information",
    }
    for y_key, filename, ylabel in (
        ("selective_risk", "risk_coverage.png", "selective risk"),
        ("center_time_accuracy", "coverage_accuracy.png", "accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for score in sorted(selected_scores):
            rows = [row for row in coverage if row["uncertainty"] == score]
            rows = sorted(rows, key=lambda row: float(row["coverage"]))
            ax.plot(
                [float(row["coverage"]) for row in rows],
                [float(row[y_key]) for row in rows],
                marker="o",
                label=score,
            )
        ax.set_xlabel("coverage")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    errors = np.argmax(cal_probs, axis=1) != labels
    fig, ax = plt.subplots(figsize=(5, 5))
    values = calibrated_quantities["mutual_information"]
    ax.boxplot([values[~errors], values[errors]], labels=["correct", "incorrect"])
    ax.set_ylabel("calibrated MC mutual information")
    fig.tight_layout()
    fig.savefig(plot_dir / "uncertainty_correct_vs_incorrect.png", dpi=160)
    plt.close(fig)

    def confusion_plot(path: Path, matrix: list[list[int]], title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(np.asarray(matrix), cmap="Blues")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
        ax.set_yticklabels(LABELS, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.75)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    confusion_plot(
        plot_dir / "confusion_matrix_deterministic.png",
        metrics["deterministic"]["confusion_matrix"],
        "Deterministic confusion matrix",
    )
    confusion_plot(
        plot_dir / "confusion_matrix_deterministic_calibrated.png",
        metrics["deterministic_calibrated"]["confusion_matrix"],
        "Deterministic calibrated confusion matrix",
    )
    confusion_plot(
        plot_dir / "confusion_matrix_mc_calibrated.png",
        metrics["mc_calibrated"]["confusion_matrix"],
        "MC calibrated confusion matrix",
    )

    rows = [
        row
        for row in per_class_uncertainty
        if row["mode"] == "mc_calibrated" and row["uncertainty"] == "mutual_information"
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row["label"] for row in rows], [float(row["mean"]) for row in rows])
    ax.set_ylabel("mean calibrated MI")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_class_uncertainty.png", dpi=160)
    plt.close(fig)

    return warnings


def write_readme(
    out_dir: Path,
    command: str,
    experiment: dict[str, Any],
    deterministic_temperature: dict[str, Any],
    mc_temperature: dict[str, Any],
    dropout_info: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    text = f"""# Study 3 PoseC3D MC-Dropout Evaluation

## Data Split

The checkpoint-selection validation subject is not used for temperature fitting.
Temperature is fitted on `{experiment['calibration_split']}` only. Final metrics
are reported on `{experiment['test_split']}` only.

Calibration subjects: {', '.join(experiment['split_safety']['calibration_subjects'])}

Test subjects: {', '.join(experiment['split_safety']['test_subjects'])}

## Checkpoint

Config: `{experiment['config']}`

Checkpoint: `{experiment['checkpoint']}`

The script changes only the evaluation config to request raw logits
(`test_cfg.average_clips='score'`). It does not retrain, alter the architecture,
or modify checkpoint weights.

## MC-Dropout

The model is placed in eval mode globally, then only dropout modules are returned
to train mode. BatchNorm layers remain frozen.

Dropout modules found: {len(dropout_info)}

MC passes: {experiment['num_passes']}

## Temperature Objective

Two scalar temperatures are fitted on `{experiment['calibration_split']}` only.

The deterministic temperature `T_det` is fitted on deterministic logits by
minimizing:

`-mean(log(softmax(z / T_det)[y]))`

The MC temperature `T_mc` is fitted on calibration MC logits by minimizing:

`-mean(log(mean_k softmax(z_k / T_mc)[y]))`

Fitted deterministic T: {deterministic_temperature['temperature']:.8f}

Fitted MC T: {mc_temperature['temperature']:.8f}

## Uncertainty Values

Saved per window: confidence, top-1/top-2 margin, predictive entropy, expected
entropy, mutual information, variation ratio, and probability variance. The
primary downstream epistemic score is calibrated MC mutual information.

## Metrics

The run reports center-time accuracy, macro-F1, per-class F1, confusion matrix,
NLL, Brier score, ECE, Edit, and F1@10/25/50 on the center-time grid.

## Reproduction Command

```bash
{command}
```

## Limitations

No uncertainty threshold is selected here. Coverage/risk curves are exported for
later threshold selection outside the test subject.
"""
    if warnings:
        text += "\n## Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n"
    out_dir.joinpath("README_experiment.md").write_text(text, encoding="utf-8")


def print_tables(
    summary: dict[str, dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    deterministic_temperature: dict[str, Any],
    mc_temperature: dict[str, Any],
    dropout_info: list[dict[str, Any]],
    num_passes: int,
    calibration_count: int,
    test_count: int,
    labels: np.ndarray,
    calibrated_predictions: np.ndarray,
) -> None:
    print("\nMethod | Acc | Macro-F1 | NLL | Brier | ECE")
    for row in metrics_table_rows(summary):
        print(
            f"{row['method']} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['nll']):.4f} | "
            f"{float(row['brier']):.4f} | {float(row['ece']):.4f}"
        )

    print("\nUncertainty | Error AUROC | Error AUPRC | Acc@80% coverage | Acc@60% coverage")
    display = {
        "deterministic_1_confidence": "Deterministic 1-confidence",
        "deterministic_calibrated_1_confidence": "Deterministic calibrated 1-confidence",
        "mc_calibrated_predictive_entropy": "Predictive entropy",
        "mc_calibrated_mutual_information": "MC mutual information",
        "mc_calibrated_variation_ratio": "Variation ratio",
    }
    for key, name in display.items():
        quality = next(row for row in uncertainty_rows if row["uncertainty"] == key)
        acc80 = next(
            row for row in coverage
            if row["uncertainty"] == key and abs(float(row["coverage"]) - 0.80) < 1e-9
        )
        acc60 = next(
            row for row in coverage
            if row["uncertainty"] == key and abs(float(row["coverage"]) - 0.60) < 1e-9
        )
        print(
            f"{name} | {float(quality['error_auroc']):.4f} | "
            f"{float(quality['error_auprc']):.4f} | "
            f"{float(acc80['center_time_accuracy']):.4f} | "
            f"{float(acc60['center_time_accuracy']):.4f}"
        )

    errors = calibrated_predictions != labels
    print(f"\nFitted deterministic temperature: {deterministic_temperature['temperature']:.6f}")
    print(f"Fitted MC temperature: {mc_temperature['temperature']:.6f}")
    print(f"Dropout modules found: {len(dropout_info)}")
    print(f"MC passes: {num_passes}")
    print(f"Calibration samples: {calibration_count}")
    print(f"Test samples: {test_count}")
    print(f"Test error count: {int(np.count_nonzero(errors))}")
    for coverage_level in (0.80, 0.60):
        row = next(
            row for row in coverage
            if row["uncertainty"] == "mc_calibrated_mutual_information"
            and abs(float(row["coverage"]) - coverage_level) < 1e-9
        )
        print(
            f"Transition retained at {int(coverage_level * 100)}% coverage "
            f"(calibrated MI): {row['transition_retained']}"
        )


def main() -> None:
    args = parse_args()
    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")
    if args.ece_bins <= 0:
        raise ValueError("--ece-bins must be positive")
    for path in (args.config, args.checkpoint, args.calibration_ann, args.test_ann):
        if not path.exists():
            raise FileNotFoundError(path)

    ensure_output_dir(args.out_dir, args.overwrite)
    command = " ".join(sys.argv)
    (args.out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    shutil.copyfile(args.config, args.out_dir / "config_snapshot.py")
    environment = write_environment(args.out_dir / "environment.txt")

    set_global_seed(args.seed)
    config = load_config(args.config)
    calibration_split, test_split = infer_splits(config, args.calibration_split, args.test_split)

    import torch
    from pyskl.apis import init_recognizer

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available")

    print(f"[INFO] Loading model config={args.config} checkpoint={args.checkpoint}")
    model = init_recognizer(config, str(args.checkpoint), device=device)

    print(f"[INFO] Building calibration dataset split={calibration_split}")
    calibration_dataset, calibration_loader = build_dataset_and_loader(
        config,
        args.calibration_ann,
        calibration_split,
        args.batch_size,
        args.num_workers,
    )
    print(f"[INFO] Building test dataset split={test_split}")
    test_dataset, test_loader = build_dataset_and_loader(
        config,
        args.test_ann,
        test_split,
        args.batch_size,
        args.num_workers,
    )

    calibration_samples = sample_metadata(calibration_dataset)
    test_samples = sample_metadata(test_dataset)
    calibration_labels = assert_valid_labels(calibration_samples, "calibration")
    test_labels = assert_valid_labels(test_samples, "test")
    assert_monotonic_by_sequence(calibration_samples, "calibration")
    assert_monotonic_by_sequence(test_samples, "test")
    split_safety = validate_split_safety(calibration_samples, test_samples, config)

    experiment = {
        "experiment": "S3",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_filename": args.checkpoint.name,
        "calibration_ann": str(args.calibration_ann),
        "test_ann": str(args.test_ann),
        "calibration_split": calibration_split,
        "test_split": test_split,
        "out_dir": str(args.out_dir),
        "num_passes": args.num_passes,
        "ece_bins": args.ece_bins,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "center_offset": CENTER_OFFSET,
        "temporal_resolution_seconds": STRIDE / FPS,
        "labels": LABELS,
        "split_safety": split_safety,
        "environment": environment,
        "created_at_unix": time.time(),
        "save_mc_logits": True,
    }
    write_json(args.out_dir / "experiment_config.json", experiment)

    print("[INFO] Running deterministic logits")
    calibration_det_logits = deterministic_logits(model, calibration_loader, len(calibration_samples))
    test_det_logits = deterministic_logits(model, test_loader, len(test_samples))
    calibration_det_probs = softmax_np(calibration_det_logits)
    test_det_probs = softmax_np(test_det_logits)
    validate_probabilities(calibration_det_probs, name="calibration_det_probs")
    validate_probabilities(test_det_probs, name="test_det_probs")

    print("[INFO] Fitting scalar temperature on deterministic calibration logits")
    deterministic_temperature = fit_temperature(
        calibration_det_logits[:, None, :],
        calibration_labels,
        num_bins=args.ece_bins,
    )
    write_json(args.out_dir / "deterministic_temperature.json", deterministic_temperature)
    fitted_deterministic_temperature = float(deterministic_temperature["temperature"])
    calibration_det_cal_probs = softmax_np(
        calibration_det_logits,
        temperature=fitted_deterministic_temperature,
    )
    test_det_cal_probs = softmax_np(
        test_det_logits,
        temperature=fitted_deterministic_temperature,
    )
    validate_probabilities(calibration_det_cal_probs, name="calibration_det_cal_probs")
    validate_probabilities(test_det_cal_probs, name="test_det_cal_probs")

    print(f"[INFO] Running MC-dropout logits K={args.num_passes} on calibration")
    calibration_mc_logits, dropout_info, first_batch_diff = mc_logits(
        model,
        calibration_loader,
        len(calibration_samples),
        args.num_passes,
    )
    print(f"[INFO] MC first-batch pass difference: {first_batch_diff:.6g}")
    print(f"[INFO] Running MC-dropout logits K={args.num_passes} on test")
    test_mc_logits, _, _ = mc_logits(
        model,
        test_loader,
        len(test_samples),
        args.num_passes,
    )

    print("[INFO] Fitting scalar temperature on calibration MC logits")
    mc_temperature = fit_temperature(
        calibration_mc_logits,
        calibration_labels,
        num_bins=args.ece_bins,
    )
    write_json(args.out_dir / "temperature.json", mc_temperature)
    write_json(args.out_dir / "mc_temperature.json", mc_temperature)
    write_json(
        args.out_dir / "temperatures.json",
        {
            "deterministic": deterministic_temperature,
            "mc": mc_temperature,
        },
    )
    fitted_mc_temperature = float(mc_temperature["temperature"])

    calibration_raw_probs_passes = softmax_np(calibration_mc_logits)
    calibration_cal_probs_passes = softmax_np(calibration_mc_logits, temperature=fitted_mc_temperature)
    test_raw_probs_passes = softmax_np(test_mc_logits)
    test_cal_probs_passes = softmax_np(test_mc_logits, temperature=fitted_mc_temperature)

    calibration_raw_quantities = predictive_quantities(calibration_raw_probs_passes)
    calibration_cal_quantities = predictive_quantities(calibration_cal_probs_passes)
    test_raw_quantities = predictive_quantities(test_raw_probs_passes)
    test_cal_quantities = predictive_quantities(test_cal_probs_passes)

    test_summary = {
        "deterministic": metrics_with_sequence(
            test_det_probs,
            test_labels,
            test_samples,
            args.ece_bins,
            "D0",
        ),
        "deterministic_calibrated": metrics_with_sequence(
            test_det_cal_probs,
            test_labels,
            test_samples,
            args.ece_bins,
            "D0T",
        ),
        "mc_raw": metrics_with_sequence(
            test_raw_quantities["probabilities"],
            test_labels,
            test_samples,
            args.ece_bins,
            "D1",
        ),
        "mc_calibrated": metrics_with_sequence(
            test_cal_quantities["probabilities"],
            test_labels,
            test_samples,
            args.ece_bins,
            "D2",
        ),
    }

    calibration_summary = {
        "deterministic": metrics_with_sequence(
            calibration_det_probs,
            calibration_labels,
            calibration_samples,
            args.ece_bins,
            "D0_calibration",
        ),
        "deterministic_calibrated": metrics_with_sequence(
            calibration_det_cal_probs,
            calibration_labels,
            calibration_samples,
            args.ece_bins,
            "D0T_calibration",
        ),
        "mc_raw": metrics_with_sequence(
            calibration_raw_quantities["probabilities"],
            calibration_labels,
            calibration_samples,
            args.ece_bins,
            "D1_calibration",
        ),
        "mc_calibrated": metrics_with_sequence(
            calibration_cal_quantities["probabilities"],
            calibration_labels,
            calibration_samples,
            args.ece_bins,
            "D2_calibration",
        ),
    }

    uncertainty_scores = uncertainty_score_maps(
        test_det_probs,
        test_det_cal_probs,
        test_raw_quantities,
        test_cal_quantities,
    )
    uncertainty_rows = uncertainty_quality_rows(
        uncertainty_scores,
        {
            "deterministic": np.argmax(test_det_probs, axis=1),
            "deterministic_calibrated": np.argmax(test_det_cal_probs, axis=1),
            "mc_raw": test_raw_quantities["prediction"],
            "mc_calibrated": test_cal_quantities["prediction"],
        },
        test_labels,
    )
    coverage = coverage_rows(
        uncertainty_scores,
        {
            "deterministic": test_det_probs,
            "deterministic_calibrated": test_det_cal_probs,
            "mc_raw": test_raw_quantities["probabilities"],
            "mc_calibrated": test_cal_quantities["probabilities"],
        },
        test_labels,
    )
    per_class_uncertainty = per_class_uncertainty_rows(
        test_samples,
        test_labels,
        test_raw_quantities,
        test_cal_quantities,
    )

    calibration = {
        "labels": calibration_labels,
        "det_logits": calibration_det_logits,
        "det_probs": calibration_det_probs,
        "det_cal_probs": calibration_det_cal_probs,
        "mc_logits": calibration_mc_logits,
        "mc_raw": calibration_raw_quantities,
        "mc_calibrated": calibration_cal_quantities,
    }
    test = {
        "labels": test_labels,
        "det_logits": test_det_logits,
        "det_probs": test_det_probs,
        "det_cal_probs": test_det_cal_probs,
        "mc_logits": test_mc_logits,
        "mc_raw": test_raw_quantities,
        "mc_calibrated": test_cal_quantities,
    }
    save_npz_outputs(
        args.out_dir,
        calibration,
        test,
        fitted_deterministic_temperature,
        fitted_mc_temperature,
    )
    save_sequence_outputs(args.out_dir, test_samples, test)

    write_csv(
        args.out_dir / "calibration_samples.csv",
        sample_rows(
            calibration_samples,
            calibration_det_logits,
            calibration_det_probs,
            calibration_det_cal_probs,
            calibration_raw_quantities,
            calibration_cal_quantities,
        ),
    )
    write_csv(
        args.out_dir / "test_samples.csv",
        sample_rows(
            test_samples,
            test_det_logits,
            test_det_probs,
            test_det_cal_probs,
            test_raw_quantities,
            test_cal_quantities,
        ),
    )
    write_csv(args.out_dir / "metrics_table.csv", metrics_table_rows(test_summary))
    write_csv(args.out_dir / "coverage_accuracy.csv", coverage)
    write_csv(args.out_dir / "risk_coverage.csv", coverage)
    write_csv(args.out_dir / "per_class_uncertainty.csv", per_class_uncertainty)
    write_csv(args.out_dir / "uncertainty_error_detection.csv", uncertainty_rows)
    save_confusion_csv(
        args.out_dir / "confusion_matrix_deterministic.csv",
        test_summary["deterministic"]["confusion_matrix"],
        "deterministic",
    )
    save_confusion_csv(
        args.out_dir / "confusion_matrix_deterministic_calibrated.csv",
        test_summary["deterministic_calibrated"]["confusion_matrix"],
        "deterministic_calibrated",
    )
    save_confusion_csv(
        args.out_dir / "confusion_matrix_mc_calibrated.csv",
        test_summary["mc_calibrated"]["confusion_matrix"],
        "mc_calibrated",
    )

    plot_warnings = make_plots(
        args.out_dir,
        test_labels,
        test_summary,
        test_det_probs,
        test_det_cal_probs,
        test_raw_quantities["probabilities"],
        test_cal_quantities["probabilities"],
        coverage,
        test_cal_quantities,
        per_class_uncertainty,
    )

    summary_metrics = {
        "experiment": experiment,
        "temperature": mc_temperature,
        "deterministic_temperature": deterministic_temperature,
        "mc_temperature": mc_temperature,
        "temperatures": {
            "deterministic": deterministic_temperature,
            "mc": mc_temperature,
        },
        "dropout_modules": dropout_info,
        "mc_first_batch_pass1_pass2_mean_abs_diff": first_batch_diff,
        "calibration": calibration_summary,
        "test": test_summary,
        "uncertainty_error_detection": uncertainty_rows,
        "plot_warnings": plot_warnings,
        "positive_class_for_auroc_auprc": "incorrect_prediction",
        "primary_epistemic_uncertainty": "mc_calibrated_mutual_information",
    }
    write_json(args.out_dir / "summary_metrics.json", summary_metrics)
    write_readme(
        args.out_dir,
        command,
        experiment,
        deterministic_temperature,
        mc_temperature,
        dropout_info,
        plot_warnings,
    )

    print_tables(
        test_summary,
        uncertainty_rows,
        coverage,
        deterministic_temperature,
        mc_temperature,
        dropout_info,
        args.num_passes,
        len(calibration_samples),
        len(test_samples),
        test_labels,
        test_cal_quantities["prediction"],
    )


if __name__ == "__main__":
    main()
