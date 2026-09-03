"""E2A MC-dropout validation inference for ST-GCN++ B checkpoints.

This script evaluates the raw MC-dropout predictive branch on the E1-B
continuous-window validation split. It uses the selected E1-B joint and bone
checkpoints, enables only ``cls_head.dropout`` during stochastic inference, and
fuses joint/bone probabilities inside every MC pass.

No temperature scaling is applied.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LABELS = [
    "lie-stationary",
    "sit-stationary",
    "walk",
    "fall",
    "transition-lie-to-sit",
    "transition-lie-to-stand",
    "transition-sit-to-lie",
    "transition-sit-to-stand",
    "transition-stand-to-sit",
]
STATE_CLASS_IDS = [0, 1, 2]
TRANSITION_CLASS_IDS = [3, 4, 5, 6, 7, 8]
FOLDS = ["a", "b", "c"]
STREAMS = ["joint", "bone"]
CONDITION_KEY = "b"
CONDITION_DIR = "b_continuous_window"
PKL_PROTOCOL_DIR = "continuous_window_w60_s12"
PKL_STEM = "radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"


@dataclass(frozen=True)
class StreamResult:
    probabilities: np.ndarray
    labels: np.ndarray
    sample_ids: list[dict[str, Any]]
    output_dir: Path


def install_numpy_pickle_compat_aliases() -> None:
    """Allow NumPy 2-generated pickles to load under NumPy 1.x."""

    try:
        importlib.import_module("numpy._core.numeric")
        return
    except ModuleNotFoundError:
        pass

    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for alias, target in aliases.items():
        try:
            sys.modules.setdefault(alias, importlib.import_module(target))
        except ModuleNotFoundError:
            continue


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_global_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def continuous_pkl_path(data_root: Path, fold: str) -> Path:
    return data_root / PKL_PROTOCOL_DIR / PKL_STEM.format(fold=fold)


def validation_config_path(config_root: Path, fold: str, stream: str) -> Path:
    return config_root / f"fold_{fold}" / stream / "validation" / "b_validation.py"


def find_selected_checkpoint(e1_work_root: Path, fold: str, stream: str) -> Path:
    checkpoint_dir = e1_work_root / f"fold_{fold}" / stream / CONDITION_DIR
    checkpoints = sorted(checkpoint_dir.glob("best_macro_f1_epoch_*.pth"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected one best_macro_f1 checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def stream_output_dir(output_root: Path, fold: str, stream: str) -> Path:
    return output_root / f"fold_{fold}" / stream / CONDITION_DIR / "validation"


def fusion_output_dir(output_root: Path, fold: str) -> Path:
    return output_root / f"fold_{fold}" / "fusion" / CONDITION_DIR / "validation"


def sample_id_from_annotation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_dir": str(item.get("frame_dir", item.get("filename", ""))),
        "session_name": str(item.get("session_name", "")),
        "window_row_start": int(item["window_row_start"]),
        "center_source_frame": int(item["center_source_frame"]),
    }


def split_annotations(pkl_file: Path, split_name: str = "val") -> list[dict[str, Any]]:
    data = load_pickle(pkl_file)
    annotations = data["annotations"]
    identifier = "filename" if "filename" in annotations[0] else "frame_dir"
    split_ids = set(data["split"][split_name])
    return [item for item in annotations if item[identifier] in split_ids]


def validation_labels_and_ids(data_root: Path, fold: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annotations = split_annotations(continuous_pkl_path(data_root, fold), "val")
    labels = np.array([int(item["label"]) for item in annotations], dtype=np.int64)
    sample_ids = [sample_id_from_annotation(item) for item in annotations]
    return labels, sample_ids


def validate_probabilities(values: np.ndarray, name: str, atol: float = 1e-5) -> None:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape[-1] != len(LABELS):
        raise ValueError(
            f"{name} expected {len(LABELS)} classes, got shape {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError(f"{name} contains values outside [0, 1]")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol, rtol=0.0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to 1; max error={max_error}")


def scores_to_probabilities(scores: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected score array [N, C], got {values.shape}")
    row_sums = values.sum(axis=1)
    if (
        np.all(values >= -1e-6)
        and np.all(values <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-4)
    ):
        validate_probabilities(values, "loaded deterministic probabilities")
        return values, "already_probabilities"

    stable = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(stable)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    validate_probabilities(probabilities, "softmax probabilities")
    return probabilities, "softmax_applied"


def load_scores(path: Path) -> np.ndarray:
    return np.asarray(load_pickle(path), dtype=np.float64)


def deterministic_fusion_probabilities(
    e1_work_root: Path,
    fold: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    fusion_path = (
        e1_work_root
        / f"fold_{fold}"
        / "fusion"
        / CONDITION_DIR
        / "validation"
        / "best_pred.pkl"
    )
    if fusion_path.exists():
        probabilities, score_format = scores_to_probabilities(load_scores(fusion_path))
        return probabilities, {
            "source": "existing_e1_validation_fusion",
            "path": str(fusion_path),
            "score_format": score_format,
        }

    stream_probs = []
    source_paths = []
    score_formats = []
    for stream in STREAMS:
        stream_path = (
            e1_work_root
            / f"fold_{fold}"
            / stream
            / CONDITION_DIR
            / "validation"
            / "best_pred.pkl"
        )
        if not stream_path.exists():
            raise FileNotFoundError(
                f"Missing deterministic E1 validation prediction: {stream_path}"
            )
        probabilities, score_format = scores_to_probabilities(load_scores(stream_path))
        stream_probs.append(probabilities)
        source_paths.append(str(stream_path))
        score_formats.append(score_format)

    if stream_probs[0].shape != stream_probs[1].shape:
        raise ValueError(f"Joint/bone deterministic shapes differ for fold {fold}")
    probabilities = 0.5 * (stream_probs[0] + stream_probs[1])
    validate_probabilities(probabilities, "deterministic fusion probabilities")
    return probabilities, {
        "source": "computed_from_e1_validation_streams",
        "paths": source_paths,
        "score_formats": score_formats,
    }


def e1_metric_reference(args: argparse.Namespace, fold: str) -> dict[str, Any] | None:
    eval_path = (
        args.e1_work_root
        / f"fold_{fold}"
        / "fusion"
        / CONDITION_DIR
        / "validation"
        / "best_eval.json"
    )
    if eval_path.exists():
        metrics = load_json(eval_path)
        return {
            "source": str(eval_path),
            "top1_acc": float(metrics["top1_acc"]),
            "macro_f1": float(metrics["macro_f1"]),
        }

    if args.e1_validation_fold_metrics.exists():
        with args.e1_validation_fold_metrics.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if (
                    row.get("condition") == CONDITION_KEY
                    and row.get("fold") == fold
                    and row.get("stream") == "fusion"
                    and row.get("split", "val") == "val"
                ):
                    return {
                        "source": str(args.e1_validation_fold_metrics),
                        "top1_acc": float(row["top1_acc"]),
                        "macro_f1": float(row["macro_f1"]),
                    }

    return None


def assert_e1_metric_alignment(
    args: argparse.Namespace,
    fold: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if args.skip_e1_alignment_check:
        return {
            "checked": False,
            "reason": "skipped_by_user",
        }

    reference = e1_metric_reference(args, fold)
    if reference is None:
        raise FileNotFoundError(
            "Could not find E1 validation fusion metrics for alignment check. "
            "Expected either "
            f"{args.e1_work_root / f'fold_{fold}' / 'fusion' / CONDITION_DIR / 'validation' / 'best_eval.json'} "
            f"or {args.e1_validation_fold_metrics}. "
            "Use --skip-e1-alignment-check only if you intentionally want to bypass this check."
        )

    top1_diff = abs(float(metrics["center_accuracy"]) - float(reference["top1_acc"]))
    macro_diff = abs(float(metrics["center_macro_f1"]) - float(reference["macro_f1"]))
    if top1_diff > args.e1_alignment_atol or macro_diff > args.e1_alignment_atol:
        raise RuntimeError(
            "E2 deterministic metric definitions do not align with E1 validation metrics "
            f"for fold {fold}: center_accuracy/top1 diff={top1_diff:.3g}, "
            f"center_macro_f1/macro_f1 diff={macro_diff:.3g}."
        )

    return {
        "checked": True,
        "source": reference["source"],
        "e1_top1_acc": reference["top1_acc"],
        "e1_macro_f1": reference["macro_f1"],
        "top1_abs_diff": top1_diff,
        "macro_f1_abs_diff": macro_diff,
        "atol": float(args.e1_alignment_atol),
    }


def per_class_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    confusion = np.bincount(
        num_classes * labels.astype(np.int64) + predictions.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).astype(np.float64)
    tp = np.diag(confusion)
    predicted = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom != 0)


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> float:
    picked = probabilities[np.arange(labels.shape[0]), labels.astype(np.int64)]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> float:
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == labels).astype(np.float64)
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
        ece += count / len(labels) * abs(float(np.mean(correct[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def predictive_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ece_bins: int,
) -> dict[str, Any]:
    """Compute E2A center metrics aligned with E1 top-1 and macro-F1."""

    validate_probabilities(probabilities, "predictive probabilities")
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    if len(predictions) != len(labels):
        raise ValueError(f"Prediction/label length mismatch: {len(predictions)} vs {len(labels)}")

    f1 = per_class_f1(predictions, labels, len(LABELS))
    return {
        "num_samples": int(len(labels)),
        "center_accuracy": float(np.mean(predictions == labels)) if len(labels) else 0.0,
        "center_macro_f1": float(np.mean(f1)),
        "state_macro_f1": float(np.mean(f1[STATE_CLASS_IDS])),
        "transition_macro_f1": float(np.mean(f1[TRANSITION_CLASS_IDS])),
        "raw_nll": negative_log_likelihood(probabilities, labels),
        "raw_brier": brier_score(probabilities, labels),
        "raw_ece": expected_calibration_error(probabilities, labels, num_bins=ece_bins),
        "error_count": int(np.count_nonzero(predictions != labels)),
    }


def categorical_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0)
    return -np.sum(values * np.log(values), axis=-1)


def mc_quantities(fused_passes: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(fused_passes, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected MC probabilities [K, N, C], got {values.shape}")
    validate_probabilities(values, "fused MC probability passes")
    mean_probabilities = values.mean(axis=0)
    validate_probabilities(mean_probabilities, "MC predictive mean")
    predictive_entropy = categorical_entropy(mean_probabilities)
    expected_entropy = categorical_entropy(values).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    if not np.all(np.isfinite(mutual_information)):
        raise ValueError("MC mutual information contains NaN or Inf")
    if np.any(mutual_information < -1e-8):
        raise ValueError("MC mutual information contains negative values")
    upper = np.log(len(LABELS)) + 1e-6
    if np.any(mutual_information > upper):
        raise ValueError("MC mutual information exceeds log(num_classes)")
    return {
        "mean_probabilities": mean_probabilities.astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "expected_entropy": expected_entropy.astype(np.float32),
        "mutual_information": mutual_information.astype(np.float32),
        "prediction": np.argmax(mean_probabilities, axis=1).astype(np.int64),
    }


def build_logit_wrapper(recognizer: Any):
    import torch
    import torch.nn as nn

    class PysklGCNLogitWrapper(nn.Module):
        """Wrap a PYSKL RecognizerGCN and return raw classifier logits."""

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            core = model.module if hasattr(model, "module") else model
            self.backbone = core.backbone
            self.cls_head = core.cls_head

        def forward(self, keypoint: torch.Tensor) -> torch.Tensor:
            if keypoint.ndim == 6:
                if keypoint.shape[1] != 1:
                    raise ValueError(
                        "Uncertainty inference expects exactly one test clip "
                        f"per window, but received {keypoint.shape[1]}."
                    )
                keypoint = keypoint[:, 0]

            if keypoint.ndim != 5:
                raise ValueError(
                    "Expected [B, M, T, V, C] after removing the clip "
                    f"dimension, but received {tuple(keypoint.shape)}."
                )

            features = self.backbone(keypoint)
            logits = self.cls_head(features)
            if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                raise RuntimeError(
                    f"Expected logits [B, {len(LABELS)}], received {tuple(logits.shape)}."
                )
            return logits

    return PysklGCNLogitWrapper(recognizer)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    keypoints = []
    labels = []
    for item in batch:
        keypoint = item["keypoint"]
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint)
        keypoints.append(keypoint.float())
        labels.append(int(torch.as_tensor(item["label"]).reshape(-1)[0].item()))
    return {
        "keypoint": torch.stack(keypoints, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def build_loader(dataset: Any, batch_size: int, num_workers: int, pin_memory: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_batch,
    )


def load_model_and_dataset(config_path: Path, checkpoint_path: Path):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from pyskl.datasets import build_dataset
    from pyskl.models import build_model
    from pyskl.utils.mc_dropout import verify_gcn_head_dropout

    cfg = Config.fromfile(str(config_path))
    cfg.data.test.test_mode = True
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    recognizer = build_model(cfg.model)
    load_checkpoint(recognizer, str(checkpoint_path), map_location="cpu")
    verification = verify_gcn_head_dropout(
        recognizer,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )
    return cfg, dataset, build_logit_wrapper(recognizer), verification


def batch_probabilities(model: Any, batch: dict[str, Any], device: Any) -> Any:
    import torch

    keypoint = batch["keypoint"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=(device.type == "cuda"),
    )
    logits = model(keypoint)
    return torch.softmax(logits, dim=-1)


def run_mc_sanity_checks(
    model: Any,
    loader: Any,
    device: Any,
    atol: float,
) -> dict[str, Any]:
    import torch
    from pyskl.utils.mc_dropout import enable_head_mc_dropout

    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Validation loader is empty") from exc

    model.to(device)
    model.eval()
    with torch.no_grad():
        deterministic_1 = batch_probabilities(model, first_batch, device)
        deterministic_2 = batch_probabilities(model, first_batch, device)
    deterministic_diff = float(torch.max(torch.abs(deterministic_1 - deterministic_2)).item())
    if not torch.allclose(deterministic_1, deterministic_2, atol=atol, rtol=0.0):
        raise RuntimeError(
            "Deterministic sanity check failed: repeated eval-mode outputs differ "
            f"with max abs diff {deterministic_diff:.3g}."
        )

    active_dropout = enable_head_mc_dropout(
        model,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )
    with torch.no_grad():
        stochastic_1 = batch_probabilities(model, first_batch, device)
        stochastic_2 = batch_probabilities(model, first_batch, device)
    stochastic_diff = float(torch.mean(torch.abs(stochastic_1 - stochastic_2)).item())
    if torch.allclose(stochastic_1, stochastic_2, atol=atol, rtol=0.0):
        raise RuntimeError(
            "MC-dropout sanity check failed: two stochastic head-dropout passes "
            "were identical within tolerance."
        )

    model.eval()
    return {
        "deterministic_max_abs_diff": deterministic_diff,
        "stochastic_mean_abs_diff": stochastic_diff,
        "atol": float(atol),
        "active_dropout": active_dropout,
    }


def collect_mc_probabilities(
    model: Any,
    loader: Any,
    device: Any,
    num_passes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    from pyskl.utils.mc_dropout import enable_head_mc_dropout

    if num_passes < 2:
        raise ValueError("MC dropout requires at least two passes.")

    model.to(device)
    enable_head_mc_dropout(
        model,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )

    all_passes = []
    reference_labels = None
    first_pass_seconds = 0.0

    for pass_index in range(num_passes):
        started = time.time()
        pass_probabilities = []
        pass_labels = []

        for batch in loader:
            with torch.no_grad():
                probabilities = batch_probabilities(model, batch, device)
            labels = batch["label"].reshape(-1).to(dtype=torch.long)
            pass_probabilities.append(probabilities.cpu())
            pass_labels.append(labels.cpu())

        current_probabilities = torch.cat(pass_probabilities, dim=0)
        current_labels = torch.cat(pass_labels, dim=0)
        if reference_labels is None:
            reference_labels = current_labels
        elif not torch.equal(reference_labels, current_labels):
            raise RuntimeError("Sample ordering changed between MC passes.")

        if pass_index == 0:
            first_pass_seconds = float(time.time() - started)
        all_passes.append(current_probabilities)

    model.eval()
    stacked = torch.stack(all_passes, dim=0).numpy().astype(np.float32, copy=False)
    labels = reference_labels.numpy().astype(np.int64, copy=False)
    validate_probabilities(stacked, "collected MC probabilities")
    return stacked, labels, first_pass_seconds


def load_existing_stream_result(
    out_dir: Path,
    expected_num_passes: int,
) -> StreamResult:
    prob_path = out_dir / "mc_prob_passes.npy"
    label_path = out_dir / "labels.npy"
    sample_id_path = out_dir / "sample_ids.json"
    probabilities = np.load(prob_path)
    labels = np.load(label_path)
    sample_ids = json.loads(sample_id_path.read_text(encoding="utf-8"))
    if probabilities.shape[0] != expected_num_passes:
        raise ValueError(
            f"{prob_path} has {probabilities.shape[0]} passes, expected {expected_num_passes}. "
            "Use --overwrite to regenerate."
        )
    validate_probabilities(probabilities, f"existing {prob_path}")
    return StreamResult(
        probabilities=probabilities,
        labels=labels.astype(np.int64, copy=False),
        sample_ids=sample_ids,
        output_dir=out_dir,
    )


def run_stream(
    args: argparse.Namespace,
    fold: str,
    stream: str,
    device: Any,
) -> StreamResult:
    import torch

    out_dir = stream_output_dir(args.output_root, fold, stream)
    expected_files = [
        out_dir / "mc_prob_passes.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
        out_dir / "metrics.json",
    ]
    if all(path.exists() for path in expected_files) and not args.overwrite:
        print(f"[SKIP] fold={fold} stream={stream}: MC outputs exist")
        return load_existing_stream_result(out_dir, args.num_passes)

    labels_from_pkl, sample_ids = validation_labels_and_ids(args.data_root, fold)
    config_path = validation_config_path(args.config_root, fold, stream)
    checkpoint_path = find_selected_checkpoint(args.e1_work_root, fold, stream)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    print(
        f"[INFO] E2A MC fold={fold} stream={stream} "
        f"checkpoint={checkpoint_path}"
    )
    cfg, dataset, model, verification = load_model_and_dataset(config_path, checkpoint_path)
    if len(dataset) != len(labels_from_pkl):
        raise ValueError(
            f"{config_path} dataset length {len(dataset)} does not match pkl val "
            f"length {len(labels_from_pkl)}"
        )

    loader = build_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    sanity = run_mc_sanity_checks(model, loader, device, args.sanity_atol)
    started = time.time()
    probabilities, collected_labels, first_pass_seconds = collect_mc_probabilities(
        model=model,
        loader=loader,
        device=device,
        num_passes=args.num_passes,
    )
    seconds = float(time.time() - started)

    if not np.array_equal(collected_labels, labels_from_pkl):
        raise RuntimeError(f"Collected labels do not match pkl validation order for {fold}/{stream}")

    mean_probabilities = probabilities.mean(axis=0)
    metrics = predictive_metrics(mean_probabilities, labels_from_pkl, args.ece_bins)
    metrics.update(
        {
            "branch": "mc_dropout_stream_mean",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": stream,
            "split": "val",
            "num_passes": int(args.num_passes),
            "seconds": seconds,
            "first_pass_seconds": first_pass_seconds,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mc_prob_passes.npy", probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "mc_mean_probabilities.npy", mean_probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels_from_pkl.astype(np.int64, copy=False))
    save_pickle(
        out_dir / "mc_mean_pred.pkl",
        mean_probabilities.astype(np.float32, copy=False).tolist(),
    )
    write_json(out_dir / "sample_ids.json", sample_ids)
    write_json(out_dir / "dropout_verification.json", verification)
    write_json(out_dir / "mc_sanity.json", sanity)
    write_json(
        out_dir / "metadata.json",
        {
            "protocol": "E2A MC-dropout stream inference",
            "condition": CONDITION_KEY,
            "split": "val",
            "stream": stream,
            "fold": fold,
            "seed_set_once_by_script": int(args.seed),
            "num_passes": int(args.num_passes),
            "no_temperature_scaling": True,
            "model_config_work_dir": str(getattr(cfg, "work_dir", "")),
            "output_shape": list(probabilities.shape),
        },
    )
    write_json(out_dir / "metrics.json", metrics)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[DONE] fold={fold} stream={stream} MC mean "
        f"acc={metrics['center_accuracy']:.4f} "
        f"macro_f1={metrics['center_macro_f1']:.4f} "
        f"nll={metrics['raw_nll']:.4f}"
    )
    return StreamResult(
        probabilities=probabilities,
        labels=labels_from_pkl,
        sample_ids=sample_ids,
        output_dir=out_dir,
    )


def summarize_fusion_for_fold(
    args: argparse.Namespace,
    fold: str,
    joint: StreamResult,
    bone: StreamResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels, sample_ids = validation_labels_and_ids(args.data_root, fold)
    if not np.array_equal(joint.labels, labels) or not np.array_equal(bone.labels, labels):
        raise ValueError(f"Joint/bone labels do not match pkl validation labels for fold {fold}")
    if joint.sample_ids != sample_ids or bone.sample_ids != sample_ids:
        raise ValueError(f"Joint/bone sample ids do not match pkl validation order for fold {fold}")
    if joint.probabilities.shape != bone.probabilities.shape:
        raise ValueError(f"Joint/bone MC probability shapes differ for fold {fold}")

    deterministic_probs, deterministic_source = deterministic_fusion_probabilities(
        args.e1_work_root,
        fold,
    )
    if deterministic_probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Deterministic validation predictions for fold {fold} have "
            f"{deterministic_probs.shape[0]} rows, expected {labels.shape[0]}"
        )

    fused_passes = 0.5 * (joint.probabilities + bone.probabilities)
    validate_probabilities(fused_passes, f"fold {fold} fused MC passes")
    quantities = mc_quantities(fused_passes)
    mean_probabilities = quantities["mean_probabilities"]
    errors = quantities["prediction"].astype(np.int64) != labels

    fusion_dir = fusion_output_dir(args.output_root, fold)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    np.save(fusion_dir / "mc_prob_passes.npy", fused_passes.astype(np.float32, copy=False))
    np.save(fusion_dir / "mc_mean_probabilities.npy", mean_probabilities.astype(np.float32, copy=False))
    np.save(fusion_dir / "labels.npy", labels.astype(np.int64, copy=False))
    np.savez_compressed(
        fusion_dir / "mc_quantities.npz",
        mean_probabilities=mean_probabilities.astype(np.float32, copy=False),
        predictive_entropy=quantities["predictive_entropy"].astype(np.float32, copy=False),
        expected_entropy=quantities["expected_entropy"].astype(np.float32, copy=False),
        mutual_information=quantities["mutual_information"].astype(np.float32, copy=False),
        prediction=quantities["prediction"].astype(np.int64, copy=False),
        label=labels.astype(np.int64, copy=False),
        error=errors.astype(np.bool_, copy=False),
    )
    save_pickle(
        fusion_dir / "mc_mean_pred.pkl",
        mean_probabilities.astype(np.float32, copy=False).tolist(),
    )
    write_json(fusion_dir / "sample_ids.json", sample_ids)

    mc_metrics = predictive_metrics(mean_probabilities, labels, args.ece_bins)
    mc_metrics.update(
        {
            "branch": "mc_dropout",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": "fusion",
            "split": "val",
            "num_passes": int(args.num_passes),
            "probability_rule": "p_MC^(k) = 0.5 * (p_joint^(k) + p_bone^(k)); predictive mean over K passes",
            "source_path": str(fusion_dir / "mc_quantities.npz"),
            "mean_mutual_information": float(np.mean(quantities["mutual_information"])),
            "max_mutual_information": float(np.max(quantities["mutual_information"])),
            "mean_predictive_entropy": float(np.mean(quantities["predictive_entropy"])),
            "mean_expected_entropy": float(np.mean(quantities["expected_entropy"])),
        }
    )
    write_json(fusion_dir / "metrics.json", mc_metrics)

    deterministic_metrics = predictive_metrics(deterministic_probs, labels, args.ece_bins)
    deterministic_metrics.update(
        {
            "branch": "deterministic",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": "fusion",
            "split": "val",
            "num_passes": 1,
            "probability_rule": "p_det = 0.5 * (p_joint + p_bone)",
            "source": deterministic_source["source"],
            "source_path": deterministic_source.get(
                "path",
                ";".join(deterministic_source.get("paths", [])),
            ),
            "source_score_format": deterministic_source.get(
                "score_format",
                ";".join(deterministic_source.get("score_formats", [])),
            ),
        }
    )
    alignment = assert_e1_metric_alignment(args, fold, deterministic_metrics)
    deterministic_metrics.update(
        {
            "e1_alignment_checked": alignment["checked"],
            "e1_alignment_source": alignment.get("source", ""),
            "e1_top1_acc": alignment.get("e1_top1_acc", ""),
            "e1_macro_f1": alignment.get("e1_macro_f1", ""),
            "e1_top1_abs_diff": alignment.get("top1_abs_diff", ""),
            "e1_macro_f1_abs_diff": alignment.get("macro_f1_abs_diff", ""),
        }
    )
    return deterministic_metrics, mc_metrics


def aggregate_mean_sd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["branch"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    metrics = [
        "center_accuracy",
        "center_macro_f1",
        "state_macro_f1",
        "transition_macro_f1",
        "raw_nll",
        "raw_brier",
        "raw_ece",
    ]
    for branch in sorted(grouped):
        branch_rows = sorted(grouped[branch], key=lambda row: str(row["fold"]))
        summary: dict[str, Any] = {
            "branch": branch,
            "condition": CONDITION_KEY,
            "stream": "fusion",
            "split": "val",
            "folds": len(branch_rows),
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in branch_rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.4f} +- {sd:.4f}"


def markdown_report(fold_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E2A Raw Predictive Branch Metrics",
        "",
        "Split: validation. Protocol: E1-B continuous windows. No temperature",
        "scaling is applied. MC dropout uses 10 stochastic passes unless the",
        "command line overrides `--num-passes`.",
        "",
        "## Mean +- SD Across Folds",
        "",
        "| Branch | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {branch} | {acc} | {macro} | {state} | {transition} | {nll} | {brier} | {ece} |".format(
                branch=row["branch"],
                acc=format_mean_sd(row["center_accuracy_mean"], row["center_accuracy_sd"]),
                macro=format_mean_sd(row["center_macro_f1_mean"], row["center_macro_f1_sd"]),
                state=format_mean_sd(row["state_macro_f1_mean"], row["state_macro_f1_sd"]),
                transition=format_mean_sd(row["transition_macro_f1_mean"], row["transition_macro_f1_sd"]),
                nll=format_mean_sd(row["raw_nll_mean"], row["raw_nll_sd"]),
                brier=format_mean_sd(row["raw_brier_mean"], row["raw_brier_sd"]),
                ece=format_mean_sd(row["raw_ece_mean"], row["raw_ece_sd"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| Branch | Fold | N | Acc | Macro F1 | State Macro F1 | Transition Macro F1 | NLL | Brier | ECE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(fold_rows, key=lambda item: (str(item["branch"]), str(item["fold"]))):
        lines.append(
            "| {branch} | {fold} | {n} | {acc:.4f} | {macro:.4f} | {state:.4f} | "
            "{transition:.4f} | {nll:.4f} | {brier:.4f} | {ece:.4f} |".format(
                branch=row["branch"],
                fold=str(row["fold"]).upper(),
                n=int(row["num_samples"]),
                acc=float(row["center_accuracy"]),
                macro=float(row["center_macro_f1"]),
                state=float(row["state_macro_f1"]),
                transition=float(row["transition_macro_f1"]),
                nll=float(row["raw_nll"]),
                brier=float(row["raw_brier"]),
                ece=float(row["raw_ece"]),
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        import mmcv  # noqa: F401
        import pyskl  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "E2A MC-dropout inference requires torch, mmcv, and pyskl in the "
            "active Python environment."
        ) from exc

    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.ece_bins <= 0:
        raise ValueError("--ece-bins must be positive")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    set_global_seed(args.seed, deterministic_cudnn=args.deterministic_cudnn)
    device = resolve_device(args.device)
    print(f"[INFO] device={device} seed={args.seed} num_passes={args.num_passes}")

    fold_rows: list[dict[str, Any]] = []
    for fold in [item.lower() for item in args.folds]:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")

        stream_results: dict[str, StreamResult] = {}
        for stream in STREAMS:
            stream_results[stream] = run_stream(args, fold, stream, device)

        deterministic_row, mc_row = summarize_fusion_for_fold(
            args,
            fold,
            stream_results["joint"],
            stream_results["bone"],
        )
        fold_rows.extend([deterministic_row, mc_row])
        print(
            f"[DONE] fold={fold} fusion MC "
            f"acc={mc_row['center_accuracy']:.4f} "
            f"macro_f1={mc_row['center_macro_f1']:.4f} "
            f"nll={mc_row['raw_nll']:.4f} "
            f"ece={mc_row['raw_ece']:.4f}"
        )

    summary_rows = aggregate_mean_sd(fold_rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "e2a_raw_predictive_fold_metrics.csv", fold_rows)
    write_csv(args.report_dir / "e2a_raw_predictive_mean_sd.csv", summary_rows)
    write_json(
        args.report_dir / "e2a_mc_dropout_summary.json",
        {
            "experiment": "E2A raw predictive-branch comparison",
            "implemented_branch": "mc_dropout",
            "included_branches": sorted({str(row["branch"]) for row in fold_rows}),
            "split": "val",
            "condition": CONDITION_KEY,
            "checkpoint_source": "E1-B selected by validation center macro-F1",
            "seed": int(args.seed),
            "num_passes": int(args.num_passes),
            "ece_bins": int(args.ece_bins),
            "no_temperature_scaling": True,
            "state_class_ids": STATE_CLASS_IDS,
            "transition_class_ids": TRANSITION_CLASS_IDS,
            "state_classes": [LABELS[index] for index in STATE_CLASS_IDS],
            "transition_classes": [LABELS[index] for index in TRANSITION_CLASS_IDS],
            "labels": LABELS,
            "metric_definitions": {
                "center_accuracy": "argmax predictive probability equals center-frame hard label; aligned with E1 top1_acc.",
                "center_macro_f1": "unweighted mean of per-class F1 over all nine final classes; aligned with E1 macro_f1.",
                "state_macro_f1": "unweighted mean over lie-stationary, sit-stationary, and walk.",
                "transition_macro_f1": "unweighted mean over fall and the five transition-* classes.",
            },
            "fold_metrics": fold_rows,
            "mean_sd": summary_rows,
        },
    )
    (args.report_dir / "e2a_mc_dropout_summary.md").write_text(
        markdown_report(fold_rows, summary_rows),
        encoding="utf-8",
        newline="\n",
    )
    print(f"[DONE] wrote E2A MC-dropout reports under {args.report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/stgcn++/stgcn++_radarv4/rerun/e1"),
        help="Root containing E1 generated fold/stream validation configs.",
    )
    parser.add_argument(
        "--e1-work-root",
        type=Path,
        default=Path("work_dirs/rerun/e1"),
        help="Root containing selected E1-B checkpoints and deterministic validation predictions.",
    )
    parser.add_argument(
        "--e1-validation-fold-metrics",
        type=Path,
        default=Path("rerun/e1/reports/e1_validation_fold_metrics.csv"),
        help="Optional E1 validation summary CSV used to cross-check metric alignment.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/radar_v4/rerun/yolo26xpose/pyskl"),
        help="Root containing the continuous-window PYSKL pkls.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work_dirs/rerun/e2/e2a_mc_dropout"),
        help="Root where MC-dropout arrays and per-fold artifacts are written.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("rerun/e2/reports"),
        help="Directory where E2A CSV/JSON/Markdown reports are written.",
    )
    parser.add_argument("--num-passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Torch CPU thread count. 0 keeps PyTorch's default.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of bins for top-label ECE. Default follows existing repo convention.",
    )
    parser.add_argument(
        "--sanity-atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance for deterministic/stochastic MC sanity checks.",
    )
    parser.add_argument(
        "--deterministic-cudnn",
        action="store_true",
        help="Request deterministic CuDNN kernels where available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing stream MC outputs.",
    )
    parser.add_argument(
        "--skip-e1-alignment-check",
        action="store_true",
        help="Skip the deterministic E2-vs-E1 metric alignment check.",
    )
    parser.add_argument(
        "--e1-alignment-atol",
        type=float,
        default=1e-10,
        help="Tolerance for center_accuracy/top1_acc and center_macro_f1/macro_f1 alignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
