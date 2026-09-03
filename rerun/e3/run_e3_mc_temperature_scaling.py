"""E3 pool-then-calibrate temperature scaling for the selected MC branch.

The selected E2 branch is MC dropout. E3 compares the raw MC predictive mean
against a temperature-calibrated MC predictive mean.

Data flow per fold:
1. generate 30 MC dropout probability passes for calibration and test splits;
2. fuse Joint/Bone probabilities within each pass by equal averaging;
3. average fused passes to get the raw MC predictive mean;
4. fit one fold-specific scalar temperature on the calibration subject;
5. freeze that temperature and evaluate raw versus calibrated probabilities
   on the outer test subject.

Temperature scaling is pool-then-calibrate:

    s_i,c = log(max(pbar_i,c, eps))
    q_i(T) = softmax(s_i / T)

No temperature is applied before Joint/Bone fusion or before MC-pass averaging.
"""

from __future__ import annotations

import argparse
import copy
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
FOLDS = ["a", "b", "c"]
STREAMS = ["joint", "bone"]
SPLITS = ["calib", "test"]
CONDITION_KEY = "b"
CONDITION_DIR = "b_continuous_window"
RANDOM_TRANSFORM_NAMES = {
    "Flip",
    "RandomResizedCrop",
    "RandomCrop",
    "RandomScale",
    "RandomRescale",
    "RandomRotation",
    "RandomRot",
    "RandomGaussianNoise",
    "UniformSample",
    "UniformSampleFrames",
}


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
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def base_config_path(config_root: Path, fold: str, stream: str) -> Path:
    return config_root / f"fold_{fold}" / stream / "b_continuous_window.py"


def find_selected_checkpoint(e1_work_root: Path, fold: str, stream: str) -> Path:
    checkpoint_dir = e1_work_root / f"fold_{fold}" / stream / CONDITION_DIR
    checkpoints = sorted(checkpoint_dir.glob("best_macro_f1_epoch_*.pth"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected one best_macro_f1 checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def stream_output_dir(output_root: Path, fold: str, stream: str, split: str) -> Path:
    return output_root / f"fold_{fold}" / stream / CONDITION_DIR / split


def fusion_output_dir(output_root: Path, fold: str, split: str) -> Path:
    return output_root / f"fold_{fold}" / "fusion" / CONDITION_DIR / split


def temperature_output_dir(output_root: Path, fold: str) -> Path:
    return output_root / f"fold_{fold}" / "temperature_scaling"


def sample_id_from_annotation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_id": str(item.get("subject", "")),
        "recording_id": str(item.get("session_name", item.get("session", ""))),
        "window_row_start": int(item["window_row_start"]),
        "center_source_frame": int(item["center_source_frame"]),
    }


def sample_ids_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    return [sample_id_from_annotation(item) for item in dataset.video_infos]


def labels_from_dataset(dataset: Any) -> np.ndarray:
    return np.array([int(item["label"]) for item in dataset.video_infos], dtype=np.int64)


def assert_unique_sample_ids(sample_ids: list[dict[str, Any]], name: str) -> None:
    keys = [
        (
            item["subject_id"],
            item["recording_id"],
            item["window_row_start"],
            item["center_source_frame"],
        )
        for item in sample_ids
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains {len(keys) - len(set(keys))} duplicate sample IDs")


def validate_probabilities(values: np.ndarray, name: str, atol: float = 1e-5) -> None:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape[-1] != len(LABELS):
        raise ValueError(f"{name} expected {len(LABELS)} classes, got {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError(f"{name} contains values outside [0, 1]")
    sums = probabilities.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=atol, rtol=0.0):
        max_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name} rows do not sum to 1; max error={max_error}")


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


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    picked = values[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    one_hot = np.zeros_like(values, dtype=np.float64)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def top_label_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    confidences = np.max(values, axis=1)
    predictions = np.argmax(values, axis=1)
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
        ece += count / len(targets) * abs(float(np.mean(correct[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ece_bins: int,
) -> dict[str, Any]:
    validate_probabilities(probabilities, "classification probabilities")
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    f1 = per_class_f1(predictions, targets, len(LABELS))
    return {
        "num_samples": int(targets.shape[0]),
        "center_accuracy": float(np.mean(predictions == targets)),
        "center_macro_f1": float(np.mean(f1)),
        "nll": multiclass_nll(probabilities, targets),
        "brier": multiclass_brier(probabilities, targets),
        "ece": top_label_ece(probabilities, targets, num_bins=ece_bins),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "error_count": int(np.count_nonzero(predictions != targets)),
    }


def softmax_np(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def apply_pool_temperature(
    probabilities: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError(f"Temperature must be finite and positive, got {temperature}")
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, eps, 1.0)
    values = values / values.sum(axis=-1, keepdims=True)
    calibrated = softmax_np(np.log(values) / float(temperature))
    validate_probabilities(calibrated, "temperature-calibrated probabilities")
    return calibrated


def fit_pool_temperature_torch(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    eps: float,
    max_iter: int,
    lbfgs_lr: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    probabilities = torch.as_tensor(calibration_probabilities, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(calibration_labels, dtype=torch.long, device="cpu").reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError("Calibration probabilities must have shape [N, C].")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Calibration probability and label counts do not match.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Calibration probabilities contain NaN or Inf.")
    if (probabilities < 0).any():
        raise ValueError("Calibration probabilities must be nonnegative.")

    probabilities = probabilities.clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    log_probabilities = probabilities.log()
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=lbfgs_lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        calibrated_log_probabilities = F.log_softmax(log_probabilities / temperature, dim=-1)
        loss = F.nll_loss(calibrated_log_probabilities, labels, reduction="mean")
        loss.backward()
        return loss

    with torch.no_grad():
        before = float(F.nll_loss(log_probabilities, labels, reduction="mean").item())
    started = time.time()
    optimizer.step(closure)
    seconds = float(time.time() - started)
    temperature = float(log_temperature.detach().exp().item())
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"Fitted temperature is not finite and positive: {temperature}")
    calibrated = apply_pool_temperature(calibration_probabilities, temperature, eps=eps)
    after = multiclass_nll(calibrated, calibration_labels, eps=eps)
    if after > before + 1e-8:
        raise RuntimeError(
            f"Calibration NLL increased after fitting: raw={before:.12g}, calibrated={after:.12g}"
        )
    return {
        "temperature": temperature,
        "optimizer": "torch_lbfgs_log_temperature",
        "max_iter": int(max_iter),
        "lbfgs_lr": float(lbfgs_lr),
        "fit_seconds": seconds,
        "calibration_nll_before": before,
        "calibration_nll_after": after,
    }


def fit_pool_temperature_numpy_golden(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    eps: float,
    max_iter: int,
    min_temperature: float,
    max_temperature: float,
) -> dict[str, Any]:
    """Optional dependency-free scalar optimizer for local report checks."""

    import math

    if min_temperature <= 0 or max_temperature <= min_temperature:
        raise ValueError("Temperature bounds must satisfy 0 < min < max")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        calibrated = apply_pool_temperature(calibration_probabilities, temperature, eps=eps)
        return multiclass_nll(calibrated, calibration_labels, eps=eps)

    before = objective(0.0)
    lower = math.log(float(min_temperature))
    upper = math.log(float(max_temperature))
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0
    left = lower + inv_phi_sq * (upper - lower)
    right = lower + inv_phi * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    started = time.time()
    for _ in range(max_iter):
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
    temperature = math.exp((lower + upper) / 2.0)
    calibrated = apply_pool_temperature(calibration_probabilities, temperature, eps=eps)
    after = multiclass_nll(calibrated, calibration_labels, eps=eps)
    if after > before + 1e-8:
        raise RuntimeError(
            f"Calibration NLL increased after fitting: raw={before:.12g}, calibrated={after:.12g}"
        )
    return {
        "temperature": float(temperature),
        "optimizer": "numpy_golden_section_log_temperature",
        "max_iter": int(max_iter),
        "temperature_bounds": [float(min_temperature), float(max_temperature)],
        "fit_seconds": float(time.time() - started),
        "calibration_nll_before": before,
        "calibration_nll_after": after,
    }


def fit_pool_temperature(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.temperature_optimizer == "torch_lbfgs":
        return fit_pool_temperature_torch(
            calibration_probabilities=calibration_probabilities,
            calibration_labels=calibration_labels,
            eps=args.eps,
            max_iter=args.temperature_max_iter,
            lbfgs_lr=args.temperature_lr,
        )
    return fit_pool_temperature_numpy_golden(
        calibration_probabilities=calibration_probabilities,
        calibration_labels=calibration_labels,
        eps=args.eps,
        max_iter=args.temperature_max_iter,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )


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
                        "MC dropout expects exactly one deterministic clip "
                        f"per window, but received {keypoint.shape[1]}."
                    )
                keypoint = keypoint[:, 0]
            if keypoint.ndim != 5:
                raise ValueError(f"Expected [B, M, T, V, C], got {tuple(keypoint.shape)}")
            logits = self.cls_head(self.backbone(keypoint))
            if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                raise RuntimeError(f"Expected logits [B, {len(LABELS)}], got {tuple(logits.shape)}")
            return logits

    return PysklGCNLogitWrapper(recognizer)


def unwrap_value(value: Any) -> Any:
    import torch

    if not torch.is_tensor(value) and hasattr(value, "data"):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    keypoints = []
    labels = []
    for item in items:
        keypoints.append(torch.as_tensor(unwrap_value(item["keypoint"]), dtype=torch.float32))
        labels.append(torch.as_tensor(unwrap_value(item["label"]), dtype=torch.long).reshape(-1)[0])
    return {
        "keypoint": torch.stack(keypoints, dim=0),
        "label": torch.stack(labels, dim=0).long(),
    }


def build_loader(dataset: Any, batch_size: int, num_workers: int, pin_memory: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )


def deterministic_dataset_cfg(cfg: Any, split: str) -> Any:
    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg.split = split
    dataset_cfg.test_mode = True
    for key in ["class_sample_strategy", "class_sample_power", "epoch_size", "class_prob"]:
        dataset_cfg.pop(key, None)
    return dataset_cfg


def describe_pipeline(dataset: Any) -> list[str]:
    transforms = getattr(getattr(dataset, "pipeline", None), "transforms", [])
    return [type(transform).__name__ for transform in transforms]


def assert_deterministic_pipeline(dataset: Any, name: str) -> list[str]:
    transform_names = describe_pipeline(dataset)
    random_names = [
        item
        for item in transform_names
        if item in RANDOM_TRANSFORM_NAMES or item.lower().startswith("random")
    ]
    if random_names:
        raise ValueError(f"{name} contains random transforms: {random_names}")
    return transform_names


def load_model_and_dataset(config_path: Path, checkpoint_path: Path, split: str):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from pyskl.datasets import build_dataset
    from pyskl.models import build_model
    from pyskl.utils.mc_dropout import verify_gcn_head_dropout

    cfg = Config.fromfile(str(config_path))
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None

    dataset = build_dataset(deterministic_dataset_cfg(cfg, split), dict(test_mode=True))
    pipeline = assert_deterministic_pipeline(dataset, f"E3 {split} dataset")
    assert_unique_sample_ids(sample_ids_from_dataset(dataset), f"E3 {split} dataset")

    recognizer = build_model(cfg.model)
    load_checkpoint(recognizer, str(checkpoint_path), map_location="cpu")
    verification = verify_gcn_head_dropout(
        recognizer,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )
    return cfg, dataset, build_logit_wrapper(recognizer), verification, pipeline


def batch_probabilities(model: Any, batch: dict[str, Any], device: Any) -> Any:
    import torch

    keypoint = batch["keypoint"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=(device.type == "cuda"),
    )
    logits = model(keypoint)
    return torch.softmax(logits, dim=-1)


def collect_mc_probabilities(model: Any, loader: Any, device: Any, num_passes: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from pyskl.utils.mc_dropout import enable_head_mc_dropout

    if num_passes < 2:
        raise ValueError("MC dropout requires at least two passes.")

    model.to(device)
    enable_head_mc_dropout(model, expected_dropout=0.5, expected_num_classes=len(LABELS))

    all_passes = []
    reference_labels = None
    for _ in range(num_passes):
        pass_probabilities = []
        pass_labels = []
        for batch in loader:
            with torch.no_grad():
                probabilities = batch_probabilities(model, batch, device)
            pass_probabilities.append(probabilities.cpu())
            pass_labels.append(batch["label"].reshape(-1).cpu())

        current_probabilities = torch.cat(pass_probabilities, dim=0)
        current_labels = torch.cat(pass_labels, dim=0)
        if reference_labels is None:
            reference_labels = current_labels
        elif not torch.equal(reference_labels, current_labels):
            raise RuntimeError("Sample ordering changed between MC passes.")
        all_passes.append(current_probabilities)

    model.eval()
    probabilities = torch.stack(all_passes, dim=0).numpy().astype(np.float32, copy=False)
    labels = reference_labels.numpy().astype(np.int64, copy=False)
    validate_probabilities(probabilities, "collected MC probabilities")
    return probabilities, labels


def load_existing_stream_result(out_dir: Path, expected_passes: int) -> StreamResult:
    probabilities = np.load(out_dir / "mc_prob_passes.npy")
    labels = np.load(out_dir / "labels.npy").astype(np.int64, copy=False)
    sample_ids = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    if probabilities.shape[0] != expected_passes:
        raise ValueError(
            f"{out_dir / 'mc_prob_passes.npy'} has {probabilities.shape[0]} passes, "
            f"expected {expected_passes}. Use --overwrite to regenerate."
        )
    validate_probabilities(probabilities, f"existing {out_dir} MC probabilities")
    return StreamResult(probabilities=probabilities, labels=labels, sample_ids=sample_ids, output_dir=out_dir)


def run_stream(args: argparse.Namespace, fold: str, stream: str, split: str, device: Any) -> StreamResult:
    import torch

    out_dir = stream_output_dir(args.output_root, fold, stream, split)
    expected_files = [
        out_dir / "mc_prob_passes.npy",
        out_dir / "mc_mean_probabilities.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
        out_dir / "metrics.json",
    ]
    if all(path.exists() for path in expected_files) and not args.overwrite:
        print(f"[SKIP] existing E3 MC stream outputs: {out_dir}")
        return load_existing_stream_result(out_dir, args.num_passes)

    config_path = base_config_path(args.config_root, fold, stream)
    checkpoint_path = find_selected_checkpoint(args.e1_work_root, fold, stream)
    for path in [config_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"[INFO] E3 MC fold={fold} stream={stream} split={split} checkpoint={checkpoint_path}")
    cfg, dataset, model, verification, pipeline = load_model_and_dataset(config_path, checkpoint_path, split)
    labels = labels_from_dataset(dataset)
    sample_ids = sample_ids_from_dataset(dataset)
    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    started = time.time()
    probabilities, collected_labels = collect_mc_probabilities(model, loader, device, args.num_passes)
    seconds = float(time.time() - started)
    if not np.array_equal(collected_labels, labels):
        raise RuntimeError(f"Collected labels do not match pkl {split} order for fold={fold} stream={stream}")

    mean_probabilities = probabilities.mean(axis=0)
    metrics = classification_metrics(mean_probabilities, labels, args.ece_bins)
    metrics.update(
        {
            "branch": "mc_dropout_stream_mean",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": stream,
            "split": split,
            "num_passes": int(args.num_passes),
            "seconds": seconds,
            "device": str(device),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "model_config_work_dir": str(getattr(cfg, "work_dir", "")),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mc_prob_passes.npy", probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "mc_mean_probabilities.npy", mean_probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels.astype(np.int64, copy=False))
    save_pickle(out_dir / "mc_mean_pred.pkl", mean_probabilities.astype(np.float32, copy=False).tolist())
    write_json(out_dir / "sample_ids.json", sample_ids)
    write_json(out_dir / "dropout_verification.json", verification)
    write_json(
        out_dir / "metadata.json",
        {
            "protocol": "E3 MC-dropout stream inference",
            "split": split,
            "fold": fold,
            "stream": stream,
            "num_passes": int(args.num_passes),
            "seed_set_once_by_script": int(args.seed),
            "pipeline": pipeline,
            "no_temperature_scaling": True,
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[DONE] fold={fold} stream={stream} split={split} "
        f"raw_nll={metrics['nll']:.4f} ece={metrics['ece']:.4f}"
    )
    return StreamResult(probabilities=probabilities, labels=labels, sample_ids=sample_ids, output_dir=out_dir)


def load_stream_result(args: argparse.Namespace, fold: str, stream: str, split: str) -> StreamResult:
    out_dir = stream_output_dir(args.output_root, fold, stream, split)
    required = [out_dir / "mc_prob_passes.npy", out_dir / "labels.npy", out_dir / "sample_ids.json"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing E3 stream output {path}. Run --mode stream first.")
    return load_existing_stream_result(out_dir, args.num_passes)


def summarize_fusion_for_split(args: argparse.Namespace, fold: str, split: str) -> dict[str, Any]:
    joint = load_stream_result(args, fold, "joint", split)
    bone = load_stream_result(args, fold, "bone", split)
    if joint.sample_ids != bone.sample_ids:
        raise ValueError(f"Joint/bone sample IDs differ for fold={fold} split={split}")
    if not np.array_equal(joint.labels, bone.labels):
        raise ValueError(f"Joint/bone labels differ for fold={fold} split={split}")
    if joint.probabilities.shape != bone.probabilities.shape:
        raise ValueError(f"Joint/bone probability shapes differ for fold={fold} split={split}")

    fused_passes = 0.5 * (joint.probabilities + bone.probabilities)
    validate_probabilities(fused_passes, f"fold={fold} split={split} fused MC passes")
    raw_mean = fused_passes.mean(axis=0).astype(np.float32, copy=False)
    labels = joint.labels
    raw_metrics = classification_metrics(raw_mean, labels, args.ece_bins)
    raw_metrics.update(
        {
            "branch": "mc_dropout",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": "fusion",
            "split": split,
            "num_passes": int(args.num_passes),
            "probability_rule": "p_MC^(k)=0.5*(p_joint^(k)+p_bone^(k)); pbar_MC=mean_k p_MC^(k)",
        }
    )

    out_dir = fusion_output_dir(args.output_root, fold, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mc_prob_passes.npy", fused_passes.astype(np.float32, copy=False))
    np.save(out_dir / "raw_mc_mean_probabilities.npy", raw_mean)
    np.save(out_dir / "labels.npy", labels.astype(np.int64, copy=False))
    save_pickle(out_dir / "raw_mc_mean_pred.pkl", raw_mean.tolist())
    write_json(out_dir / "sample_ids.json", joint.sample_ids)
    write_json(out_dir / "raw_metrics.json", raw_metrics)
    print(f"[DONE] fold={fold} split={split} fused raw MC nll={raw_metrics['nll']:.4f}")
    return raw_metrics


def load_fused_split(args: argparse.Namespace, fold: str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    out_dir = fusion_output_dir(args.output_root, fold, split)
    required = [
        out_dir / "raw_mc_mean_probabilities.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing fused E3 artifact {path}. Run --mode fusion first.")
    probabilities = np.load(out_dir / "raw_mc_mean_probabilities.npy").astype(np.float64, copy=False)
    labels = np.load(out_dir / "labels.npy").astype(np.int64, copy=False)
    sample_ids = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    validate_probabilities(probabilities, f"fold={fold} split={split} raw MC mean")
    if probabilities.shape[0] != labels.shape[0] or len(sample_ids) != labels.shape[0]:
        raise ValueError(f"fold={fold} split={split} fused artifact lengths do not match")
    return probabilities, labels, sample_ids


def compare_raw_calibrated_metrics(
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    labels: np.ndarray,
    ece_bins: int,
) -> dict[str, Any]:
    raw_pred = np.argmax(raw_probabilities, axis=1)
    cal_pred = np.argmax(calibrated_probabilities, axis=1)
    if not np.array_equal(raw_pred, cal_pred):
        raise RuntimeError("Pool-temperature scaling changed predicted classes.")
    raw_metrics = classification_metrics(raw_probabilities, labels, ece_bins)
    cal_metrics = classification_metrics(calibrated_probabilities, labels, ece_bins)
    return {
        "num_samples": int(labels.shape[0]),
        "raw_center_accuracy": raw_metrics["center_accuracy"],
        "calibrated_center_accuracy": cal_metrics["center_accuracy"],
        "raw_center_macro_f1": raw_metrics["center_macro_f1"],
        "calibrated_center_macro_f1": cal_metrics["center_macro_f1"],
        "raw_nll": raw_metrics["nll"],
        "calibrated_nll": cal_metrics["nll"],
        "delta_nll": cal_metrics["nll"] - raw_metrics["nll"],
        "raw_brier": raw_metrics["brier"],
        "calibrated_brier": cal_metrics["brier"],
        "delta_brier": cal_metrics["brier"] - raw_metrics["brier"],
        "raw_ece": raw_metrics["ece"],
        "calibrated_ece": cal_metrics["ece"],
        "delta_ece": cal_metrics["ece"] - raw_metrics["ece"],
        "raw_mean_confidence": raw_metrics["mean_confidence"],
        "calibrated_mean_confidence": cal_metrics["mean_confidence"],
        "argmax_preserved": True,
    }


def run_temperature_for_fold(args: argparse.Namespace, fold: str) -> tuple[dict[str, Any], dict[str, Any]]:
    calib_probs, calib_labels, calib_ids = load_fused_split(args, fold, "calib")
    test_probs, test_labels, test_ids = load_fused_split(args, fold, "test")

    fit = fit_pool_temperature(calib_probs, calib_labels, args)
    temperature = float(fit["temperature"])
    calib_calibrated = apply_pool_temperature(calib_probs, temperature, eps=args.eps)
    test_calibrated = apply_pool_temperature(test_probs, temperature, eps=args.eps)
    calibration_metrics = compare_raw_calibrated_metrics(
        calib_probs,
        calib_calibrated,
        calib_labels,
        args.ece_bins,
    )
    test_metrics = compare_raw_calibrated_metrics(
        test_probs,
        test_calibrated,
        test_labels,
        args.ece_bins,
    )

    if calibration_metrics["calibrated_nll"] > calibration_metrics["raw_nll"] + 1e-8:
        raise RuntimeError(
            f"fold={fold}: calibrated calibration NLL is higher than raw calibration NLL"
        )

    for split, calibrated, raw_probs, labels, sample_ids in [
        ("calib", calib_calibrated, calib_probs, calib_labels, calib_ids),
        ("test", test_calibrated, test_probs, test_labels, test_ids),
    ]:
        out_dir = fusion_output_dir(args.output_root, fold, split)
        np.save(out_dir / "temperature_calibrated_probabilities.npy", calibrated.astype(np.float32, copy=False))
        save_pickle(out_dir / "temperature_calibrated_pred.pkl", calibrated.astype(np.float32, copy=False).tolist())
        write_json(
            out_dir / "temperature_calibrated_metrics.json",
            {
                "fold": fold,
                "split": split,
                "temperature": temperature,
                "metrics": compare_raw_calibrated_metrics(raw_probs, calibrated, labels, args.ece_bins),
                "sample_ids_head": sample_ids[:5],
                "sample_ids_tail": sample_ids[-5:],
            },
        )

    temp_dir = temperature_output_dir(args.output_root, fold)
    write_json(
        temp_dir / "temperature.json",
        {
            "fold": fold,
            "temperature": temperature,
            "fit": fit,
            "calibration_subject_split": "calib",
            "test_subject_split": "test",
            "calibration_windows": int(calib_labels.shape[0]),
            "test_windows": int(test_labels.shape[0]),
            "eps": float(args.eps),
            "ece_bins": int(args.ece_bins),
            "ece_bin_type": "fixed_equal_width",
            "ece_bin_edge_convention": "[lower, upper) except final bin [lower, upper]",
            "brier_definition": "mean_over_samples(sum_over_classes((p-onehot)^2))",
            "divide_by_num_classes": False,
            "temperature_scaling_rule": "q=softmax(log(max(pbar_MC, eps))/T)",
            "argmax_preserved_on_calibration": bool(calibration_metrics["argmax_preserved"]),
            "argmax_preserved_on_test": bool(test_metrics["argmax_preserved"]),
            "no_test_decision": True,
        },
    )

    calibration_row = {
        "fold": fold,
        "split": "calib",
        "temperature": temperature,
        **calibration_metrics,
    }
    test_row = {
        "fold": fold,
        "split": "test",
        "temperature": temperature,
        **test_metrics,
    }
    print(
        f"[DONE] fold={fold} T={temperature:.6g} "
        f"calib_nll={calibration_row['raw_nll']:.4f}->{calibration_row['calibrated_nll']:.4f} "
        f"test_nll={test_row['raw_nll']:.4f}->{test_row['calibrated_nll']:.4f}"
    )
    return calibration_row, test_row


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, sd


def aggregate_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    metrics = [
        "raw_nll",
        "calibrated_nll",
        "delta_nll",
        "raw_brier",
        "calibrated_brier",
        "delta_brier",
        "raw_ece",
        "calibrated_ece",
        "delta_ece",
        "raw_center_accuracy",
        "calibrated_center_accuracy",
        "raw_center_macro_f1",
        "calibrated_center_macro_f1",
        "raw_mean_confidence",
        "calibrated_mean_confidence",
        "temperature",
    ]
    split_rows = [row for row in rows if row["split"] == split]
    summary: dict[str, Any] = {
        "fold": "mean_sd",
        "split": split,
        "num_folds": len(split_rows),
    }
    for metric in metrics:
        mean, sd = mean_sd([float(row[metric]) for row in split_rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_sd"] = sd
    return summary


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def format_mean_sd(mean: float, sd: float) -> str:
    return f"{float(mean):.4f} +- {float(sd):.4f}"


def markdown_table(test_rows: list[dict[str, Any]], test_summary: dict[str, Any]) -> str:
    lines = [
        "# E3 MC Pool-Then-Calibrate Temperature Scaling",
        "",
        "Main result split: outer test subject. Temperature is fitted on the calibration subject only.",
        "Deltas are calibrated minus raw, so negative values indicate improvement.",
        "",
        "| Fold | T* | Raw NLL | Cal. NLL | Delta NLL | Raw Brier | Cal. Brier | Delta Brier | Raw ECE | Cal. ECE | Delta ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(test_rows, key=lambda item: str(item["fold"])):
        lines.append(
            "| {fold} | {temp} | {raw_nll} | {cal_nll} | {delta_nll} | {raw_brier} | {cal_brier} | {delta_brier} | {raw_ece} | {cal_ece} | {delta_ece} |".format(
                fold=str(row["fold"]).upper(),
                temp=format_float(row["temperature"]),
                raw_nll=format_float(row["raw_nll"]),
                cal_nll=format_float(row["calibrated_nll"]),
                delta_nll=format_float(row["delta_nll"]),
                raw_brier=format_float(row["raw_brier"]),
                cal_brier=format_float(row["calibrated_brier"]),
                delta_brier=format_float(row["delta_brier"]),
                raw_ece=format_float(row["raw_ece"]),
                cal_ece=format_float(row["calibrated_ece"]),
                delta_ece=format_float(row["delta_ece"]),
            )
        )
    lines.append(
        "| Mean +- SD | - | {raw_nll} | {cal_nll} | {delta_nll} | {raw_brier} | {cal_brier} | {delta_brier} | {raw_ece} | {cal_ece} | {delta_ece} |".format(
            raw_nll=format_mean_sd(test_summary["raw_nll_mean"], test_summary["raw_nll_sd"]),
            cal_nll=format_mean_sd(test_summary["calibrated_nll_mean"], test_summary["calibrated_nll_sd"]),
            delta_nll=format_mean_sd(test_summary["delta_nll_mean"], test_summary["delta_nll_sd"]),
            raw_brier=format_mean_sd(test_summary["raw_brier_mean"], test_summary["raw_brier_sd"]),
            cal_brier=format_mean_sd(test_summary["calibrated_brier_mean"], test_summary["calibrated_brier_sd"]),
            delta_brier=format_mean_sd(test_summary["delta_brier_mean"], test_summary["delta_brier_sd"]),
            raw_ece=format_mean_sd(test_summary["raw_ece_mean"], test_summary["raw_ece_sd"]),
            cal_ece=format_mean_sd(test_summary["calibrated_ece_mean"], test_summary["calibrated_ece_sd"]),
            delta_ece=format_mean_sd(test_summary["delta_ece_mean"], test_summary["delta_ece_sd"]),
        )
    )
    lines.extend(
        [
            "",
            "Saved but not emphasized: raw/calibrated center accuracy and macro-F1. They are expected to match because pool-temperature scaling preserves class ordering.",
            "",
        ]
    )
    return "\n".join(lines)


def write_delta_svg(path: Path, test_rows: list[dict[str, Any]], test_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("delta_nll", "Delta NLL"),
        ("delta_brier", "Delta Brier"),
        ("delta_ece", "Delta ECE"),
    ]
    width = 860
    height = 360
    left = 70
    right = width - 35
    top = 45
    bottom = height - 65
    values = [float(row[key]) for row in test_rows for key, _ in metrics]
    max_abs = max(max(abs(value) for value in values), 1e-6) * 1.25

    def sy(value: float) -> float:
        return bottom - ((value + max_abs) / (2.0 * max_abs)) * (bottom - top)

    def sx(index: int) -> float:
        return left + (right - left) * (index + 0.5) / len(metrics)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="430" y="24" text-anchor="middle" font-size="18" font-weight="700">E3 Test Calibration Deltas</text>',
        f'<line x1="{left}" y1="{sy(0):.1f}" x2="{right}" y2="{sy(0):.1f}" stroke="#333" stroke-width="1.2" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1" />',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333" stroke-width="1" />',
        '<text x="430" y="342" text-anchor="middle" font-size="11" fill="#555">Delta = calibrated - raw. Negative is better.</text>',
    ]
    for tick in [-max_abs, 0.0, max_abs]:
        y = sy(tick)
        lines.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333" />')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{tick:.3f}</text>')

    colors = {"a": "#2f6fed", "b": "#c75a2a", "c": "#278456"}
    for metric_index, (metric, title) in enumerate(metrics):
        x = sx(metric_index)
        mean = float(test_summary[f"{metric}_mean"])
        sd = float(test_summary[f"{metric}_sd"])
        y_mean = sy(mean)
        y_low = sy(mean - sd)
        y_high = sy(mean + sd)
        lines.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="#111" stroke-width="1.5" />')
        lines.append(f'<line x1="{x - 12:.1f}" y1="{y_mean:.1f}" x2="{x + 12:.1f}" y2="{y_mean:.1f}" stroke="#111" stroke-width="3" />')
        for fold_index, row in enumerate(sorted(test_rows, key=lambda item: str(item["fold"]))):
            fold = str(row["fold"])
            jitter = [-18, 0, 18][fold_index % 3]
            value = float(row[metric])
            lines.append(
                f'<circle cx="{x + jitter:.1f}" cy="{sy(value):.1f}" r="5" fill="{colors.get(fold, "#666")}">'
                f'<title>Fold {fold.upper()} {title}: {value:.4f}</title></circle>'
            )
        lines.append(f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" font-size="12">{title}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_temperature_and_reports(args: argparse.Namespace) -> None:
    calibration_rows = []
    test_rows = []
    for fold in [item.lower() for item in args.folds]:
        calibration_row, test_row = run_temperature_for_fold(args, fold)
        calibration_rows.append(calibration_row)
        test_rows.append(test_row)

    calibration_summary = aggregate_rows(calibration_rows, "calib")
    test_summary = aggregate_rows(test_rows, "test")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "e3_calibration_fit_sanity.csv", calibration_rows)
    write_csv(args.report_dir / "e3_test_fold_metrics.csv", test_rows)
    write_csv(args.report_dir / "e3_test_mean_sd.csv", [test_summary])
    write_json(
        args.report_dir / "e3_mc_temperature_scaling_summary.json",
        {
            "experiment": "E3 MC pool-then-calibrate temperature scaling",
            "selected_branch": "mc_dropout",
            "main_split": "test",
            "temperature_fit_split": "calib",
            "calibration_is_sanity_not_main_result": True,
            "no_test_decision": True,
            "num_passes": int(args.num_passes),
            "eps": float(args.eps),
            "temperature_optimizer": args.temperature_optimizer,
            "ece_bins": int(args.ece_bins),
            "ece_bin_type": "fixed_equal_width",
            "ece_bin_edge_convention": "[lower, upper) except final bin [lower, upper]",
            "nll_definition": "mean_over_samples(-log(p_true)); natural logarithm",
            "brier_definition": "mean_over_samples(sum_over_classes((p-onehot)^2))",
            "divide_by_num_classes": False,
            "temperature_rule": "q=softmax(log(max(pbar_MC, eps))/T)",
            "delta_definition": "calibrated_minus_raw; negative means improvement",
            "fold_test_metrics": test_rows,
            "fold_calibration_sanity": calibration_rows,
            "test_mean_sd": test_summary,
            "calibration_mean_sd": calibration_summary,
        },
    )
    (args.report_dir / "e3_mc_temperature_scaling_summary.md").write_text(
        markdown_table(test_rows, test_summary),
        encoding="utf-8",
        newline="\n",
    )
    write_delta_svg(args.report_dir / "e3_test_calibration_deltas.svg", test_rows, test_summary)
    print(f"[DONE] wrote E3 reports under {args.report_dir}")


def run(args: argparse.Namespace) -> None:
    folds = [item.lower() for item in args.folds]
    streams = [item.lower() for item in args.streams]
    splits = [item.lower() for item in args.splits]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")
    for stream in streams:
        if stream not in STREAMS:
            raise ValueError(f"Unknown stream {stream!r}; expected one of {STREAMS}")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")

    if args.mode in {"all", "stream"}:
        try:
            import torch  # noqa: F401
            import mmcv  # noqa: F401
            import pyskl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "E3 MC inference requires torch, mmcv, and pyskl in the active Python environment."
            ) from exc
        import torch

        if args.num_threads > 0:
            torch.set_num_threads(args.num_threads)
        set_global_seed(args.seed, deterministic_cudnn=args.deterministic_cudnn)
        device = resolve_device(args.device)
        print(f"[INFO] E3 mode={args.mode} device={device} seed={args.seed} passes={args.num_passes}")
        for fold in folds:
            for split in splits:
                for stream in streams:
                    run_stream(args, fold, stream, split, device)

    if args.mode in {"all", "fusion"}:
        for fold in folds:
            for split in splits:
                summarize_fusion_for_split(args, fold, split)

    if args.mode in {"all", "temperature"}:
        missing_splits = [split for split in ["calib", "test"] if split not in splits]
        if args.mode == "all" and missing_splits:
            raise ValueError("E3 temperature fitting requires both calib and test splits.")
        run_temperature_and_reports(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "stream", "fusion", "temperature"],
        default="all",
        help="stream generates stream MC outputs; fusion fuses streams; temperature fits T and reports.",
    )
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--streams", nargs="+", default=STREAMS)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/stgcn++/stgcn++_radarv4/rerun/e1"),
    )
    parser.add_argument(
        "--e1-work-root",
        type=Path,
        default=Path("work_dirs/rerun/e1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work_dirs/rerun/e3/mc_temperature_scaling"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("rerun/e3/reports"),
    )
    parser.add_argument("--num-passes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic-cudnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--temperature-optimizer",
        choices=["torch_lbfgs", "numpy_golden"],
        default="torch_lbfgs",
    )
    parser.add_argument("--temperature-max-iter", type=int, default=100)
    parser.add_argument("--temperature-lr", type=float, default=0.1)
    parser.add_argument("--min-temperature", type=float, default=0.01)
    parser.add_argument("--max-temperature", type=float, default=100.0)
    args = parser.parse_args()

    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.ece_bins <= 0:
        raise ValueError("--ece-bins must be positive")
    if args.temperature_max_iter <= 0:
        raise ValueError("--temperature-max-iter must be positive")
    if args.temperature_lr <= 0:
        raise ValueError("--temperature-lr must be positive")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
