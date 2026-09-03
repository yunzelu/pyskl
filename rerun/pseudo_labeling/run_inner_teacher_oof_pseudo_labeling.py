"""Generate one inner teacher system's OOF skeleton pseudo-label artifacts.

This script runs steps 1-5 of the canonical out-of-fold pseudo-label
generation protocol for one outer fold and one inner teacher:

1. Run 30-pass MC-dropout inference on the teacher's calibration subject.
2. Fuse Joint/Bone probabilities within every MC pass.
3. Fit one post-fusion pool-temperature on the calibration MC mean.
4. Use calibration-set raw MC MI to estimate the teacher-specific q95 scale.
5. Run 30-pass MC-dropout inference on the two pseudo-target subjects and
   save training-safe pseudo labels, audit labels, and full fused MC passes.

The radar-training-safe pseudo table intentionally omits manual labels. The
audit table keeps manual labels and boundary distance for diagnostics only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import pickle
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun.pseudo_labeling.generate_inner_teacher_training_artifacts import (  # noqa: E402
    FOLDS,
    LABELS,
)


SCHEMA_VERSION = "oof_skeleton_pseudo_labels_v1"
DATASET_ID = "radarv4_yolo26xpose"
WINDOW_SIZE = 60
STRIDE = 12
CENTER_OFFSET = 30
CAMERA_FPS = 30.0
CONDITION = "inner_teacher_continuous_window_w60_s12"

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for index, label in enumerate(LABELS)}
LABEL_ALIASES = {
    "layfloor-stationary": "lie-stationary",
    "laybed-stationary": "lie-stationary",
    "lie-stationary": "lie-stationary",
    "lying-stationary": "lie-stationary",
    "sit-stationary": "sit-stationary",
    "walking": "walk",
    "walk": "walk",
    "falling": "fall",
    "fall": "fall",
    "transition-layfloor-to-sit": "transition-lie-to-sit",
    "transition-laybed-to-sit": "transition-lie-to-sit",
    "transition-lie-to-sit": "transition-lie-to-sit",
    "transition-layfloor-to-stand": "transition-lie-to-stand",
    "transition-laybed-to-stand": "transition-lie-to-stand",
    "transition-lie-to-stand": "transition-lie-to-stand",
    "transition-sit-to-layfloor": "transition-sit-to-lie",
    "transition-sit-to-laybed": "transition-sit-to-lie",
    "transition-sit-to-lie": "transition-sit-to-lie",
    "transition-sit-to-stand": "transition-sit-to-stand",
    "transition-stand-to-sit": "transition-stand-to-sit",
}

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
    video_infos: list[dict[str, Any]]
    output_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str | None
    checkpoint_selection: dict[str, Any]
    config_path: Path
    dropout_verification: dict[str, Any]
    pipeline: list[str]


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
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write row dictionaries to Parquet using pyarrow or pandas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(json_ready(rows))
        pq.write_table(table, path)
        return
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Writing Parquet requires either pyarrow or pandas with a Parquet "
            "engine installed in the active environment."
        ) from exc


def read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet table into row dictionaries using pyarrow or pandas."""

    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Reading Parquet requires either pyarrow or pandas with a Parquet "
            "engine installed in the active environment."
        ) from exc


def normalized_existing_path(value: Any) -> Path:
    raw = str(value)
    direct = Path(raw)
    if direct.exists():
        return direct
    normalized = Path(raw.replace("\\", "/"))
    if normalized.exists():
        return normalized
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    key = str(label).strip().lower()
    if not key:
        return None
    return LABEL_ALIASES.get(key)


def softmax_np(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


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


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray, eps: float) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    picked = values[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def apply_pool_temperature(
    probabilities: np.ndarray,
    temperature: float,
    eps: float,
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
        loss = F.nll_loss(F.log_softmax(log_probabilities / temperature, dim=-1), labels)
        loss.backward()
        return loss

    with torch.no_grad():
        raw_nll = float(F.nll_loss(log_probabilities, labels).item())
    started = time.time()
    optimizer.step(closure)
    seconds = float(time.time() - started)
    temperature = float(log_temperature.detach().exp().item())
    calibrated = apply_pool_temperature(calibration_probabilities, temperature, eps)
    calibrated_nll = multiclass_nll(calibrated, calibration_labels, eps)
    if calibrated_nll > raw_nll + 1e-8:
        raise RuntimeError(
            f"Calibration NLL increased after fitting: raw={raw_nll:.12g}, "
            f"calibrated={calibrated_nll:.12g}"
        )
    return {
        "temperature": temperature,
        "optimizer": "torch_lbfgs_log_temperature",
        "max_iter": int(max_iter),
        "lbfgs_lr": float(lbfgs_lr),
        "fit_seconds": seconds,
        "raw_calibration_nll": raw_nll,
        "calibrated_calibration_nll": calibrated_nll,
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


def config_path(config_root: Path, fold: str, teacher: str, stream: str) -> Path:
    return config_root / f"fold_{fold}" / teacher / f"{stream}.py"


def teacher_output_root(output_root: Path, fold: str, teacher: str) -> Path:
    return output_root / f"fold_{fold}" / teacher


def stream_output_dir(output_root: Path, fold: str, teacher: str, split: str, stream: str) -> Path:
    return teacher_output_root(output_root, fold, teacher) / "stream_outputs" / split / stream


def parse_checkpoint_epoch(path: Path) -> int:
    match = re.search(r"best_macro_f1_epoch_(\d+)\.pth$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_selected_checkpoint(work_root: Path, fold: str, teacher: str, stream: str) -> tuple[Path, dict[str, Any]]:
    checkpoint_dir = work_root / f"fold_{fold}" / teacher / stream
    checkpoints = sorted(checkpoint_dir.glob("best_macro_f1_epoch_*.pth"), key=parse_checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No best_macro_f1 checkpoint found in {checkpoint_dir}")
    selected = checkpoints[-1]
    return selected, {
        "checkpoint_dir": checkpoint_dir,
        "candidate_count": len(checkpoints),
        "candidates": checkpoints,
        "selected": selected,
        "selection_rule": (
            "Select best_macro_f1_epoch_*.pth. If multiple files remain, use "
            "the largest epoch id, matching the documented post-training "
            "--test-best fallback."
        ),
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


def load_model_and_dataset(config_file: Path, checkpoint_path: Path, split: str):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from pyskl.datasets import build_dataset
    from pyskl.models import build_model
    from pyskl.utils.mc_dropout import verify_gcn_head_dropout

    cfg = Config.fromfile(str(config_file))
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None

    dataset = build_dataset(deterministic_dataset_cfg(cfg, split), dict(test_mode=True))
    pipeline = assert_deterministic_pipeline(dataset, f"OOF {split} dataset")
    assert_unique_sample_ids(sample_ids_from_video_infos(dataset.video_infos), f"OOF {split} dataset")

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


def run_mc_sanity_checks(model: Any, loader: Any, device: Any, atol: float) -> dict[str, Any]:
    import torch
    from pyskl.utils.mc_dropout import enable_head_mc_dropout

    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("MC sanity-check loader is empty") from exc

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
        raise RuntimeError("MC-dropout sanity check failed: stochastic passes matched.")

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
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from pyskl.utils.mc_dropout import enable_head_mc_dropout

    model.to(device)
    enable_head_mc_dropout(model, expected_dropout=0.5, expected_num_classes=len(LABELS))

    all_passes = []
    reference_labels = None
    for pass_index in range(num_passes):
        started = time.time()
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
        print(f"[INFO] MC pass {pass_index + 1}/{num_passes} took {time.time() - started:.1f}s")

    model.eval()
    probabilities = torch.stack(all_passes, dim=0).numpy().astype(np.float32, copy=False)
    labels = reference_labels.numpy().astype(np.int64, copy=False)
    validate_probabilities(probabilities, "collected MC probabilities")
    return probabilities, labels


def parse_window_candidate_index(frame_dir: Any, fallback_start: int) -> int:
    match = re.search(r"__win(\d+)__", str(frame_dir))
    if match:
        return int(match.group(1))
    return int(fallback_start // STRIDE)


def skeleton_sample_id_from_item(item: dict[str, Any], fold: str, teacher: str) -> str:
    recording_id = str(item["session_name"])
    start_retained = int(item["window_row_start"])
    center_source_frame = int(item["center_source_frame"])
    return (
        f"fold_{fold}|{teacher}|{recording_id}|"
        f"start_{start_retained:06d}|center_frame_{center_source_frame:06d}"
    )


def sample_identity_from_item(item: dict[str, Any], fold: str, teacher: str) -> dict[str, Any]:
    source_frames = np.asarray(item["source_frame_indices"], dtype=np.int64)
    timestamps = np.asarray(item["timestamps_sec"], dtype=np.float64)
    if source_frames.shape[0] != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} source frames, got {source_frames.shape}")
    if timestamps.shape[0] != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} timestamps, got {timestamps.shape}")

    start_retained = int(item["window_row_start"])
    end_retained_exclusive = int(item["window_row_end_exclusive"])
    center_retained = start_retained + CENTER_OFFSET
    if end_retained_exclusive - start_retained != WINDOW_SIZE:
        raise ValueError("Retained window length does not equal 60")
    if int(item["center_row_offset"]) != CENTER_OFFSET:
        raise ValueError("Unexpected center offset in source annotation")
    if int(source_frames[CENTER_OFFSET]) != int(item["center_source_frame"]):
        raise ValueError("source_frame_indices[30] does not match center_source_frame")
    if start_retained % STRIDE != 0:
        raise ValueError(f"window_row_start {start_retained} is not divisible by stride {STRIDE}")

    frame_dir = str(item.get("frame_dir", ""))
    center_source_frame = int(item["center_source_frame"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "outer_fold": fold.upper(),
        "inner_teacher_id": teacher.upper(),
        "subject_id": str(item["subject"]).lower(),
        "recording_id": str(item["session_name"]).lower(),
        "skeleton_sample_id": skeleton_sample_id_from_item(item, fold, teacher),
        "window_candidate_index": int(start_retained // STRIDE),
        "accepted_window_index_in_recording": parse_window_candidate_index(frame_dir, start_retained),
        "session_index": int(item.get("session_index", -1)),
        "session": str(item.get("session", "")),
        "session_family": str(item.get("session_family", "")),
        "window_start_retained_idx": start_retained,
        "window_end_retained_idx_exclusive": end_retained_exclusive,
        "center_retained_idx": center_retained,
        "source_frame_start": int(source_frames[0]),
        "source_frame_center": center_source_frame,
        "source_frame_end": int(source_frames[-1]),
        "source_timestamp_start_sec": float(timestamps[0]),
        "source_timestamp_center_sec": float(timestamps[CENTER_OFFSET]),
        "source_timestamp_end_sec": float(timestamps[-1]),
        "center_timestamp_sec": float(item["center_timestamp_sec"]),
        "nominal_camera_time_center_sec": float(center_source_frame / CAMERA_FPS),
        "camera_fps_for_nominal_time": CAMERA_FPS,
        "center_timestamp_policy": "use original source frame timestamp, not retained index / fps",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "center_offset": CENTER_OFFSET,
        "max_adjacent_gap_sec": float(item["max_adjacent_gap_sec"]),
        "window_span_sec": float(item["window_span_sec"]),
        "source_jsonl_path": str(item.get("source_jsonl_path", "")),
        "processed_jsonl_path": str(item.get("processed_jsonl_path", "")),
    }


def sample_ids_from_video_infos(video_infos: list[dict[str, Any]], fold: str = "x", teacher: str = "tx") -> list[dict[str, Any]]:
    return [sample_identity_from_item(item, fold, teacher) for item in video_infos]


def assert_unique_sample_ids(sample_ids: list[dict[str, Any]], name: str) -> None:
    keys = [item["skeleton_sample_id"] for item in sample_ids]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains {len(keys) - len(set(keys))} duplicate skeleton_sample_id values")


def labels_from_video_infos(video_infos: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(item["label"]) for item in video_infos], dtype=np.int64)


def source_frame_arrays(video_infos: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    frames = np.stack(
        [np.asarray(item["source_frame_indices"], dtype=np.int32) for item in video_infos],
        axis=0,
    )
    timestamps = np.stack(
        [np.asarray(item["timestamps_sec"], dtype=np.float64) for item in video_infos],
        axis=0,
    )
    return frames, timestamps


def load_raw_segments(path_value: Any, cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    path = normalized_existing_path(path_value)
    cache_key = str(path)
    if cache_key in cache:
        return cache[cache_key]
    if not path.exists():
        raise FileNotFoundError(f"Cannot find source JSONL metadata file: {path_value}")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.loads(f.readline())
    segments = metadata.get("annotation_info", {}).get("segments")
    if not isinstance(segments, list):
        raise RuntimeError(f"Missing annotation_info.segments in {path}")
    cache[cache_key] = [dict(segment) for segment in segments]
    return cache[cache_key]


def manual_boundary_info(
    item: dict[str, Any],
    segment_cache: dict[str, list[dict[str, Any]]],
    allow_missing: bool,
) -> dict[str, Any]:
    center_frame = int(item["center_source_frame"])
    try:
        segments = load_raw_segments(item["source_jsonl_path"], segment_cache)
    except FileNotFoundError:
        if not allow_missing:
            raise
        label_id = int(item["label"])
        return {
            "manual_label_at_skeleton_center": label_id,
            "manual_label_name_at_skeleton_center": ID_TO_LABEL[label_id],
            "manual_center_raw_label": str(item.get("center_raw_label", "")),
            "source_pyskl_frame_dir": str(item.get("frame_dir", "")),
            "manual_segment_start_frame": None,
            "manual_segment_end_frame": None,
            "distance_to_manual_boundary_frames": None,
            "manual_boundary_source": "missing_source_jsonl",
        }

    for segment_index, segment in enumerate(segments):
        if "start_frame" not in segment or "end_frame" not in segment:
            continue
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start <= center_frame <= end:
            label_name = normalize_label(segment.get("label"))
            label_id = None if label_name is None else LABEL_TO_ID[label_name]
            if label_id is None and not allow_missing:
                raise RuntimeError(
                    f"Center frame {center_frame} has non-final manual label {segment.get('label')!r}"
                )
            if label_id is not None and label_id != int(item["label"]):
                raise RuntimeError(
                    f"Manual center label mismatch for {item.get('frame_dir')}: "
                    f"JSONL={label_id}, pkl={item['label']}"
                )
            return {
                "manual_label_at_skeleton_center": None if label_id is None else int(label_id),
                "manual_label_name_at_skeleton_center": "" if label_name is None else label_name,
                "manual_center_raw_label": str(segment.get("label", "")),
                "source_pyskl_frame_dir": str(item.get("frame_dir", "")),
                "manual_segment_index": int(segment_index),
                "manual_segment_start_frame": start,
                "manual_segment_end_frame": end,
                "distance_to_manual_boundary_frames": int(min(center_frame - start, end - center_frame)),
                "manual_boundary_source": "annotation_info.segments",
            }

    if not allow_missing:
        raise RuntimeError(f"No manual segment contains center frame {center_frame} for {item.get('frame_dir')}")
    label_id = int(item["label"])
    return {
        "manual_label_at_skeleton_center": label_id,
        "manual_label_name_at_skeleton_center": ID_TO_LABEL[label_id],
        "manual_center_raw_label": str(item.get("center_raw_label", "")),
        "source_pyskl_frame_dir": str(item.get("frame_dir", "")),
        "manual_segment_start_frame": None,
        "manual_segment_end_frame": None,
        "distance_to_manual_boundary_frames": None,
        "manual_boundary_source": "no_matching_segment",
    }


def rows_from_predictions(
    video_infos: list[dict[str, Any]],
    quantities: dict[str, np.ndarray],
    calibrated_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    fold: str,
    teacher: str,
    split: str,
    num_passes: int,
    temperature: float,
    mi_q95: float,
    eps: float,
    segment_cache: dict[str, list[dict[str, Any]]],
    include_manual_audit_fields: bool,
    allow_missing_boundary_audit: bool,
) -> list[dict[str, Any]]:
    raw_mean = quantities["mean_probabilities"]
    predictions = quantities["prediction"].astype(np.int64)
    validate_probabilities(raw_mean, "raw MC mean")
    validate_probabilities(calibrated_probabilities, "calibrated MC mean")
    if raw_mean.shape != calibrated_probabilities.shape:
        raise ValueError("Raw and calibrated probability shapes differ")
    calibrated_predictions = np.argmax(calibrated_probabilities, axis=1).astype(np.int64)
    if not np.array_equal(predictions, calibrated_predictions):
        raise RuntimeError("Pool-temperature scaling changed pseudo-label argmax")

    denominator = max(float(mi_q95), float(eps))
    u_norm = np.minimum(quantities["mutual_information"].astype(np.float64) / denominator, 1.0)
    reliability = 0.1 + 0.9 * (1.0 - u_norm)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(video_infos):
        if int(item["label"]) != int(labels[index]):
            raise RuntimeError("Dataset label and collected label differ")
        row = sample_identity_from_item(item, fold, teacher)
        row.update(
            {
                "split_role": split,
                "mc_passes": int(num_passes),
                "hard_pseudo_label_id": int(predictions[index]),
                "hard_pseudo_label_name": ID_TO_LABEL[int(predictions[index])],
                "mc_predictive_entropy": float(quantities["predictive_entropy"][index]),
                "mc_expected_entropy": float(quantities["expected_entropy"][index]),
                "mc_mi_raw": float(quantities["mutual_information"][index]),
                "temperature": float(temperature),
                "calibrated_argmax_id": int(calibrated_predictions[index]),
                "calibrated_argmax_name": ID_TO_LABEL[int(calibrated_predictions[index])],
                "mi_q95_calibration": float(mi_q95),
                "reliability_weight": float(reliability[index]),
                "reliability_weight_formula": "0.1 + 0.9 * (1 - min(mi / max(q95, eps), 1))",
            }
        )
        for class_id in range(len(LABELS)):
            row[f"mc_raw_p{class_id}"] = float(raw_mean[index, class_id])
            row[f"mc_cal_p{class_id}"] = float(calibrated_probabilities[index, class_id])
        if include_manual_audit_fields:
            audit = manual_boundary_info(item, segment_cache, allow_missing_boundary_audit)
            row.update(audit)
            manual_id = row["manual_label_at_skeleton_center"]
            row["pseudo_label_correct"] = None if manual_id is None else bool(int(predictions[index]) == int(manual_id))
        rows.append(row)
    return rows


def save_fused_npz(
    path: Path,
    *,
    sample_ids: list[str],
    fused_probabilities: np.ndarray,
    video_infos: list[dict[str, Any]],
    include_labels: bool,
    labels: np.ndarray | None,
    joint_probabilities: np.ndarray | None,
    bone_probabilities: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_frames, timestamps = source_frame_arrays(video_infos)
    payload: dict[str, Any] = {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "fused_mc_probabilities": fused_probabilities.astype(np.float32, copy=False),
        "source_frame_indices": source_frames.astype(np.int32, copy=False),
        "timestamps_sec": timestamps.astype(np.float64, copy=False),
    }
    if include_labels:
        if labels is None:
            raise ValueError("labels are required when include_labels=True")
        payload["manual_labels"] = labels.astype(np.int64, copy=False)
    if joint_probabilities is not None:
        payload["joint_mc_probabilities"] = joint_probabilities.astype(np.float32, copy=False)
    if bone_probabilities is not None:
        payload["bone_mc_probabilities"] = bone_probabilities.astype(np.float32, copy=False)
    np.savez_compressed(path, **payload)


def load_existing_stream_result(
    out_dir: Path,
    expected_passes: int,
    fold: str,
    teacher: str,
    config_file: Path,
    checkpoint_path: Path,
    checkpoint_selection: dict[str, Any],
    checkpoint_sha256: str | None,
) -> StreamResult:
    probabilities = np.load(out_dir / "mc_prob_passes.npy")
    labels = np.load(out_dir / "labels.npy").astype(np.int64, copy=False)
    sample_ids = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    video_infos = load_pickle(out_dir / "video_infos.pkl")
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    if probabilities.shape[0] != expected_passes:
        raise ValueError(
            f"{out_dir / 'mc_prob_passes.npy'} has {probabilities.shape[0]} passes, expected {expected_passes}."
        )
    validate_probabilities(probabilities, f"existing {out_dir} MC probabilities")
    return StreamResult(
        probabilities=probabilities,
        labels=labels,
        sample_ids=sample_ids,
        video_infos=video_infos,
        output_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_selection=checkpoint_selection,
        config_path=config_file,
        dropout_verification=metadata.get("dropout_verification", {}),
        pipeline=metadata.get("pipeline", []),
    )


def load_pickle(path: Path) -> Any:
    install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_stream(
    args: argparse.Namespace,
    fold: str,
    teacher: str,
    stream: str,
    split: str,
    device: Any,
) -> StreamResult:
    import torch

    out_dir = stream_output_dir(args.output_root, fold, teacher, split, stream)
    config_file = config_path(args.config_root, fold, teacher, stream)
    checkpoint_path, checkpoint_selection = find_selected_checkpoint(
        args.inner_teacher_work_root,
        fold,
        teacher,
        stream,
    )
    checkpoint_sha256 = None if args.skip_checkpoint_hash else sha256_file(checkpoint_path)

    expected_files = [
        out_dir / "mc_prob_passes.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
        out_dir / "video_infos.pkl",
        out_dir / "metadata.json",
    ]
    if all(path.exists() for path in expected_files) and not args.overwrite:
        print(f"[SKIP] existing stream MC outputs: {out_dir}")
        return load_existing_stream_result(
            out_dir,
            args.num_passes,
            fold,
            teacher,
            config_file,
            checkpoint_path,
            checkpoint_selection,
            checkpoint_sha256,
        )

    print(
        f"[INFO] OOF MC fold={fold} teacher={teacher} stream={stream} split={split} "
        f"checkpoint={checkpoint_path}"
    )
    _, dataset, model, verification, pipeline = load_model_and_dataset(config_file, checkpoint_path, split)
    labels = labels_from_video_infos(dataset.video_infos)
    sample_ids = sample_ids_from_video_infos(dataset.video_infos, fold, teacher)
    assert_unique_sample_ids(sample_ids, f"{fold}/{teacher}/{stream}/{split}")

    loader = build_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    sanity = None
    if not args.skip_sanity_check:
        sanity = run_mc_sanity_checks(model, loader, device, args.sanity_atol)

    started = time.time()
    probabilities, collected_labels = collect_mc_probabilities(model, loader, device, args.num_passes)
    seconds = float(time.time() - started)
    if not np.array_equal(collected_labels, labels):
        raise RuntimeError(f"Collected labels do not match dataset order for {fold}/{teacher}/{stream}/{split}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mc_prob_passes.npy", probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels.astype(np.int64, copy=False))
    write_json(out_dir / "sample_ids.json", sample_ids)
    save_pickle(out_dir / "video_infos.pkl", dataset.video_infos)
    write_json(
        out_dir / "metadata.json",
        {
            "protocol": "OOF inner-teacher stream MC-dropout inference",
            "schema_version": SCHEMA_VERSION,
            "fold": fold,
            "teacher": teacher,
            "stream": stream,
            "split": split,
            "num_passes": int(args.num_passes),
            "seed_set_once_by_teacher_script": int(args.seed),
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selection": checkpoint_selection,
            "config": config_file,
            "pipeline": pipeline,
            "dropout_verification": verification,
            "sanity_check": sanity,
            "seconds": seconds,
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[DONE] fold={fold} teacher={teacher} stream={stream} split={split} "
        f"N={len(labels)} seconds={seconds:.1f}"
    )
    return StreamResult(
        probabilities=probabilities,
        labels=labels,
        sample_ids=sample_ids,
        video_infos=dataset.video_infos,
        output_dir=out_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_selection=checkpoint_selection,
        config_path=config_file,
        dropout_verification=verification,
        pipeline=pipeline,
    )


def assert_stream_alignment(joint: StreamResult, bone: StreamResult, split: str) -> None:
    if joint.sample_ids != bone.sample_ids:
        raise RuntimeError(f"Joint/Bone sample ids differ for split={split}")
    if not np.array_equal(joint.labels, bone.labels):
        raise RuntimeError(f"Joint/Bone labels differ for split={split}")
    if joint.probabilities.shape != bone.probabilities.shape:
        raise RuntimeError(f"Joint/Bone MC probability shapes differ for split={split}")


def fuse_streams(joint: StreamResult, bone: StreamResult, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    assert_stream_alignment(joint, bone, split)
    fused = 0.5 * (joint.probabilities + bone.probabilities)
    validate_probabilities(fused, f"{split} fused MC probabilities")
    return fused.astype(np.float32, copy=False), mc_quantities(fused)


def write_global_metadata_files(output_root: Path) -> None:
    write_json(
        output_root / "class_mapping.json",
        {
            "dataset_id": DATASET_ID,
            "classes": [{"label_id": index, "label_name": label} for index, label in enumerate(LABELS)],
            "label_to_id": LABEL_TO_ID,
        },
    )
    write_json(
        output_root / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "row_level_pseudo_label_format": "parquet",
            "full_mc_array_format": "compressed_npz",
            "window_definition": {
                "window_size": WINDOW_SIZE,
                "stride": STRIDE,
                "center_offset": CENTER_OFFSET,
                "retained_index_fields": [
                    "window_start_retained_idx",
                    "center_retained_idx",
                    "window_end_retained_idx_exclusive",
                ],
                "source_frame_fields": [
                    "source_frame_start",
                    "source_frame_center",
                    "source_frame_end",
                ],
                "window_candidate_index": "window_start_retained_idx / stride",
                "accepted_window_index_in_recording": "parsed from generated win000000 sample token",
                "do_not_use": "center_retained_idx / 30 as source time",
            },
            "probability_fields": {
                "mc_raw_p0_to_p8": "raw 30-pass post-fusion MC predictive mean",
                "mc_cal_p0_to_p8": "pool-temperature-calibrated MC predictive mean",
            },
            "training_safe_file_excludes_manual_labels": True,
            "audit_file_includes_manual_labels": True,
            "reliability_weight": {
                "mi_norm": "min(mi / max(mi_q95_calibration, eps), 1)",
                "weight": "0.1 + 0.9 * (1 - mi_norm)",
                "gamma": 1,
            },
        },
    )


def write_rows(path: Path, rows: list[dict[str, Any]], write_csv_copy: bool) -> None:
    write_parquet(path, rows)
    if write_csv_copy:
        write_csv(path.with_suffix(".csv"), rows)


def run_teacher(args: argparse.Namespace) -> None:
    try:
        import torch  # noqa: F401
        import mmcv  # noqa: F401
        import pyskl  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OOF pseudo-label inference requires torch, mmcv, and pyskl in the active environment."
        ) from exc

    fold = args.fold.lower()
    teacher = args.teacher.lower()
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold {fold!r}")
    if teacher not in FOLDS[fold]["teachers"]:
        raise ValueError(f"Unknown teacher {teacher!r} for fold {fold}")
    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")

    teacher_spec = FOLDS[fold]["teachers"][teacher]
    set_global_seed(args.seed, deterministic_cudnn=args.deterministic_cudnn)
    device = resolve_device(args.device)
    write_global_metadata_files(args.output_root)

    print(
        f"[INFO] OOF teacher start fold={fold} teacher={teacher} "
        f"device={device} seed={args.seed} passes={args.num_passes}"
    )

    calibration_streams = {
        stream: run_stream(args, fold, teacher, stream, "calib", device)
        for stream in ["joint", "bone"]
    }
    calibration_fused, calibration_quantities = fuse_streams(
        calibration_streams["joint"],
        calibration_streams["bone"],
        "calib",
    )
    calibration_mean = calibration_quantities["mean_probabilities"]
    calibration_labels = calibration_streams["joint"].labels
    temperature_fit = fit_pool_temperature_torch(
        calibration_mean,
        calibration_labels,
        eps=args.eps,
        max_iter=args.temperature_max_iter,
        lbfgs_lr=args.temperature_lr,
    )
    temperature = float(temperature_fit["temperature"])
    mi_q95 = float(np.quantile(calibration_quantities["mutual_information"], args.mi_quantile))
    calibration_calibrated = apply_pool_temperature(calibration_mean, temperature, args.eps)

    pseudo_streams = {
        stream: run_stream(args, fold, teacher, stream, "pseudo_target", device)
        for stream in ["joint", "bone"]
    }
    pseudo_fused, pseudo_quantities = fuse_streams(
        pseudo_streams["joint"],
        pseudo_streams["bone"],
        "pseudo_target",
    )
    pseudo_mean = pseudo_quantities["mean_probabilities"]
    pseudo_calibrated = apply_pool_temperature(pseudo_mean, temperature, args.eps)

    teacher_root = teacher_output_root(args.output_root, fold, teacher)
    segment_cache: dict[str, list[dict[str, Any]]] = {}

    calibration_rows = rows_from_predictions(
        calibration_streams["joint"].video_infos,
        calibration_quantities,
        calibration_calibrated,
        calibration_labels,
        fold=fold,
        teacher=teacher,
        split="calib",
        num_passes=args.num_passes,
        temperature=temperature,
        mi_q95=mi_q95,
        eps=args.eps,
        segment_cache=segment_cache,
        include_manual_audit_fields=True,
        allow_missing_boundary_audit=args.allow_missing_boundary_audit,
    )
    pseudo_safe_rows = rows_from_predictions(
        pseudo_streams["joint"].video_infos,
        pseudo_quantities,
        pseudo_calibrated,
        pseudo_streams["joint"].labels,
        fold=fold,
        teacher=teacher,
        split="pseudo_target",
        num_passes=args.num_passes,
        temperature=temperature,
        mi_q95=mi_q95,
        eps=args.eps,
        segment_cache=segment_cache,
        include_manual_audit_fields=False,
        allow_missing_boundary_audit=args.allow_missing_boundary_audit,
    )
    pseudo_audit_rows = rows_from_predictions(
        pseudo_streams["joint"].video_infos,
        pseudo_quantities,
        pseudo_calibrated,
        pseudo_streams["joint"].labels,
        fold=fold,
        teacher=teacher,
        split="pseudo_target",
        num_passes=args.num_passes,
        temperature=temperature,
        mi_q95=mi_q95,
        eps=args.eps,
        segment_cache=segment_cache,
        include_manual_audit_fields=True,
        allow_missing_boundary_audit=args.allow_missing_boundary_audit,
    )

    expected_pseudo = int(teacher_spec["expected_pseudo_target_windows"])
    if len(pseudo_safe_rows) != expected_pseudo:
        raise RuntimeError(
            f"Pseudo-target row count {len(pseudo_safe_rows)} does not match expected {expected_pseudo}"
        )

    write_rows(teacher_root / "calibration_predictions.parquet", calibration_rows, args.write_csv_copy)
    write_rows(teacher_root / "pseudo_predictions.parquet", pseudo_safe_rows, args.write_csv_copy)
    write_rows(teacher_root / "pseudo_predictions_audit.parquet", pseudo_audit_rows, args.write_csv_copy)

    calibration_sample_ids = [row["skeleton_sample_id"] for row in calibration_rows]
    pseudo_sample_ids = [row["skeleton_sample_id"] for row in pseudo_safe_rows]
    save_fused_npz(
        teacher_root / "calibration_mc_fused_samples.npz",
        sample_ids=calibration_sample_ids,
        fused_probabilities=calibration_fused,
        video_infos=calibration_streams["joint"].video_infos,
        include_labels=True,
        labels=calibration_labels,
        joint_probabilities=calibration_streams["joint"].probabilities if args.save_stream_pass_probabilities else None,
        bone_probabilities=calibration_streams["bone"].probabilities if args.save_stream_pass_probabilities else None,
    )
    save_fused_npz(
        teacher_root / "mc_fused_samples.npz",
        sample_ids=pseudo_sample_ids,
        fused_probabilities=pseudo_fused,
        video_infos=pseudo_streams["joint"].video_infos,
        include_labels=False,
        labels=None,
        joint_probabilities=pseudo_streams["joint"].probabilities if args.save_stream_pass_probabilities else None,
        bone_probabilities=pseudo_streams["bone"].probabilities if args.save_stream_pass_probabilities else None,
    )

    metadata = {
        "protocol": "OOF inner-teacher skeleton pseudo-label generation",
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "outer_fold": fold.upper(),
        "inner_teacher_id": teacher.upper(),
        "train_subjects": teacher_spec["train"],
        "validation_subject": FOLDS[fold]["val"],
        "calibration_subject": FOLDS[fold]["calib"],
        "pseudo_target_subjects": teacher_spec["pseudo_target"],
        "outer_test_subject_excluded": FOLDS[fold]["test"],
        "joint_checkpoint_path": calibration_streams["joint"].checkpoint_path,
        "joint_checkpoint_sha256": calibration_streams["joint"].checkpoint_sha256,
        "bone_checkpoint_path": calibration_streams["bone"].checkpoint_path,
        "bone_checkpoint_sha256": calibration_streams["bone"].checkpoint_sha256,
        "checkpoint_selection": {
            "joint": calibration_streams["joint"].checkpoint_selection,
            "bone": calibration_streams["bone"].checkpoint_selection,
        },
        "dropout_rate": 0.5,
        "mc_seed": int(args.seed),
        "mc_collection_order": ["calib_joint", "calib_bone", "pseudo_target_joint", "pseudo_target_bone"],
        "num_passes": int(args.num_passes),
        "fusion_rule": "p_fused^(k) = 0.5 * (p_joint^(k) + p_bone^(k))",
        "temperature": temperature,
        "temperature_fit": temperature_fit,
        "mi_quantile": float(args.mi_quantile),
        "mi_q95_calibration": mi_q95,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "window_definition": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "center_offset": CENTER_OFFSET,
            "max_adjacent_gap_sec": 0.5,
            "max_window_span_sec": 2.5,
        },
        "counts": {
            "calibration_windows": int(len(calibration_rows)),
            "pseudo_target_windows": int(len(pseudo_safe_rows)),
            "expected_pseudo_target_windows": expected_pseudo,
        },
        "outputs": {
            "calibration_predictions": teacher_root / "calibration_predictions.parquet",
            "pseudo_predictions": teacher_root / "pseudo_predictions.parquet",
            "pseudo_predictions_audit": teacher_root / "pseudo_predictions_audit.parquet",
            "calibration_mc_fused_samples": teacher_root / "calibration_mc_fused_samples.npz",
            "mc_fused_samples": teacher_root / "mc_fused_samples.npz",
        },
    }
    write_json(teacher_root / "teacher_metadata.json", metadata)

    print(
        f"[DONE] fold={fold} teacher={teacher} T={temperature:.6g} "
        f"mi_q95={mi_q95:.6g} pseudo_rows={len(pseudo_safe_rows)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=sorted(FOLDS))
    parser.add_argument("--teacher", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/stgcn++/stgcn++_radarv4/rerun/pseudo_labeling/inner_teachers"),
    )
    parser.add_argument(
        "--inner-teacher-work-root",
        type=Path,
        default=Path("work_dirs/rerun/pseudo_labeling/inner_teachers"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/radar_v4/rerun/yolo26xpose/pseudo_labels_v1"),
    )
    parser.add_argument("--num-passes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic-cudnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sanity-check", action="store_true")
    parser.add_argument("--sanity-atol", type=float, default=1e-7)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--save-stream-pass-probabilities", action="store_true")
    parser.add_argument("--write-csv-copy", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--temperature-max-iter", type=int, default=100)
    parser.add_argument("--temperature-lr", type=float, default=0.1)
    parser.add_argument("--mi-quantile", type=float, default=0.95)
    parser.add_argument(
        "--allow-missing-boundary-audit",
        action="store_true",
        help="Allow audit boundary-distance fields to be null if source JSONLs are unavailable.",
    )
    args = parser.parse_args()
    if args.num_passes < 2:
        raise ValueError("--num-passes must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.temperature_max_iter <= 0:
        raise ValueError("--temperature-max-iter must be positive")
    if args.temperature_lr <= 0:
        raise ValueError("--temperature-lr must be positive")
    if not (0.0 < args.mi_quantile < 1.0):
        raise ValueError("--mi-quantile must be between 0 and 1")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        import torch

        if args.num_threads > 0:
            torch.set_num_threads(args.num_threads)
        run_teacher(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
