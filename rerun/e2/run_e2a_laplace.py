"""E2A last-layer Laplace validation inference for E1-B checkpoints.

This script fits one last-layer Laplace approximation for each selected E1-B
ST-GCN++ checkpoint and evaluates the raw Laplace predictive branch on the
validation subject.

Protocol:
- checkpoint source: E1-B continuous-window models selected by validation
  center macro-F1;
- fit split: that fold's outer training subjects;
- fit loader: every unique training window once, deterministic preprocessing,
  no random augmentation, no square-root sampler, no replacement;
- probabilistic parameters: only ``cls_head.fc_cls``;
- dropout: disabled throughout fitting and prediction;
- approximation: last-layer Laplace, kron Hessian, Curvlinops GGN backend;
- prior precision: one scalar per fold/stream, optimized by marginal
  likelihood;
- prediction: 30 posterior probability samples by default, ``pred_type='nn'``;
- stream fusion: equal probability fusion for each posterior sample;
- evaluation split: validation subject only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
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
LAST_LAYER_NAME = "cls_head.fc_cls"
PKL_PROTOCOL_DIR = "continuous_window_w60_s12"
PKL_STEM = "radarv4_yolo26xpose_continuous_window_w60_s12_fold_{fold}.pkl"
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


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def accepts_keyword(func: Any, name: str) -> bool:
    signature = inspect.signature(func)
    parameters = signature.parameters
    if name in parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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


def e1_stream_prediction_path(e1_work_root: Path, fold: str, stream: str) -> Path:
    return (
        e1_work_root
        / f"fold_{fold}"
        / stream
        / CONDITION_DIR
        / "validation"
        / "best_pred.pkl"
    )


def e1_fusion_prediction_path(e1_work_root: Path, fold: str) -> Path:
    return (
        e1_work_root
        / f"fold_{fold}"
        / "fusion"
        / CONDITION_DIR
        / "validation"
        / "best_pred.pkl"
    )


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
        duplicates = len(keys) - len(set(keys))
        raise ValueError(f"{name} contains {duplicates} duplicate sample IDs")


def split_annotations(pkl_file: Path, split_name: str) -> list[dict[str, Any]]:
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


def load_probability_array(path: Path, name: str, atol: float = 1e-5) -> np.ndarray:
    values = np.asarray(load_pickle(path), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} expected [N, C], got {values.shape}")
    row_sums = values.sum(axis=1)
    if (
        np.all(values >= -atol)
        and np.all(values <= 1.0 + atol)
        and np.allclose(row_sums, 1.0, atol=atol, rtol=0.0)
    ):
        validate_probabilities(values, name, atol=atol)
        return values

    stable = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(stable)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    validate_probabilities(probabilities, f"{name} after softmax", atol=atol)
    return probabilities


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


def deterministic_fusion_probabilities(
    e1_work_root: Path,
    fold: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    fusion_path = e1_fusion_prediction_path(e1_work_root, fold)
    if fusion_path.exists():
        probabilities = load_probability_array(fusion_path, f"fold {fold} E1 fusion")
        return probabilities, {
            "source": "existing_e1_validation_fusion",
            "path": str(fusion_path),
        }

    stream_probabilities = []
    paths = []
    for stream in STREAMS:
        path = e1_stream_prediction_path(e1_work_root, fold, stream)
        if not path.exists():
            raise FileNotFoundError(f"Missing E1 stream prediction: {path}")
        stream_probabilities.append(load_probability_array(path, f"fold {fold} {stream} E1 stream"))
        paths.append(str(path))
    if stream_probabilities[0].shape != stream_probabilities[1].shape:
        raise ValueError(f"Joint/bone E1 validation prediction shapes differ for fold {fold}")
    probabilities = 0.5 * (stream_probabilities[0] + stream_probabilities[1])
    validate_probabilities(probabilities, f"fold {fold} deterministic fusion")
    return probabilities, {
        "source": "computed_from_e1_validation_streams",
        "paths": paths,
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


def laplace_quantities(fused_samples: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(fused_samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected Laplace probabilities [S, N, C], got {values.shape}")
    validate_probabilities(values, "fused Laplace posterior probability samples")
    mean_probabilities = values.mean(axis=0)
    validate_probabilities(mean_probabilities, "Laplace predictive mean")
    predictive_entropy = categorical_entropy(mean_probabilities)
    expected_entropy = categorical_entropy(values).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    if not np.all(np.isfinite(mutual_information)):
        raise ValueError("Laplace mutual information contains NaN or Inf")
    upper = np.log(len(LABELS)) + 1e-6
    if np.any(mutual_information > upper):
        raise ValueError("Laplace mutual information exceeds log(num_classes)")
    return {
        "mean_probabilities": mean_probabilities.astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "expected_entropy": expected_entropy.astype(np.float32),
        "mutual_information": mutual_information.astype(np.float32),
        "prediction": np.argmax(mean_probabilities, axis=1).astype(np.int64),
    }


def build_laplace_logit_wrapper(recognizer: Any):
    import torch
    import torch.nn as nn

    class PysklGCNLogitWrapper(nn.Module):
        """Return deterministic raw logits from a trained PYSKL RecognizerGCN."""

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            core = model.module if hasattr(model, "module") else model
            self.backbone = core.backbone
            self.cls_head = core.cls_head

        def train(self, mode: bool = True) -> "PysklGCNLogitWrapper":
            # Last-layer Laplace uses a deterministic MAP network. Ignore
            # attempts to enable training mode so dropout/BatchNorm stay off.
            super().train(False)
            return self

        def forward(self, keypoint: torch.Tensor) -> torch.Tensor:
            if keypoint.ndim == 6:
                if keypoint.shape[1] != 1:
                    raise ValueError(
                        "Last-layer Laplace expects one deterministic clip "
                        f"per window, but received {keypoint.shape[1]} clips."
                    )
                keypoint = keypoint[:, 0]

            if keypoint.ndim != 5:
                raise ValueError(
                    "Expected [B, M, T, V, C], but received "
                    f"{tuple(keypoint.shape)}."
                )

            features = self.backbone(keypoint)
            logits = self.cls_head(features)
            if logits.ndim != 2 or logits.shape[1] != len(LABELS):
                raise RuntimeError(
                    f"Expected logits [B, {len(LABELS)}], received {tuple(logits.shape)}."
                )
            return logits

    wrapper = PysklGCNLogitWrapper(recognizer)
    wrapper.eval()
    return wrapper


def verify_laplace_ready(wrapper: Any, expected_in_features: int | None) -> dict[str, Any]:
    import torch.nn as nn
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.dropout import _DropoutNd

    wrapper.eval()
    modules = dict(wrapper.named_modules())
    if LAST_LAYER_NAME not in modules:
        available = sorted(name for name in modules if name)
        raise AssertionError(
            f"Expected final-layer module {LAST_LAYER_NAME!r}. "
            f"Available modules include: {available[:30]}"
        )

    last_layer = modules[LAST_LAYER_NAME]
    dropout = getattr(wrapper.cls_head, "dropout", None)
    dropout_ratio = float(getattr(wrapper.cls_head, "dropout_ratio"))

    assert isinstance(last_layer, nn.Linear)
    assert int(last_layer.out_features) == len(LABELS)
    if expected_in_features is not None:
        assert int(last_layer.in_features) == int(expected_in_features)
    assert dropout is not None
    assert isinstance(dropout, nn.Dropout)
    assert abs(float(dropout.p) - 0.5) < 1e-8
    assert abs(dropout_ratio - 0.5) < 1e-8
    assert not dropout.training

    active_stochastic_modules = []
    for name, module in wrapper.named_modules():
        if isinstance(module, (_DropoutNd, _BatchNorm)) and module.training:
            active_stochastic_modules.append(name)
    if active_stochastic_modules:
        raise AssertionError(
            "Dropout or BatchNorm is active during Laplace: "
            f"{active_stochastic_modules}"
        )

    return {
        "last_layer_name": LAST_LAYER_NAME,
        "last_layer_repr": repr(last_layer),
        "last_layer_in_features": int(last_layer.in_features),
        "last_layer_out_features": int(last_layer.out_features),
        "dropout_repr": repr(dropout),
        "dropout_ratio": dropout_ratio,
        "dropout_p": float(dropout.p),
        "dropout_training": bool(dropout.training),
        "active_stochastic_modules": active_stochastic_modules,
    }


def unwrap_value(value: Any) -> Any:
    import torch

    if not torch.is_tensor(value) and hasattr(value, "data"):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


class LaplaceDatasetAdapter:
    """Adapt a PYSKL dataset to ``(keypoint, label)`` batches for Laplace."""

    def __init__(self, pyskl_dataset: Any) -> None:
        self.pyskl_dataset = pyskl_dataset

    def __len__(self) -> int:
        return len(self.pyskl_dataset)

    def __getitem__(self, index: int):
        import torch

        item = self.pyskl_dataset[index]
        keypoint = torch.as_tensor(unwrap_value(item["keypoint"]), dtype=torch.float32)
        label = torch.as_tensor(unwrap_value(item["label"]), dtype=torch.long).reshape(-1)[0]
        return keypoint, label


def build_loader(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        LaplaceDatasetAdapter(dataset),
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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


def load_model_and_datasets(config_path: Path, checkpoint_path: Path, args: argparse.Namespace):
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from pyskl.datasets import build_dataset
    from pyskl.models import build_model
    from pyskl.utils.mc_dropout import verify_gcn_head_dropout

    cfg = Config.fromfile(str(config_path))
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None

    train_dataset = build_dataset(deterministic_dataset_cfg(cfg, "train"), dict(test_mode=True))
    eval_dataset = build_dataset(deterministic_dataset_cfg(cfg, "val"), dict(test_mode=True))
    train_pipeline = assert_deterministic_pipeline(train_dataset, "Laplace train dataset")
    eval_pipeline = assert_deterministic_pipeline(eval_dataset, "Laplace validation dataset")

    assert_unique_sample_ids(sample_ids_from_dataset(train_dataset), "Laplace train dataset")
    assert_unique_sample_ids(sample_ids_from_dataset(eval_dataset), "Laplace validation dataset")

    recognizer = build_model(cfg.model)
    load_checkpoint(recognizer, str(checkpoint_path), map_location="cpu")
    head_verification = verify_gcn_head_dropout(
        recognizer,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )
    wrapper = build_laplace_logit_wrapper(recognizer)
    laplace_verification = verify_laplace_ready(
        wrapper,
        expected_in_features=args.expected_in_features,
    )

    return {
        "cfg": cfg,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "wrapper": wrapper,
        "head_verification": head_verification,
        "laplace_verification": laplace_verification,
        "train_pipeline": train_pipeline,
        "eval_pipeline": eval_pipeline,
    }


def fit_laplace_model(
    wrapper: Any,
    train_loader: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    import torch

    try:
        from laplace import Laplace
        from laplace.curvature import CurvlinopsGGN
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "E2A Laplace fitting requires laplace-torch with the Curvlinops "
            "backend in the active Python environment. This script was written "
            "for laplace-torch==0.2.2.2."
        ) from exc

    wrapper.eval()
    verify_laplace_ready(wrapper, expected_in_features=args.expected_in_features)

    started = time.time()
    laplace_model = Laplace(
        model=wrapper,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
        backend=CurvlinopsGGN,
        last_layer_name=LAST_LAYER_NAME,
        prior_precision=float(args.init_prior_precision),
        temperature=1.0,
    )

    fit_started = time.time()
    if accepts_keyword(laplace_model.fit, "progress_bar"):
        laplace_model.fit(train_loader, progress_bar=args.progress_bar)
    else:
        laplace_model.fit(train_loader)
    fit_seconds = float(time.time() - fit_started)

    optimize_started = time.time()
    optimize_kwargs = dict(
        method="marglik",
        pred_type="nn",
        link_approx="mc",
        prior_structure="scalar",
        init_prior_prec=float(args.init_prior_precision),
        n_steps=int(args.prior_steps),
        lr=float(args.prior_lr),
        verbose=bool(args.verbose_prior),
    )
    optimize_call_kwargs = {
        key: value
        for key, value in optimize_kwargs.items()
        if accepts_keyword(laplace_model.optimize_prior_precision, key)
    }
    if accepts_keyword(laplace_model.optimize_prior_precision, "progress_bar"):
        optimize_call_kwargs["progress_bar"] = args.progress_bar
    laplace_model.optimize_prior_precision(**optimize_call_kwargs)
    optimize_seconds = float(time.time() - optimize_started)

    prior_precision = laplace_model.prior_precision.detach().cpu()
    selected_prior_precision = float(prior_precision.reshape(-1)[0].item())
    wrapper.eval()
    if hasattr(laplace_model, "model"):
        laplace_model.model.eval()

    return laplace_model, {
        "laplace_class": type(laplace_model).__name__,
        "backend": "CurvlinopsGGN",
        "hessian_structure": "kron",
        "subset_of_weights": "last_layer",
        "last_layer_name": LAST_LAYER_NAME,
        "temperature": 1.0,
        "init_prior_precision": float(args.init_prior_precision),
        "selected_prior_precision": selected_prior_precision,
        "prior_precision_tensor": prior_precision.numpy(),
        "prior_optimization_method": "marglik",
        "prior_structure": "scalar",
        "prior_steps": int(args.prior_steps),
        "prior_lr": float(args.prior_lr),
        "fit_seconds": fit_seconds,
        "optimize_prior_seconds": optimize_seconds,
        "total_fit_seconds": float(time.time() - started),
        "torch_version": torch.__version__,
    }


@dataclass(frozen=True)
class PredictionResult:
    probabilities: np.ndarray
    labels: np.ndarray


def predictive_samples_with_generator(
    laplace_model: Any,
    keypoint: Any,
    num_samples: int,
    generator: Any,
) -> Any:
    kwargs = {
        "pred_type": "nn",
        "n_samples": num_samples,
    }
    if accepts_keyword(laplace_model.predictive_samples, "generator"):
        kwargs["generator"] = generator
    return laplace_model.predictive_samples(keypoint, **kwargs)


def predict_laplace_batches(
    laplace_model: Any,
    loader: Any,
    device: Any,
    num_samples: int,
    seed: int,
) -> PredictionResult:
    import torch

    if num_samples < 2:
        raise ValueError("Laplace posterior sampling requires at least two samples")

    model_device = device
    if hasattr(laplace_model, "model"):
        laplace_model.model.eval()
        try:
            model_device = next(laplace_model.model.parameters()).device
        except StopIteration:
            model_device = device

    try:
        generator = torch.Generator(device=model_device).manual_seed(seed)
    except RuntimeError:
        generator = torch.Generator(device=model_device.type).manual_seed(seed)

    probability_parts = []
    label_parts = []
    with torch.no_grad():
        for keypoint, labels in loader:
            keypoint = keypoint.to(
                device=device,
                dtype=torch.float32,
                non_blocking=(device.type == "cuda"),
            )
            samples = predictive_samples_with_generator(
                laplace_model=laplace_model,
                keypoint=keypoint,
                num_samples=num_samples,
                generator=generator,
            )
            if samples.ndim != 3:
                raise RuntimeError(f"Unexpected Laplace sample shape: {tuple(samples.shape)}")
            if samples.shape[0] != num_samples:
                raise RuntimeError(
                    f"Expected {num_samples} posterior samples, got {samples.shape[0]}"
                )
            if samples.shape[-1] != len(LABELS):
                raise RuntimeError(f"Expected {len(LABELS)} classes, got {samples.shape[-1]}")
            probability_parts.append(samples.detach().cpu())
            label_parts.append(labels.reshape(-1).cpu())

    probabilities = torch.cat(probability_parts, dim=1).numpy().astype(np.float32, copy=False)
    labels = torch.cat(label_parts, dim=0).numpy().astype(np.int64, copy=False)
    validate_probabilities(probabilities, "Laplace posterior probability samples")
    return PredictionResult(probabilities=probabilities, labels=labels)


def save_laplace_state(path: Path, laplace_model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": json_ready(metadata),
        "state_dict": laplace_model.state_dict(),
    }
    torch.save(payload, path)
    return {
        "saved": True,
        "path": str(path),
    }


def load_existing_stream_result(out_dir: Path, expected_samples: int) -> StreamResult:
    probabilities = np.load(out_dir / "laplace_prob_samples.npy")
    labels = np.load(out_dir / "labels.npy").astype(np.int64, copy=False)
    sample_ids = json.loads((out_dir / "sample_ids.json").read_text(encoding="utf-8"))
    if probabilities.shape[0] != expected_samples:
        raise ValueError(
            f"{out_dir / 'laplace_prob_samples.npy'} has {probabilities.shape[0]} "
            f"samples, expected {expected_samples}. Use --overwrite to regenerate."
        )
    validate_probabilities(probabilities, f"existing {out_dir} Laplace probabilities")
    return StreamResult(
        probabilities=probabilities,
        labels=labels,
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
        out_dir / "laplace_prob_samples.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
        out_dir / "metrics.json",
        out_dir / "fit_metadata.json",
    ]
    if all(path.exists() for path in expected_files) and not args.overwrite:
        print(f"[SKIP] existing Laplace stream outputs: {out_dir}")
        return load_existing_stream_result(out_dir, args.num_posterior_samples)

    config_path = validation_config_path(args.config_root, fold, stream)
    checkpoint_path = find_selected_checkpoint(args.e1_work_root, fold, stream)
    for path in [config_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        f"[INFO] E2A Laplace fold={fold} stream={stream} "
        f"checkpoint={checkpoint_path}"
    )
    loaded = load_model_and_datasets(config_path, checkpoint_path, args)
    wrapper = loaded["wrapper"].to(device)
    verify_laplace_ready(wrapper, expected_in_features=args.expected_in_features)

    train_dataset = loaded["train_dataset"]
    eval_dataset = loaded["eval_dataset"]
    train_sample_ids = sample_ids_from_dataset(train_dataset)
    eval_sample_ids = sample_ids_from_dataset(eval_dataset)
    train_labels = labels_from_dataset(train_dataset)
    eval_labels = labels_from_dataset(eval_dataset)

    train_loader = build_loader(
        train_dataset,
        batch_size=args.fit_batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = build_loader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    laplace_model, fit_metadata = fit_laplace_model(wrapper, train_loader, args)

    prediction_seed = args.posterior_seed_joint if stream == "joint" else args.posterior_seed_bone
    predict_started = time.time()
    prediction = predict_laplace_batches(
        laplace_model=laplace_model,
        loader=eval_loader,
        device=device,
        num_samples=args.num_posterior_samples,
        seed=prediction_seed,
    )
    prediction_seconds = float(time.time() - predict_started)

    if not np.array_equal(prediction.labels, eval_labels):
        raise RuntimeError(f"Collected labels do not match validation annotation order for {fold}/{stream}")

    mean_probabilities = prediction.probabilities.mean(axis=0)
    metrics = predictive_metrics(mean_probabilities, prediction.labels, args.ece_bins)
    metrics.update(
        {
            "branch": "laplace_stream_mean",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": stream,
            "split": "val",
            "num_posterior_samples": int(args.num_posterior_samples),
            "posterior_seed": int(prediction_seed),
            "selected_prior_precision": fit_metadata["selected_prior_precision"],
            "seconds_predict": prediction_seconds,
            "device": str(device),
            "fit_batch_size": int(args.fit_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "num_workers": int(args.num_workers),
            "num_fit_samples": int(len(train_dataset)),
            "num_validation_samples": int(len(eval_dataset)),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "laplace_prob_samples.npy", prediction.probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "laplace_mean_probabilities.npy", mean_probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", prediction.labels.astype(np.int64, copy=False))
    save_pickle(out_dir / "laplace_mean_pred.pkl", mean_probabilities.astype(np.float32, copy=False).tolist())
    write_json(out_dir / "sample_ids.json", eval_sample_ids)
    write_json(
        out_dir / "train_manifest.json",
        {
            "num_samples": int(len(train_dataset)),
            "class_counts": {
                LABELS[index]: int(np.count_nonzero(train_labels == index))
                for index in range(len(LABELS))
            },
            "sample_ids_head": train_sample_ids[:5],
            "sample_ids_tail": train_sample_ids[-5:],
        },
    )
    write_json(
        out_dir / "fit_metadata.json",
        {
            **fit_metadata,
            "fold": fold,
            "stream": stream,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "train_pipeline": loaded["train_pipeline"],
            "eval_pipeline": loaded["eval_pipeline"],
            "head_verification": loaded["head_verification"],
            "laplace_verification": loaded["laplace_verification"],
            "fit_data_rule": "outer training split, deterministic pipeline, each unique window once, no sampler",
            "dropout_rule": "wrapper stays in eval mode; cls_head.dropout disabled",
            "no_temperature_scaling": True,
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    if args.save_laplace_state:
        state_info = save_laplace_state(
            out_dir / "laplace_state_dict.pt",
            laplace_model,
            {
                "fold": fold,
                "stream": stream,
                "selected_prior_precision": fit_metadata["selected_prior_precision"],
                "checkpoint": str(checkpoint_path),
                "last_layer_name": LAST_LAYER_NAME,
            },
        )
        write_json(out_dir / "laplace_state_metadata.json", state_info)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[DONE] fold={fold} stream={stream} Laplace mean "
        f"acc={metrics['center_accuracy']:.4f} "
        f"macro_f1={metrics['center_macro_f1']:.4f} "
        f"prior={metrics['selected_prior_precision']:.6g}"
    )
    return StreamResult(
        probabilities=prediction.probabilities,
        labels=prediction.labels,
        sample_ids=eval_sample_ids,
        output_dir=out_dir,
    )


def load_stream_result(args: argparse.Namespace, fold: str, stream: str) -> StreamResult:
    out_dir = stream_output_dir(args.output_root, fold, stream)
    required = [
        out_dir / "laplace_prob_samples.npy",
        out_dir / "labels.npy",
        out_dir / "sample_ids.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Laplace stream output {path}. Run --mode stream first."
            )
    return load_existing_stream_result(out_dir, args.num_posterior_samples)


def load_stream_prior(out_dir: Path) -> float | None:
    path = out_dir / "fit_metadata.json"
    if not path.exists():
        return None
    metadata = load_json(path)
    return float(metadata["selected_prior_precision"])


def summarize_fusion_for_fold(args: argparse.Namespace, fold: str) -> dict[str, Any]:
    labels, expected_sample_ids = validation_labels_and_ids(args.data_root, fold)
    joint = load_stream_result(args, fold, "joint")
    bone = load_stream_result(args, fold, "bone")
    if joint.sample_ids != expected_sample_ids or bone.sample_ids != expected_sample_ids:
        raise ValueError(f"Joint/bone sample IDs do not match pkl validation order for fold {fold}")
    if not np.array_equal(joint.labels, labels) or not np.array_equal(bone.labels, labels):
        raise ValueError(f"Joint/bone labels do not match pkl validation labels for fold {fold}")
    if joint.sample_ids != bone.sample_ids:
        raise ValueError(f"Joint/bone sample IDs differ for fold {fold}")
    if not np.array_equal(joint.labels, bone.labels):
        raise ValueError(f"Joint/bone labels differ for fold {fold}")
    if joint.probabilities.shape != bone.probabilities.shape:
        raise ValueError(f"Joint/bone posterior sample shapes differ for fold {fold}")

    fused_samples = 0.5 * (joint.probabilities + bone.probabilities)
    validate_probabilities(fused_samples, f"fold {fold} fused Laplace posterior samples")
    quantities = laplace_quantities(fused_samples)
    mean_probabilities = quantities["mean_probabilities"]
    errors = quantities["prediction"].astype(np.int64) != labels

    out_dir = fusion_output_dir(args.output_root, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "laplace_prob_samples.npy", fused_samples.astype(np.float32, copy=False))
    np.save(out_dir / "laplace_mean_probabilities.npy", mean_probabilities.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels.astype(np.int64, copy=False))
    np.savez_compressed(
        out_dir / "laplace_quantities.npz",
        mean_probabilities=mean_probabilities.astype(np.float32, copy=False),
        predictive_entropy=quantities["predictive_entropy"].astype(np.float32, copy=False),
        expected_entropy=quantities["expected_entropy"].astype(np.float32, copy=False),
        mutual_information=quantities["mutual_information"].astype(np.float32, copy=False),
        prediction=quantities["prediction"].astype(np.int64, copy=False),
        label=labels.astype(np.int64, copy=False),
        error=errors.astype(np.bool_, copy=False),
    )
    save_pickle(out_dir / "laplace_mean_pred.pkl", mean_probabilities.astype(np.float32, copy=False).tolist())
    write_json(out_dir / "sample_ids.json", expected_sample_ids)

    joint_prior = load_stream_prior(joint.output_dir)
    bone_prior = load_stream_prior(bone.output_dir)
    metrics = predictive_metrics(mean_probabilities, labels, args.ece_bins)
    metrics.update(
        {
            "branch": "laplace",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": "fusion",
            "split": "val",
            "num_posterior_samples": int(args.num_posterior_samples),
            "probability_rule": "p_LA^(s) = 0.5 * (p_joint^(s) + p_bone^(s)); predictive mean over S posterior samples",
            "source_path": str(out_dir / "laplace_quantities.npz"),
            "mean_mutual_information": float(np.mean(quantities["mutual_information"])),
            "max_mutual_information": float(np.max(quantities["mutual_information"])),
            "mean_predictive_entropy": float(np.mean(quantities["predictive_entropy"])),
            "mean_expected_entropy": float(np.mean(quantities["expected_entropy"])),
            "joint_selected_prior_precision": "" if joint_prior is None else joint_prior,
            "bone_selected_prior_precision": "" if bone_prior is None else bone_prior,
        }
    )
    write_json(out_dir / "metrics.json", metrics)
    print(
        f"[DONE] fold={fold} fusion Laplace "
        f"acc={metrics['center_accuracy']:.4f} "
        f"macro_f1={metrics['center_macro_f1']:.4f} "
        f"nll={metrics['raw_nll']:.4f}"
    )
    return metrics


def deterministic_metrics_for_fold(args: argparse.Namespace, fold: str) -> dict[str, Any]:
    labels, _ = validation_labels_and_ids(args.data_root, fold)
    probabilities, source = deterministic_fusion_probabilities(args.e1_work_root, fold)
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Deterministic validation predictions for fold {fold} have "
            f"{probabilities.shape[0]} rows, expected {labels.shape[0]}"
        )
    metrics = predictive_metrics(probabilities, labels, args.ece_bins)
    metrics.update(
        {
            "branch": "deterministic",
            "condition": CONDITION_KEY,
            "fold": fold,
            "stream": "fusion",
            "split": "val",
            "num_posterior_samples": 1,
            "probability_rule": "p_det = 0.5 * (p_joint + p_bone)",
            "source": source["source"],
            "source_path": source.get("path", ";".join(source.get("paths", []))),
        }
    )
    return metrics


def mc_dropout_metrics_for_fold(args: argparse.Namespace, fold: str) -> dict[str, Any] | None:
    path = (
        args.mc_output_root
        / f"fold_{fold}"
        / "fusion"
        / CONDITION_DIR
        / "validation"
        / "metrics.json"
    )
    if not path.exists():
        return None
    row = load_json(path)
    if str(row.get("branch")) != "mc_dropout":
        row["branch"] = "mc_dropout"
    return row


def aggregate_mean_sd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["branch"]), []).append(row)

    summary_rows = []
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
        "Split: validation. Protocol: E1-B continuous windows. No temperature scaling.",
        "Laplace uses last-layer kron/Curvlinops approximations over cls_head.fc_cls.",
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


def write_combined_reports(args: argparse.Namespace, folds: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        rows.append(deterministic_metrics_for_fold(args, fold))
        mc_row = mc_dropout_metrics_for_fold(args, fold)
        if mc_row is not None:
            rows.append(mc_row)
        laplace_path = fusion_output_dir(args.output_root, fold) / "metrics.json"
        if laplace_path.exists():
            rows.append(load_json(laplace_path))

    if not rows:
        raise ValueError("No E2A rows were available for reporting")

    keyed = {}
    for row in rows:
        key = (
            str(row.get("branch", "")),
            str(row.get("fold", "")),
            str(row.get("stream", "fusion")),
            str(row.get("split", "val")),
        )
        keyed[key] = row
    rows = sorted(
        keyed.values(),
        key=lambda row: (str(row["branch"]), str(row["fold"]), str(row.get("stream", ""))),
    )
    summary_rows = aggregate_mean_sd(rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "e2a_raw_predictive_fold_metrics.csv", rows)
    write_csv(args.report_dir / "e2a_raw_predictive_mean_sd.csv", summary_rows)
    write_json(
        args.report_dir / "e2a_raw_predictive_summary.json",
        {
            "experiment": "E2A raw predictive-branch comparison",
            "split": "val",
            "condition": CONDITION_KEY,
            "included_branches": sorted({str(row["branch"]) for row in rows}),
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
                "raw_nll": "mean negative log likelihood of the raw predictive mean.",
                "raw_brier": "mean multiclass Brier score of the raw predictive mean.",
                "raw_ece": "top-label expected calibration error with fixed bins.",
            },
            "fold_metrics": rows,
            "mean_sd": summary_rows,
        },
    )
    (args.report_dir / "e2a_raw_predictive_summary.md").write_text(
        markdown_report(rows, summary_rows),
        encoding="utf-8",
        newline="\n",
    )
    print(f"[DONE] wrote combined E2A reports under {args.report_dir}")


def run(args: argparse.Namespace) -> None:
    if args.mode in {"all", "stream"}:
        try:
            import torch  # noqa: F401
            import mmcv  # noqa: F401
            import pyskl  # noqa: F401
            import laplace  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "E2A Laplace stream fitting requires torch, mmcv, pyskl, and "
                "laplace-torch in the active Python environment."
            ) from exc

    if args.num_posterior_samples < 2:
        raise ValueError("--num-posterior-samples must be at least 2")
    if args.fit_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.ece_bins <= 0:
        raise ValueError("--ece-bins must be positive")
    if args.expected_in_features is not None and args.expected_in_features <= 0:
        raise ValueError("--expected-in-features must be positive when provided")

    folds = [item.lower() for item in args.folds]
    streams = [item.lower() for item in args.streams]
    for fold in folds:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")
    for stream in streams:
        if stream not in STREAMS:
            raise ValueError(f"Unknown stream {stream!r}; expected one of {STREAMS}")

    if args.mode in {"all", "stream"}:
        import torch

        if args.num_threads > 0:
            torch.set_num_threads(args.num_threads)
        set_global_seed(args.seed, deterministic_cudnn=args.deterministic_cudnn)
        device = resolve_device(args.device)
        print(
            f"[INFO] mode={args.mode} device={device} seed={args.seed} "
            f"posterior_samples={args.num_posterior_samples}"
        )
        for fold in folds:
            for stream in streams:
                run_stream(args, fold, stream, device)

    if args.mode in {"all", "fusion"}:
        for fold in folds:
            summarize_fusion_for_fold(args, fold)
        write_combined_reports(args, folds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "stream", "fusion"],
        default="all",
        help="stream fits/predicts stream outputs; fusion reads saved streams and writes final reports.",
    )
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--streams", nargs="+", default=STREAMS)
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
        help="Root containing selected E1-B checkpoints and validation predictions.",
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
        default=Path("work_dirs/rerun/e2/e2a_laplace"),
        help="Root where Laplace stream/fusion artifacts are written.",
    )
    parser.add_argument(
        "--mc-output-root",
        type=Path,
        default=Path("work_dirs/rerun/e2/e2a_mc_dropout"),
        help="Root containing existing E2A MC-dropout fusion metrics.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("rerun/e2/reports"),
        help="Directory where combined E2A reports are written.",
    )
    parser.add_argument("--fit-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-posterior-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--posterior-seed-joint", type=int, default=42001)
    parser.add_argument("--posterior-seed-bone", type=int, default=42002)
    parser.add_argument("--init-prior-precision", type=float, default=1.0)
    parser.add_argument("--prior-steps", type=int, default=100)
    parser.add_argument("--prior-lr", type=float, default=0.1)
    parser.add_argument("--verbose-prior", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--deterministic-cudnn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-in-features",
        type=int,
        default=256,
        help="Expected cls_head.fc_cls input dimension. Use 0 to skip.",
    )
    parser.add_argument(
        "--save-laplace-state",
        action="store_true",
        help="Save laplace_model.state_dict() for each fold/stream.",
    )
    args = parser.parse_args()
    if args.expected_in_features == 0:
        args.expected_in_features = None
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
