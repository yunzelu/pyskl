"""Check the raw-logit wrapper needed for E2A last-layer Laplace.

This is an integration check only. It intentionally does not import
``laplace-torch``. For each E1-B fold/stream checkpoint, it:

1. loads the E1-B validation config and selected checkpoint;
2. verifies the ST-GCN++ head, dropout module, and final classifier layer;
3. wraps the PYSKL recognizer as ``logits = wrapper(keypoint)``;
4. compares ``softmax(logits)`` against the stored PYSKL validation
   probabilities in ``best_pred.pkl``.

Passing this check means the later Laplace fitting script can use
``cls_head.fc_cls`` as the probabilistic last layer without a PYSKL
``forward_test``/clip-averaging mismatch.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
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
CONDITION_DIR = "b_continuous_window"
LAST_LAYER_NAME = "cls_head.fc_cls"


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


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def validation_config_path(config_root: Path, fold: str, stream: str) -> Path:
    return config_root / f"fold_{fold}" / stream / "validation" / "b_validation.py"


def stored_prediction_path(e1_work_root: Path, fold: str, stream: str) -> Path:
    return (
        e1_work_root
        / f"fold_{fold}"
        / stream
        / CONDITION_DIR
        / "validation"
        / "best_pred.pkl"
    )


def find_selected_checkpoint(e1_work_root: Path, fold: str, stream: str) -> Path:
    checkpoint_dir = e1_work_root / f"fold_{fold}" / stream / CONDITION_DIR
    checkpoints = sorted(checkpoint_dir.glob("best_macro_f1_epoch_*.pth"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected one best_macro_f1 checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def load_stored_probabilities(path: Path, probability_atol: float) -> np.ndarray:
    values = np.asarray(load_pickle(path), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path} should contain [N, C] probabilities, got {values.shape}")
    if values.shape[1] != len(LABELS):
        raise ValueError(f"{path} should contain {len(LABELS)} classes, got {values.shape[1]}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains NaN or Inf")
    if np.any(values < -probability_atol) or np.any(values > 1.0 + probability_atol):
        raise ValueError(f"{path} is not a probability array; values are outside [0, 1]")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=probability_atol, rtol=0.0):
        max_error = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"{path} rows do not sum to 1; max error={max_error:.6g}")
    return values


def build_laplace_logit_wrapper(recognizer: Any):
    import torch
    import torch.nn as nn

    class PysklGCNLogitWrapper(nn.Module):
        """Return raw logits from a trained PYSKL RecognizerGCN."""

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            core = model.module if hasattr(model, "module") else model
            self.backbone = core.backbone
            self.cls_head = core.cls_head

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

            if logits.ndim != 2:
                raise RuntimeError(f"Expected 2D logits, received {tuple(logits.shape)}.")
            if logits.shape[1] != len(LABELS):
                raise RuntimeError(
                    f"Expected nine output classes, received {logits.shape[1]}."
                )
            return logits

    return PysklGCNLogitWrapper(recognizer)


def verify_laplace_ready(
    wrapper: Any,
    expected_dropout: float,
    expected_in_features: int | None,
) -> dict[str, Any]:
    import torch.nn as nn
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

    print("Laplace final-layer name:", LAST_LAYER_NAME)
    print("Laplace final layer:", last_layer)
    print("Head dropout after wrapper.eval():", dropout)
    print("Dropout training mode:", getattr(dropout, "training", None))

    assert isinstance(last_layer, nn.Linear)
    assert int(last_layer.out_features) == len(LABELS)
    if expected_in_features is not None:
        assert int(last_layer.in_features) == int(expected_in_features)
    assert dropout is not None
    assert isinstance(dropout, nn.Dropout)
    assert abs(float(dropout.p) - float(expected_dropout)) < 1e-8
    assert abs(dropout_ratio - float(expected_dropout)) < 1e-8
    assert not dropout.training

    active_dropout = []
    for name, module in wrapper.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, _DropoutNd)):
            if module.training:
                active_dropout.append(name)
    if active_dropout:
        raise AssertionError(f"Dropout is active during Laplace check: {active_dropout}")

    return {
        "last_layer_name": LAST_LAYER_NAME,
        "last_layer_repr": repr(last_layer),
        "last_layer_in_features": int(last_layer.in_features),
        "last_layer_out_features": int(last_layer.out_features),
        "dropout_repr": repr(dropout),
        "dropout_ratio": dropout_ratio,
        "dropout_p": float(dropout.p),
        "dropout_training": bool(dropout.training),
        "active_dropout_modules": active_dropout,
    }


def collate_first_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    keypoints = []
    labels = []
    for item in items:
        keypoint = item["keypoint"]
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint)
        keypoints.append(keypoint.float())
        labels.append(int(torch.as_tensor(item["label"]).reshape(-1)[0].item()))

    return {
        "keypoint": torch.stack(keypoints, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def first_batch(dataset: Any, num_samples: int) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")
    count = min(int(num_samples), len(dataset))
    return collate_first_batch([dataset[index] for index in range(count)])


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

    head_verification = verify_gcn_head_dropout(
        recognizer,
        expected_dropout=0.5,
        expected_num_classes=len(LABELS),
    )
    wrapper = build_laplace_logit_wrapper(recognizer)
    return cfg, dataset, recognizer, wrapper, head_verification


def batch_to_device(batch: dict[str, Any], device: Any):
    import torch

    return batch["keypoint"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=(device.type == "cuda"),
    )


def compare_wrapper_to_pyskl_forward(
    recognizer: Any,
    wrapper: Any,
    keypoint: Any,
) -> dict[str, Any]:
    import torch

    recognizer.eval()
    wrapper.eval()
    with torch.no_grad():
        wrapper_probabilities = torch.softmax(wrapper(keypoint), dim=-1).detach().cpu().numpy()
        pyskl_probabilities = recognizer(keypoint, return_loss=False)

    pyskl_probabilities = np.asarray(pyskl_probabilities, dtype=np.float64)
    if pyskl_probabilities.shape != wrapper_probabilities.shape:
        raise RuntimeError(
            "PYSKL forward_test and wrapper probability shapes differ: "
            f"{pyskl_probabilities.shape} vs {wrapper_probabilities.shape}"
        )

    diff = np.abs(wrapper_probabilities.astype(np.float64) - pyskl_probabilities)
    return {
        "pyskl_forward_test_max_abs_diff": float(np.max(diff)),
        "pyskl_forward_test_mean_abs_diff": float(np.mean(diff)),
    }


def run_one(
    args: argparse.Namespace,
    fold: str,
    stream: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    config_path = validation_config_path(args.config_root, fold, stream)
    checkpoint_path = find_selected_checkpoint(args.e1_work_root, fold, stream)
    prediction_path = stored_prediction_path(args.e1_work_root, fold, stream)

    for path in [config_path, checkpoint_path, prediction_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        f"[INFO] Laplace wrapper check fold={fold} stream={stream} "
        f"checkpoint={checkpoint_path}"
    )
    stored_probabilities = load_stored_probabilities(
        prediction_path,
        probability_atol=args.probability_atol,
    )
    cfg, dataset, recognizer, wrapper, head_verification = load_model_and_dataset(
        config_path,
        checkpoint_path,
    )

    if len(dataset) != stored_probabilities.shape[0]:
        raise ValueError(
            f"Validation dataset length {len(dataset)} does not match "
            f"{prediction_path} rows {stored_probabilities.shape[0]}"
        )

    laplace_verification = verify_laplace_ready(
        wrapper,
        expected_dropout=0.5,
        expected_in_features=args.expected_in_features,
    )

    batch = first_batch(dataset, args.num_check_samples)
    video_info_labels = np.asarray(
        [int(item["label"]) for item in dataset.video_infos[: batch["label"].shape[0]]],
        dtype=np.int64,
    )
    batch_labels = batch["label"].cpu().numpy().astype(np.int64)
    if not np.array_equal(batch_labels, video_info_labels):
        raise RuntimeError("Pipeline labels do not match validation annotation order.")

    recognizer.to(device)
    wrapper.to(device)
    keypoint = batch_to_device(batch, device)
    wrapper.eval()

    with torch.no_grad():
        logits = wrapper(keypoint)
        wrapper_probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    expected = stored_probabilities[: wrapper_probabilities.shape[0]]
    diff = np.abs(wrapper_probabilities.astype(np.float64) - expected)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    allclose = bool(
        np.allclose(
            wrapper_probabilities,
            expected,
            atol=args.atol,
            rtol=args.rtol,
        )
    )

    forward_test_check = compare_wrapper_to_pyskl_forward(recognizer, wrapper, keypoint)

    if not allclose and not args.allow_mismatch:
        worst_flat = int(np.argmax(diff))
        worst_sample, worst_class = np.unravel_index(worst_flat, diff.shape)
        raise RuntimeError(
            "Wrapper probabilities do not match stored PYSKL probabilities for "
            f"fold={fold} stream={stream}. Max abs diff={max_abs_diff:.6g} at "
            f"checked sample {worst_sample}, class {worst_class}; "
            f"atol={args.atol}, rtol={args.rtol}."
        )

    row = {
        "fold": fold,
        "stream": stream,
        "status": "passed" if allclose else "mismatch_allowed",
        "num_validation_samples": int(len(dataset)),
        "num_checked_samples": int(wrapper_probabilities.shape[0]),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "stored_prediction": str(prediction_path),
        "device": str(device),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "wrapper_probability_row_sum_min": float(wrapper_probabilities.sum(axis=1).min()),
        "wrapper_probability_row_sum_max": float(wrapper_probabilities.sum(axis=1).max()),
        "model_config_work_dir": str(getattr(cfg, "work_dir", "")),
        **forward_test_check,
        **laplace_verification,
    }

    out_dir = args.output_dir / f"fold_{fold}" / stream
    write_json(
        out_dir / "wrapper_check.json",
        {
            "protocol": "E2A last-layer Laplace wrapper integration check",
            "requires_laplace_torch": False,
            "row": row,
            "head_verification": head_verification,
            "laplace_verification": laplace_verification,
        },
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[DONE] fold={fold} stream={stream} "
        f"max_abs_diff={max_abs_diff:.6g} mean_abs_diff={mean_abs_diff:.6g}"
    )
    return row


def run(args: argparse.Namespace) -> None:
    try:
        import torch  # noqa: F401
        import mmcv  # noqa: F401
        import pyskl  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Laplace wrapper check requires torch, mmcv, and pyskl in the "
            "active Python environment. It does not require laplace-torch."
        ) from exc

    if args.num_check_samples <= 0:
        raise ValueError("--num-check-samples must be positive")
    if args.expected_in_features is not None and args.expected_in_features <= 0:
        raise ValueError("--expected-in-features must be positive when provided")

    device = resolve_device(args.device)
    rows = []
    for fold in [item.lower() for item in args.folds]:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")
        for stream in [item.lower() for item in args.streams]:
            if stream not in STREAMS:
                raise ValueError(f"Unknown stream {stream!r}; expected one of {STREAMS}")
            rows.append(run_one(args, fold, stream, device))

    write_csv(args.output_dir / "check_results.csv", rows)
    write_json(
        args.output_dir / "check_summary.json",
        {
            "experiment": "E2A raw Laplace integration preparation",
            "check": "PYSKL raw-logit wrapper versus stored validation probabilities",
            "requires_laplace_torch": False,
            "condition": "E1-B continuous-window validation",
            "folds": [item.lower() for item in args.folds],
            "streams": [item.lower() for item in args.streams],
            "num_checks": len(rows),
            "all_passed": all(row["status"] == "passed" for row in rows),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "rows": rows,
        },
    )
    print(f"[DONE] wrote Laplace wrapper check reports under {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output-dir",
        type=Path,
        default=Path("rerun/e2/reports/laplace_wrapper_check"),
        help="Directory where check CSV/JSON outputs are written.",
    )
    parser.add_argument(
        "--num-check-samples",
        type=int,
        default=128,
        help="Number of leading validation samples to compare per fold/stream.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for wrapper-vs-stored probability comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for wrapper-vs-stored probability comparison.",
    )
    parser.add_argument(
        "--probability-atol",
        type=float,
        default=1e-4,
        help="Tolerance used to validate stored probability row sums.",
    )
    parser.add_argument(
        "--expected-in-features",
        type=int,
        default=256,
        help="Expected cls_head.fc_cls input dimension. Use 0 to skip.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Write diagnostics instead of failing when wrapper probabilities mismatch.",
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
