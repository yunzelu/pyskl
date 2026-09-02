"""Run E1 validation-subject inference locally on CPU.

This is a non-distributed replacement for the SLURM validation jobs. It uses
the generated validation configs, loads each fold/stream selected checkpoint,
and writes ``validation/best_pred.pkl`` plus ``validation/best_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FOLDS = ["a", "b", "c"]
STREAMS = ["joint", "bone"]


@dataclass(frozen=True)
class Condition:
    key: str
    result_dir: str
    checkpoint_result_dir: str


CONDITIONS = [
    Condition("a1", "a1_activity_aligned", "a1_activity_aligned"),
    Condition("a2", "a2_activity_checkpoint_on_continuous", "a1_activity_aligned"),
    Condition("b", "b_continuous_window", "b_continuous_window"),
    Condition("c", "c_triangular_temporal_composition", "c_triangular_temporal_composition"),
]


def condition_by_key() -> dict[str, Condition]:
    return {condition.key: condition for condition in CONDITIONS}


def validation_config_path(
    config_root: Path,
    fold: str,
    stream: str,
    condition: Condition,
) -> Path:
    return (
        config_root
        / f"fold_{fold}"
        / stream
        / "validation"
        / f"{condition.key}_validation.py"
    )


def find_selected_checkpoint(
    work_root: Path,
    fold: str,
    stream: str,
    condition: Condition,
) -> Path:
    checkpoint_dir = (
        work_root
        / f"fold_{fold}"
        / stream
        / condition.checkpoint_result_dir
    )
    checkpoints = sorted(checkpoint_dir.glob("best_macro_f1_epoch_*.pth"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(
            f"Expected one best_macro_f1 checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )
    return checkpoints[0]


def output_dir(
    work_root: Path,
    fold: str,
    stream: str,
    condition: Condition,
) -> Path:
    return work_root / f"fold_{fold}" / stream / condition.result_dir / "validation"


def scores_to_jsonable_eval(metrics: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def stack_keypoints(batch: list[dict[str, Any]]) -> torch.Tensor:
    import torch

    tensors = []
    for item in batch:
        keypoint = item["keypoint"]
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint)
        tensors.append(keypoint.float())
    return torch.stack(tensors, dim=0)


def iter_batches(dataset: Any, batch_size: int):
    batch: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        batch.append(dataset[index])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_one(
    config_path: Path,
    checkpoint_path: Path,
    pred_path: Path,
    eval_path: Path,
    batch_size: int,
) -> dict[str, float]:
    try:
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint

        from pyskl.datasets import build_dataset
        from pyskl.models import build_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CPU validation inference requires a local Python environment "
            "where torch, mmcv, and pyskl are importable."
        ) from exc

    cfg = Config.fromfile(str(config_path))
    cfg.data.test.test_mode = True
    if cfg.model.get("backbone") is not None:
        cfg.model.backbone.pretrained = None

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    model = build_model(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model.cpu()
    model.eval()

    outputs: list[np.ndarray] = []
    start_time = time.time()
    with torch.no_grad():
        for batch in iter_batches(dataset, batch_size=batch_size):
            keypoint = stack_keypoints(batch)
            scores = model(keypoint=keypoint, return_loss=False)
            scores = np.asarray(scores, dtype=np.float32)
            outputs.extend(scores)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("wb") as f:
        pickle.dump(
            np.asarray(outputs, dtype=np.float32).tolist(),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    eval_cfg = cfg.get("test_evaluation", cfg.get("evaluation", {})).copy()
    for key in [
        "interval",
        "tmpdir",
        "start",
        "save_best",
        "rule",
        "by_epoch",
        "broadcast_bn_buffers",
    ]:
        eval_cfg.pop(key, None)
    metrics = scores_to_jsonable_eval(dataset.evaluate(outputs, **eval_cfg))
    metrics.update(
        {
            "num_samples": float(len(dataset)),
            "batch_size": float(batch_size),
            "seconds": float(time.time() - start_time),
            "device": "cpu",
        }
    )
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--streams", nargs="+", default=STREAMS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[condition.key for condition in CONDITIONS],
        choices=[condition.key for condition in CONDITIONS],
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/stgcn++/stgcn++_radarv4/rerun/e1"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs/rerun/e1"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Torch CPU thread count. 0 keeps PyTorch's default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation best_pred.pkl/best_eval.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_threads > 0:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CPU validation inference requires torch to set --num-threads."
            ) from exc
        torch.set_num_threads(args.num_threads)

    conditions = condition_by_key()
    for fold in [item.lower() for item in args.folds]:
        if fold not in FOLDS:
            raise ValueError(f"Unknown fold {fold!r}; expected one of {FOLDS}")
        for stream in [item.lower() for item in args.streams]:
            if stream not in STREAMS:
                raise ValueError(f"Unknown stream {stream!r}; expected one of {STREAMS}")
            for condition_key in args.conditions:
                condition = conditions[condition_key]
                config_path = validation_config_path(
                    args.config_root, fold, stream, condition
                )
                checkpoint_path = find_selected_checkpoint(
                    args.work_root, fold, stream, condition
                )
                out_dir = output_dir(args.work_root, fold, stream, condition)
                pred_path = out_dir / "best_pred.pkl"
                eval_path = out_dir / "best_eval.json"
                if pred_path.exists() and eval_path.exists() and not args.overwrite:
                    print(
                        f"[SKIP] fold={fold} stream={stream} "
                        f"condition={condition.key}: validation outputs exist"
                    )
                    continue
                if not config_path.exists():
                    raise FileNotFoundError(config_path)

                print(
                    f"[INFO] fold={fold} stream={stream} condition={condition.key} "
                    f"checkpoint={checkpoint_path}"
                )
                metrics = run_one(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    pred_path=pred_path,
                    eval_path=eval_path,
                    batch_size=args.batch_size,
                )
                print(
                    f"[DONE] fold={fold} stream={stream} condition={condition.key} "
                    f"top1={metrics.get('top1_acc', float('nan')):.4f} "
                    f"macro_f1={metrics.get('macro_f1', float('nan')):.4f}"
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
