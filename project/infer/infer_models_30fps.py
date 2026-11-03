"""Load the selected 30 fps joint and bone ST-GCN++ models.

This module intentionally imports the model runtime only when a predictor is
constructed, allowing CSV/window validation without PyTorch or MMCV.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

STREAMS = {"j": "joint", "b": "bone"}
NUM_CLASSES = 9
WINDOW_SIZE = 60


def find_artifacts(model_root: Path) -> list[dict]:
    """Require one best checkpoint and the saved config for each stream."""
    artifacts = []
    for stream, name in STREAMS.items():
        directory = Path(model_root) / name / "b_continuous_window"
        config = directory / "b_continuous_window.py"
        if not config.is_file():
            raise FileNotFoundError(f"Saved {name} config not found: {config}")
        checkpoints = sorted(directory.glob("best_macro_f1*.pth"))
        if len(checkpoints) != 1:
            found = ", ".join(p.name for p in checkpoints) or "none"
            raise ValueError(f"Expected exactly one best_macro_f1*.pth in {directory}; found {found}")
        checkpoint = checkpoints[0]
        artifacts.append({"stream": stream, "name": name, "config": str(config.resolve()),
                          "checkpoint": str(checkpoint.resolve()),
                          "checkpoint_bytes": checkpoint.stat().st_size})
    return artifacts


def validate_config(config, stream: str, path: str) -> None:
    """Reject artifacts incompatible with 60-frame COCO-17 inference."""
    def require(condition: bool, description: str) -> None:
        if not condition:
            raise ValueError(f"Incompatible {stream} config {path}: {description}")

    model = config.get("model", {})
    backbone = model.get("backbone", {})
    require(config.get("stream") == stream, f"stream must be {stream!r}")
    require(model.get("type") == "RecognizerGCN", "model must be RecognizerGCN")
    require(backbone.get("type") == "STGCN", "backbone must be STGCN")
    require(backbone.get("num_person") == 1, "num_person must be 1")
    require(backbone.get("in_channels") == 3, "in_channels must be 3")
    require(backbone.get("graph_cfg", {}).get("layout") == "coco", "layout must be coco")
    require(model.get("cls_head", {}).get("num_classes") == NUM_CLASSES, "num_classes must be 9")
    pipeline = config.get("test_pipeline", [])
    require(bool(pipeline), "test_pipeline is missing")
    feature_steps = [step for step in pipeline if step.get("type") == "GenSkeFeat"]
    require(len(feature_steps) == 1 and feature_steps[0].get("feats") == [stream]
            and feature_steps[0].get("dataset") == "coco",
            f"test_pipeline must generate only COCO {stream} features")
    sampling_steps = [step for step in pipeline if step.get("type") == "MonotonicUniformResample"]
    require(len(sampling_steps) == 1 and sampling_steps[0].get("clip_len") == WINDOW_SIZE,
            "test_pipeline must use MonotonicUniformResample with clip_len=60")


class TwoStreamPredictor:
    """Run each saved test pipeline and return probabilities for j and b."""

    def __init__(self, model_root: Path, device: str = "cpu", batch_size: int = 128,
                 num_threads: int = 4) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 0:
            raise ValueError("num_threads must be a nonnegative integer")
        self.batch_size = batch_size
        self.num_threads = num_threads
        self._artifacts = find_artifacts(Path(model_root))
        try:
            import torch
            import mmcv
            from mmcv import Config
            from mmcv.runner import load_checkpoint
            from pyskl.datasets.pipelines import Compose
            from pyskl.models import build_recognizer
        except (ImportError, AttributeError, AssertionError, OSError) as exc:
            raise RuntimeError(
                "Model inference requires the training environment with PyTorch and "
                "MMCV 1.x (including mmcv.Config and mmcv.runner). Activate the "
                f"repository environment; use validation-only without models. Original error: {exc}"
            ) from exc

        if num_threads:
            torch.set_num_threads(num_threads)
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        self._versions = {"torch": torch.__version__, "mmcv": mmcv.__version__}
        self._models = {}
        self._pipelines = {}
        configs = []
        for artifact in self._artifacts:
            config = Config.fromfile(artifact["config"])
            validate_config(config, artifact["stream"], artifact["config"])
            configs.append(config)
        for artifact, config in zip(self._artifacts, configs):
            stream = artifact["stream"]
            model_config = copy.deepcopy(config.model)
            model_config.backbone.pretrained = None
            model_config.setdefault("test_cfg", {})["average_clips"] = "prob"
            model_config["test_cfg"]["feat_ext"] = False
            model_config["test_cfg"]["score_ext"] = False
            model = build_recognizer(model_config)
            load_checkpoint(model, artifact["checkpoint"], map_location="cpu", strict=True)
            model.to(self.device)
            model.eval()
            self._models[stream] = model
            self._pipelines[stream] = Compose(copy.deepcopy(config.test_pipeline))

    def describe(self) -> dict:
        return {"streams": copy.deepcopy(self._artifacts), "device": str(self.device),
                "batch_size": self.batch_size, "num_threads": self.num_threads,
                "average_clips": "prob", "versions": dict(self._versions)}

    def predict(self, windows: list[dict]) -> dict[str, np.ndarray]:
        """Return probabilities without applying a second softmax."""
        torch = self._torch
        predictions = {}
        for stream in STREAMS:
            probabilities = np.empty((len(windows), NUM_CLASSES), dtype=np.float32)
            model = self._models[stream]
            pipeline = self._pipelines[stream]
            with torch.no_grad():
                for start in range(0, len(windows), self.batch_size):
                    batch_windows = windows[start:start + self.batch_size]
                    tensors = []
                    for annotation in batch_windows:
                        processed = pipeline(copy.deepcopy(annotation))
                        keypoint = processed["keypoint"]
                        expected = (1, 1, WINDOW_SIZE, 17, 3)
                        if tuple(keypoint.shape) != expected:
                            raise ValueError(f"{stream} test pipeline produced {tuple(keypoint.shape)}; expected {expected}")
                        tensors.append(keypoint)
                    batch = torch.stack(tensors, dim=0).to(self.device)
                    output = np.asarray(model(keypoint=batch, return_loss=False), dtype=np.float32)
                    expected = (len(batch_windows), NUM_CLASSES)
                    if output.shape != expected:
                        raise ValueError(f"{stream} returned {output.shape}; expected {expected}")
                    if (not np.isfinite(output).all() or (output < -1e-6).any()
                            or (output > 1 + 1e-6).any()
                            or not np.allclose(output.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)):
                        raise ValueError(f"{stream} returned invalid class probabilities")
                    probabilities[start:start + len(batch_windows)] = output
            predictions[stream] = probabilities
        return predictions
