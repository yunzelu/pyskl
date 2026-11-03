"""Load the four CTR-GCN streams and return probabilities in canonical order.

The checkpoint label order differs from the order used by the existing
inference CSV and evaluation code. Every stream is reordered before fusion.
Heavy PYSKL/MMCV/PyTorch imports are deferred until model initialization.
"""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np

STREAMS = ('j', 'b', 'jm', 'bm')
NUM_CLASSES = 9
WINDOW_SIZE = 20
NATIVE_LABELS = (
    'Falling', 'Lying-Stationary', 'Sit-Stationary',
    'Transition-LayBed-to-Sit', 'Transition-LayFloor-to-Stand',
    'Transition-Sit-to-LayBed', 'Transition-Sit-to-Stand',
    'Transition-Stand-to-Sit', 'Walking',
)
CANONICAL_LABELS = (
    'lie-stationary', 'sit-stationary', 'walk', 'fall',
    'transition-lie-to-sit', 'transition-lie-to-stand',
    'transition-sit-to-lie', 'transition-sit-to-stand',
    'transition-stand-to-sit',
)
# canonical index -> CTR-GCN native training label index
CANONICAL_FROM_NATIVE = (1, 2, 8, 0, 3, 4, 5, 6, 7)


def canonicalize(probabilities: np.ndarray) -> np.ndarray:
    """Reorder (N, 9) class probabilities into the existing CSV label order."""
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise ValueError(f'Expected (N, 9) model probabilities, got {values.shape}')
    return values[:, CANONICAL_FROM_NATIVE]


def _find_artifacts(model_root: Path, config_root: Path) -> list[dict]:
    result = []
    for stream in STREAMS:
        config = config_root / f'{stream}.py'
        # The supplied model tree contains this specific checkpoint per stream.
        checkpoint = model_root / stream / 'epoch_20.pth'
        for path in (config, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'Missing {stream} model artifact: {path}')
        result.append(dict(stream=stream, config=str(config.resolve()),
                           checkpoint=str(checkpoint.resolve()),
                           checkpoint_bytes=checkpoint.stat().st_size))
    return result


def _validate_config(config, stream: str, path: str) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f'Incompatible {stream} config {path}: {message}')
    model = config.get('model', {})
    backbone = model.get('backbone', {})
    require(config.get('stream') == stream, f'stream must be {stream!r}')
    require(model.get('type') == 'RecognizerGCN', 'model must be RecognizerGCN')
    require(backbone.get('type') == 'CTRGCN', 'backbone must be CTRGCN')
    require(backbone.get('num_person') == 1, 'num_person must be 1')
    require(backbone.get('in_channels') == 3, 'in_channels must be 3')
    require(backbone.get('graph_cfg', {}).get('layout') == 'coco', 'layout must be coco')
    require(model.get('cls_head', {}).get('num_classes') == 9, 'num_classes must be 9')
    pipeline = config.get('test_pipeline', [])
    require(bool(pipeline), 'test_pipeline is missing')
    features = [x for x in pipeline if x.get('type') == 'GenSkeFeat']
    require(len(features) == 1 and features[0].get('dataset') == 'coco'
            and features[0].get('feats') == [stream],
            f'test pipeline must generate COCO {stream} features')
    samples = [x for x in pipeline if x.get('type') == 'UniformSample']
    require(len(samples) == 1 and samples[0].get('clip_len') == 20
            and samples[0].get('num_clips', 1) == 1,
            'test pipeline must use UniformSample clip_len=20 and num_clips=1')


class CTRGCNPredictor:
    """Four streams, 20 frames, canonical output order, one softmax per stream."""

    def __init__(self, model_root: Path, config_root: Path, device: str = 'cpu',
                 batch_size: int = 128, num_threads: int = 0) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 0:
            raise ValueError('num_threads must be a nonnegative integer')
        self.batch_size = batch_size
        self.num_threads = num_threads
        self._artifacts = _find_artifacts(Path(model_root), Path(config_root))
        try:
            import torch
            import mmcv
            from mmcv import Config
            from mmcv.runner import load_checkpoint
            from pyskl.datasets.pipelines import Compose
            from pyskl.models import build_recognizer
        except (ImportError, AttributeError, AssertionError, OSError) as exc:
            raise RuntimeError('CTR-GCN inference needs the compatible PYSKL training '
                               'environment with PyTorch and MMCV 1.x. '
                               f'Original import error: {exc}') from exc
        self._torch = torch
        if num_threads:
            torch.set_num_threads(num_threads)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'Requested {device}, but CUDA is unavailable')
        self._versions = {'torch': torch.__version__, 'mmcv': mmcv.__version__}
        self._models, self._pipelines = {}, {}
        configs = []
        for artifact in self._artifacts:
            config = Config.fromfile(artifact['config'])
            _validate_config(config, artifact['stream'], artifact['config'])
            configs.append(config)
        for artifact, config in zip(self._artifacts, configs):
            stream = artifact['stream']
            model_config = copy.deepcopy(config.model)
            model_config.backbone.pretrained = None
            model_config.setdefault('test_cfg', {})['average_clips'] = 'prob'
            model_config['test_cfg']['feat_ext'] = False
            model_config['test_cfg']['score_ext'] = False
            model = build_recognizer(model_config)
            load_checkpoint(model, artifact['checkpoint'], map_location='cpu', strict=True)
            model.to(self.device)
            model.eval()
            self._models[stream] = model
            self._pipelines[stream] = Compose(copy.deepcopy(config.test_pipeline))

    def describe(self) -> dict:
        return dict(streams=copy.deepcopy(self._artifacts), device=str(self.device),
                    batch_size=self.batch_size, num_threads=self.num_threads,
                    average_clips='prob', versions=dict(self._versions),
                    native_label_order=list(NATIVE_LABELS),
                    output_label_order=list(CANONICAL_LABELS),
                    output_to_native_indices=list(CANONICAL_FROM_NATIVE),
                    label_mapping_note=('Native bed-specific lie-to-sit and sit-to-lie, '
                                        'and floor-specific lie-to-stand, are mapped to '
                                        'the corresponding broader legacy labels.'))

    def predict(self, windows: list[dict]) -> dict[str, np.ndarray]:
        torch = self._torch
        predictions = {}
        for stream in STREAMS:
            probabilities = np.empty((len(windows), NUM_CLASSES), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(windows), self.batch_size):
                    batch_windows = windows[start:start + self.batch_size]
                    tensors = []
                    for annotation in batch_windows:
                        processed = self._pipelines[stream](copy.deepcopy(annotation))
                        keypoint = processed['keypoint']
                        if tuple(keypoint.shape) != (1, 1, WINDOW_SIZE, 17, 3):
                            raise ValueError(f'{stream} pipeline produced unexpected keypoint shape '
                                             f'{tuple(keypoint.shape)}')
                        tensors.append(keypoint)
                    batch = torch.stack(tensors, dim=0).to(self.device)
                    output = np.asarray(self._models[stream](keypoint=batch, return_loss=False),
                                        dtype=np.float32)
                    if output.shape != (len(batch_windows), NUM_CLASSES):
                        raise ValueError(f'{stream} returned invalid shape: {output.shape}')
                    if (not np.isfinite(output).all() or (output < -1e-6).any()
                            or (output > 1 + 1e-6).any()
                            or not np.allclose(output.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)):
                        raise ValueError(f'{stream} returned invalid class probabilities')
                    probabilities[start:start + len(batch_windows)] = canonicalize(output)
            predictions[stream] = probabilities
        return predictions
