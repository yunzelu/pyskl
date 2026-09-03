"""Utilities for Monte Carlo dropout inference."""

from __future__ import annotations

from typing import Any


def _torch_modules() -> tuple[Any, Any, tuple[type, ...], tuple[type, ...]]:
    import torch.nn as nn
    from torch.nn.modules.dropout import _DropoutNd

    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        _DropoutNd,
    )
    batchnorm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    return nn, _DropoutNd, dropout_types, batchnorm_types


def is_dropout_module(module: Any) -> bool:
    """Return True for PyTorch dropout modules and subclasses."""
    _, _, dropout_types, _ = _torch_modules()
    return isinstance(module, dropout_types)


def is_batchnorm_module(module: Any) -> bool:
    """Return True for PyTorch BatchNorm modules."""
    _, _, _, batchnorm_types = _torch_modules()
    return isinstance(module, batchnorm_types)


def dropout_modules(model: Any) -> list[tuple[str, Any]]:
    """List named dropout modules in ``model``."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if is_dropout_module(module)
    ]


def batchnorm_modules(model: Any) -> list[tuple[str, Any]]:
    """List named BatchNorm modules in ``model``."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if is_batchnorm_module(module)
    ]


def unwrap_model(model: Any) -> Any:
    """Remove a common parallel wrapper when present."""
    return model.module if hasattr(model, "module") else model


def verify_gcn_head_dropout(
    model: Any,
    expected_dropout: float = 0.5,
    expected_num_classes: int = 9,
) -> dict[str, Any]:
    """Verify the E2 ST-GCN++ classifier head and report dropout modules."""
    nn, _, dropout_types, _ = _torch_modules()
    core = unwrap_model(model)
    if not hasattr(core, "cls_head"):
        raise AssertionError("Model does not expose cls_head")

    head = core.cls_head
    dropout = getattr(head, "dropout", None)
    fc_cls = getattr(head, "fc_cls", None)
    dropout_ratio = float(getattr(head, "dropout_ratio"))

    print("Head:", head)
    print("Configured dropout ratio:", dropout_ratio)
    print("Dropout module:", dropout)
    print("Final classifier:", fc_cls)

    assert dropout is not None
    assert isinstance(dropout, nn.Dropout)
    assert abs(float(dropout.p) - float(expected_dropout)) < 1e-8
    assert isinstance(fc_cls, nn.Linear)
    assert int(fc_cls.out_features) == int(expected_num_classes)

    dropout_modules_info = []
    print("\nAll dropout modules in the model:")
    for name, module in core.named_modules():
        if isinstance(module, dropout_types):
            probability = float(getattr(module, "p", 0.0))
            print(name, type(module).__name__, probability)
            dropout_modules_info.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "p": probability,
                }
            )

    return {
        "head": repr(head),
        "dropout_ratio": dropout_ratio,
        "dropout": repr(dropout),
        "fc_cls": repr(fc_cls),
        "dropout_modules": dropout_modules_info,
        "expected_dropout": float(expected_dropout),
        "expected_num_classes": int(expected_num_classes),
    }


def enable_head_mc_dropout(
    model: Any,
    expected_dropout: float = 0.5,
    expected_num_classes: int = 9,
) -> dict[str, Any]:
    """Enable only ``cls_head.dropout`` for MC-dropout inference.

    This is intentionally narrower than ``enable_mc_dropout``. E2 uses
    classifier-head dropout only; BatchNorm and all other modules remain in
    evaluation mode.
    """
    nn, _, dropout_types, batchnorm_types = _torch_modules()
    core = unwrap_model(model)
    model.eval()

    if not hasattr(core, "cls_head"):
        raise AssertionError("Model does not expose cls_head")
    head = core.cls_head
    dropout = getattr(head, "dropout", None)
    fc_cls = getattr(head, "fc_cls", None)
    dropout_ratio = float(getattr(head, "dropout_ratio"))

    assert dropout is not None
    assert isinstance(dropout, nn.Dropout)
    assert abs(float(dropout.p) - float(expected_dropout)) < 1e-8
    assert abs(dropout_ratio - float(expected_dropout)) < 1e-8
    assert isinstance(fc_cls, nn.Linear)
    assert int(fc_cls.out_features) == int(expected_num_classes)

    dropout.train()

    target_id = id(dropout)
    for name, module in core.named_modules():
        if id(module) == target_id:
            assert module.training, "cls_head.dropout is not in train mode"
            continue
        if isinstance(module, batchnorm_types):
            assert not module.training, f"BatchNorm module {name!r} is in train mode"
            continue
        if isinstance(module, dropout_types):
            assert not module.training, f"Non-head dropout module {name!r} is in train mode"
            continue
        if name != "":
            assert not module.training, f"Non-dropout module {name!r} is in train mode"

    return {
        "name": "cls_head.dropout",
        "type": type(dropout).__name__,
        "p": float(dropout.p),
        "mode": "train",
        "scope": "classifier_head_only",
    }


def enable_mc_dropout(model: Any) -> list[dict[str, Any]]:
    """Enable dropout stochasticity while keeping the rest of the model frozen.

    The function first calls ``model.eval()`` globally. It then switches only
    dropout modules back to train mode, preserving BatchNorm and all other
    modules in evaluation mode.

    Returns:
        A serializable list describing the dropout modules found.

    Raises:
        RuntimeError: if no dropout module with ``p > 0`` is present.
        AssertionError: if the resulting module modes are inconsistent.
    """
    model.eval()
    found = dropout_modules(model)
    searched = [
        "torch.nn.Dropout",
        "torch.nn.Dropout1d",
        "torch.nn.Dropout2d",
        "torch.nn.Dropout3d",
        "torch.nn.modules.dropout._DropoutNd",
    ]

    active = []
    for name, module in found:
        probability = float(getattr(module, "p", 0.0))
        if probability <= 0:
            continue
        module.train()
        active.append(
            {
                "name": name,
                "type": type(module).__name__,
                "p": probability,
            }
        )

    if not active:
        raise RuntimeError(
            "No active dropout module with p > 0 was found. "
            f"Searched for: {', '.join(searched)}"
        )

    assert_mc_dropout_state(model)
    return active


def assert_mc_dropout_state(model: Any) -> None:
    """Assert that only dropout modules are in train mode."""
    for name, module in dropout_modules(model):
        probability = float(getattr(module, "p", 0.0))
        if probability > 0:
            assert module.training, f"Dropout module {name!r} is not in train mode"

    for name, module in batchnorm_modules(model):
        assert not module.training, f"BatchNorm module {name!r} is in train mode"

    for name, module in model.named_modules():
        if name == "":
            continue
        if is_dropout_module(module):
            continue
        assert not module.training, f"Non-dropout module {name!r} is in train mode"
