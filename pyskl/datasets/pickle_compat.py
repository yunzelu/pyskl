# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility helpers for loading annotation pickles."""

from __future__ import annotations

import importlib
import sys


def install_numpy_pickle_compat_aliases() -> None:
    """Allow NumPy 2 pickles to load under NumPy 1.x.

    NumPy 2 pickles ndarray internals through modules such as
    ``numpy._core.numeric``. NumPy 1.x exposes the same implementation under
    ``numpy.core.numeric``. Registering aliases before unpickling keeps dataset
    pickles portable across the two NumPy module layouts.
    """

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
