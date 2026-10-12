from __future__ import annotations

from typing import Callable

import numpy as np


def rel_error(x, y) -> float:
    """Max relative error between two arrays."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y))))


def numeric_gradient(f: Callable[[], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of a scalar function f w.r.t. array x.

    f must read the current value of x, e.g. `lambda: loss_fn(x, y)`.
    x is modified in place during the sweep and restored afterwards.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        fxph = f()
        x[idx] = old - h
        fxmh = f()
        x[idx] = old
        grad[idx] = (fxph - fxmh) / (2 * h)
        it.iternext()
    return grad
