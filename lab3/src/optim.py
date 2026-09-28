from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def adam(w: Array, dw: Array, config: dict[str, Any] | None = None) -> tuple[Array, dict[str, Any]]:
    """Adam with bias correction.

    Maintains running averages of the gradient (first moment) and of its
    square (second moment). Dividing by the square root of the second moment
    gives every coordinate its own effective step size. Both averages start
    at zero and are therefore biased toward zero early on, which the
    correction terms compensate for.

    config format:
    - learning_rate: scalar learning rate
    - beta1, beta2: decay rates for the first and second moment
    - epsilon: added to the denominator for numerical stability
    - m, v: running averages, zero-initialized on first call
    - t: iteration counter, incremented before use

    Returns a tuple of (next_w, config).
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    beta1, beta2 = config["beta1"], config["beta2"]
    config["t"] += 1
    t = config["t"]

    config["m"] = beta1 * config["m"] + (1 - beta1) * dw
    config["v"] = beta2 * config["v"] + (1 - beta2) * dw * dw

    m_hat = config["m"] / (1 - beta1**t)
    v_hat = config["v"] / (1 - beta2**t)

    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])
    return next_w, config


def clip_grad_norm(grads: dict[str, Array], max_norm: float) -> float:
    """Rescale a set of gradients so that their combined L2 norm is at most max_norm.

    The norm is taken over all gradients together, as if every array in
    `grads` were flattened and concatenated into one long vector. If that
    norm exceeds max_norm, every array is multiplied by the same factor
    max_norm / norm, so the direction of the overall update is unchanged.
    Otherwise nothing is modified.

    The arrays are replaced in place in the dict, the way
    torch.nn.utils.clip_grad_norm_ modifies .grad in place.

    Args:
        grads: parameter name -> gradient array. Modified in place.
        max_norm: the largest norm allowed, a positive float. Pass np.inf to
            only measure the norm without clipping.

    Returns:
        The combined norm before clipping, a Python float.
    """
    # Compute global norm
    total_sq = 0.0
    for g in grads.values():
        total_sq += np.sum(g * g)
    total_norm = np.sqrt(total_sq)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in grads:
            grads[k] *= scale

    return float(total_norm)
