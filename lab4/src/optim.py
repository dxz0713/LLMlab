from __future__ import annotations

import numpy as np


def sgd(w, dw, config=None):
    """Vanilla stochastic gradient descent.

    config format:
    - learning_rate: scalar learning rate

    Returns a tuple of (next_w, config).
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    next_w = w - config["learning_rate"] * dw
    return next_w, config


def sgd_momentum(w, dw, config=None):
    """SGD with momentum.

    Keeps a velocity vector that accumulates an exponentially decaying sum of
    past gradients, which damps oscillation across steep directions and
    accelerates progress along shallow but consistent ones.

    config format:
    - learning_rate: scalar learning rate
    - momentum: scalar in [0, 1); 0 reduces this to plain sgd
    - velocity: array of the same shape as w, zero-initialized on first call

    Returns a tuple of (next_w, config).
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config


def adam(w, dw, config=None):
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
