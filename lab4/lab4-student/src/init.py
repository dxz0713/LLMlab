from __future__ import annotations

import numpy as np


def kaiming(fan_in, fan_out, rng):
    """He/Kaiming normal initialization for a dense weight matrix.

    Each output unit is a dot product over fan_in inputs, so sampling from
    N(0, 2 / fan_in) keeps the variance of activations stable across ReLU
    layers. This is the same rule as lab1's Linear layers and lab2's
    he_conv with kernel size 1.

    Returns an array of shape (fan_in, fan_out).
    """
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))
