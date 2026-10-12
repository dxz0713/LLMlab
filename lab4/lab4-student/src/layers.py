from __future__ import annotations

import numpy as np


def affine_forward(x, w, b):
    """Forward pass for an affine (fully connected) layer.

    Inputs:
    - x: (N, d_1, ..., d_k), flattened internally to (N, D) with D = prod(d_i)
    - w: (D, M)
    - b: (M,)

    Returns a tuple of:
    - out: (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = x.reshape(N, -1) @ w + b
    return out, (x, w, b)


def affine_backward(dout, cache):
    """Backward pass for an affine layer.

    Inputs:
    - dout: (N, M) upstream gradient
    - cache: (x, w, b) from affine_forward

    Returns a tuple of:
    - dx: same shape as x
    - dw: (D, M)
    - db: (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    dx = (dout @ w.T).reshape(x.shape)
    dw = x.reshape(N, -1).T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """ReLU forward pass. Returns (out, cache)."""
    out = np.maximum(0.0, x)
    return out, x


def relu_backward(dout, cache):
    """ReLU backward pass. Gradient is zero where the input was non-positive."""
    x = cache
    return dout * (x > 0)


def softmax_loss(scores, y):
    """Softmax cross-entropy loss and its gradient w.r.t. scores.

    Inputs:
    - scores: (N, C)
    - y: (N,) integer labels in [0, C)

    Returns a tuple of:
    - loss: float
    - dscores: (N, C)
    """
    N = scores.shape[0]
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    logp = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    loss = float(-np.sum(logp[np.arange(N), y]) / N)
    dscores = np.exp(logp)
    dscores[np.arange(N), y] -= 1.0
    dscores /= N
    return loss, dscores
