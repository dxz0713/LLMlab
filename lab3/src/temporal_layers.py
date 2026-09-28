from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
Cache = Any


def temporal_affine_forward(h: Array, W: Array, b: Array) -> tuple[Array, Cache]:
    """Apply the same affine layer to the hidden state at every time step.

    Args:
        h: hidden states, shape (N, T, H).
        W: weights, shape (H, M).
        b: bias, shape (M,).

    Returns:
        out: shape (N, T, M), with out[:, t] = h[:, t] @ W + b for every t.
        cache: values needed by temporal_affine_backward.
    """
    N, T, H = h.shape
    M = W.shape[1]
    # Reshape h to (N*T, H), compute affine, reshape back
    h_flat = h.reshape(N * T, H)
    out_flat = h_flat @ W + b  # (N*T, M)
    out = out_flat.reshape(N, T, M)
    cache = (h, W, b)
    return out, cache


def temporal_affine_backward(dout: Array, cache: Cache) -> tuple[Array, Array, Array]:
    """Backward pass for temporal_affine_forward.

    Args:
        dout: upstream gradient, shape (N, T, M).
        cache: the cache returned by temporal_affine_forward.

    Returns:
        dh: gradient w.r.t. h, shape (N, T, H).
        dW: gradient w.r.t. W, shape (H, M).
        db: gradient w.r.t. b, shape (M,).
    """
    h, W, b = cache
    N, T, H = h.shape
    M = W.shape[1]

    # Reshape for matmul
    dout_flat = dout.reshape(N * T, M)  # (N*T, M)
    h_flat = h.reshape(N * T, H)  # (N*T, H)

    # dh = dout @ W^T, reshaped back
    dh_flat = dout_flat @ W.T  # (N*T, H)
    dh = dh_flat.reshape(N, T, H)  # (N, T, H)

    # dW = h^T @ dout, summed across all N*T positions
    dW = h_flat.T @ dout_flat  # (H, M)

    # db = sum of dout across N*T positions
    db = np.sum(dout_flat, axis=0)  # (M,)

    return dh, dW, db


def temporal_softmax_loss(scores: Array, y: IntArray) -> tuple[float, Array]:
    """Softmax cross-entropy at every time step, averaged over all of them.

    Every one of the N * T positions is an independent classification
    problem, and the loss is the mean over all N * T of them -- not the sum
    over time. With this convention the loss of an untrained model is about
    log(C) no matter how long the sequences are.

    Args:
        scores: unnormalized class scores, shape (N, T, C).
        y: correct class index at every position, integer array of shape
            (N, T) with entries in [0, C).

    Returns:
        loss: the mean cross-entropy, a Python float.
        dscores: gradient of the loss w.r.t. scores, shape (N, T, C).
    """
    N, T, C = scores.shape

    # Numerical stability: subtract the max per position
    scores_max = np.max(scores, axis=2, keepdims=True)
    scores_stable = scores - scores_max

    # Compute stable softmax
    exp_scores = np.exp(scores_stable)
    sum_exp = np.sum(exp_scores, axis=2, keepdims=True)

    # Log-softmax via log-sum-exp trick to avoid log(0) from underflow
    log_softmax = scores_stable - np.log(sum_exp)

    # Loss: negative log-likelihood of correct class, averaged over N*T positions
    correct_log_probs = log_softmax[np.arange(N)[:, None, None],
                                     np.arange(T)[None, :, None],
                                     y[:, :, None]]
    loss = -float(np.mean(correct_log_probs))

    # Gradient: (softmax - one_hot) / (N*T)
    probs = exp_scores / sum_exp
    dscores = probs.copy()
    dscores[np.arange(N)[:, None, None],
            np.arange(T)[None, :, None],
            y[:, :, None]] -= 1.0
    dscores /= (N * T)

    return loss, dscores
