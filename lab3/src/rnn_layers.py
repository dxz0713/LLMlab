from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Whatever a forward function saves for its backward function. Callers treat
# it as opaque and only ever pass it straight back, so its contents are up to
# you.
Cache = Any


def rnn_step_forward(
    x: Array, prev_h: Array, Wx: Array, Wh: Array, b: Array
) -> tuple[Array, Cache]:
    """Run one time step of a vanilla (tanh) RNN.

    Computes next_h = tanh(x @ Wx + prev_h @ Wh + b) for a whole minibatch
    at once.

    Args:
        x: input at this time step, shape (N, D).
        prev_h: hidden state from the previous time step, shape (N, H).
        Wx: input-to-hidden weights, shape (D, H).
        Wh: hidden-to-hidden weights, shape (H, H).
        b: bias, shape (H,).

    Returns:
        next_h: the new hidden state, shape (N, H).
        cache: values needed by rnn_step_backward.
    """
    a = x @ Wx + prev_h @ Wh + b  # (N, H)
    next_h = np.tanh(a)  # (N, H)
    cache = (x, prev_h, Wx, Wh, a, next_h)
    return next_h, cache


def rnn_step_backward(
    dnext_h: Array, cache: Cache
) -> tuple[Array, Array, Array, Array, Array]:
    """Backward pass for one time step of a vanilla RNN.

    Args:
        dnext_h: gradient of the loss w.r.t. next_h, shape (N, H).
        cache: the cache returned by rnn_step_forward.

    Returns:
        dx: gradient w.r.t. x, shape (N, D).
        dprev_h: gradient w.r.t. prev_h, shape (N, H).
        dWx: gradient w.r.t. Wx, shape (D, H).
        dWh: gradient w.r.t. Wh, shape (H, H).
        db: gradient w.r.t. b, shape (H,).
    """
    x, prev_h, Wx, Wh, a, next_h = cache
    da = dnext_h * (1 - next_h ** 2)  # (N, H)
    dx = da @ Wx.T  # (N, D)
    dprev_h = da @ Wh.T  # (N, H)
    dWx = x.T @ da  # (D, H)
    dWh = prev_h.T @ da  # (H, H)
    db = np.sum(da, axis=0)  # (H,)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(
    x: Array, h0: Array, Wx: Array, Wh: Array, b: Array
) -> tuple[Array, Cache]:
    """Run a vanilla RNN over a whole sequence.

    The same Wx, Wh and b are used at every time step.

    Args:
        x: input sequence, shape (N, T, D). x[:, t] is the input at step t.
        h0: initial hidden state, shape (N, H).
        Wx: input-to-hidden weights, shape (D, H).
        Wh: hidden-to-hidden weights, shape (H, H).
        b: bias, shape (H,).

    Returns:
        h: hidden states for all time steps, shape (N, T, H). h[:, t] is the
            state right after reading x[:, t]; h0 itself is not included.
        cache: values needed by rnn_backward.
    """
    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N, T, H), dtype=np.float64)
    step_caches = []
    prev_h = h0
    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t], prev_h, Wx, Wh, b)
        h[:, t] = next_h
        prev_h = next_h
        step_caches.append(step_cache)
    cache = (x, h0, Wx, Wh, b, step_caches, h)
    return h, cache


def rnn_backward(dh: Array, cache: Cache) -> tuple[Array, Array, Array, Array, Array]:
    """Backpropagation through time for a vanilla RNN.

    Args:
        dh: upstream gradients of the loss w.r.t. every hidden state, shape
            (N, T, H). dh[:, t] holds only the gradient that reaches h[:, t]
            from the layers above it (for example the output layer at step t).
            The gradient that reaches h[:, t] through h[:, t + 1] is NOT
            included -- computing that is this function's job.
        cache: the cache returned by rnn_forward.

    Returns:
        dx: gradient w.r.t. the input sequence, shape (N, T, D).
        dh0: gradient w.r.t. the initial hidden state, shape (N, H).
        dWx: gradient w.r.t. Wx, shape (D, H).
        dWh: gradient w.r.t. Wh, shape (H, H).
        db: gradient w.r.t. b, shape (H,).
    """
    x, h0, Wx, Wh, b, step_caches, h = cache
    N, T, D = x.shape
    H = h0.shape[1]

    dx = np.zeros((N, T, D), dtype=np.float64)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    dprev_h = np.zeros((N, H), dtype=np.float64)  # gradient flowing from step t+1 to step t

    for t in reversed(range(T)):
        # Total gradient at h[:, t] = upstream dh[:, t] + dprev_h from next step
        dh_t = dh[:, t] + dprev_h

        # Get the cache for this step
        step_cache = step_caches[t]

        # Backward through this step
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh_t, step_cache)

        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h  # gradient w.r.t. h0 (flows from step 0)

    return dx, dh0, dWx, dWh, db
