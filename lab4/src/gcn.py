from __future__ import annotations

import numpy as np


def aggregate(A_norm, X):
    """Graph propagation step: H_agg = A_norm @ X.

    Each node's new (pre-transform) representation is a weighted sum of its
    neighbours' rows, with weights given by the normalized adjacency matrix.
    This is the "message passing" half of a graph convolution and carries
    no parameters.

    Inputs:
    - A_norm: (N, N) normalized adjacency (e.g. symmetric_normalize(A + I))
    - X: (N, F_in) node features

    Returns a tuple of:
    - H_agg: (N, F_in)
    - cache: (A_norm, X)
    """
    raise NotImplementedError

    # return H_agg, (A_norm, X)


def aggregate_backward(dH_agg, cache):
    """Backward pass for aggregate.

    H_agg = A_norm @ X, so dX = A_norm^T @ dH_agg: gradients flow back
    along the same edges the messages came from. Note A_norm^T appears,
    not A_norm itself -- the two coincide only for symmetric normalization.

    Inputs:
    - dH_agg: (N, F_in) upstream gradient
    - cache: (A_norm, X) from aggregate

    Returns:
    - dX: (N, F_in)
    """
    A_norm, _ = cache
    raise NotImplementedError


def graph_conv_forward_naive(A_norm, X, W, b):
    """Direct graph convolution with explicit loops. Slow but obviously correct.

    This exists as a reference to check the matrix version against. Never
    use it for anything but small test inputs.

    out[i] = b + W^T ( sum_j A_norm[i, j] * X[j] )

    Inputs:
    - A_norm: (N, N) normalized adjacency
    - X: (N, F_in)
    - W: (F_in, F_out)
    - b: (F_out,)

    Returns a tuple of:
    - out: (N, F_out)
    - cache: (A_norm, X, W, b)
    """
    N, F_in = X.shape
    F_out = W.shape[1]
    raise NotImplementedError

    # return out, (A_norm, X, W, b)


def graph_conv_forward(A_norm, X, W, b):
    """Graph convolution as two matrix multiplies: out = A_norm @ X @ W + b.

    Numerically identical to graph_conv_forward_naive but orders of
    magnitude faster, because both the propagation and the feature
    transform are handed to optimized GEMM calls.

    Returns a tuple of:
    - out: (N, F_out)
    - cache: (A_norm, X, W, b, H_agg)
    """
    raise NotImplementedError

    # return out, (A_norm, X, W, b, H_agg)


def graph_conv_backward(dout, cache):
    """Backward pass for the graph convolution.

    The forward pass is affine in (H_agg, W, b), so the backward pass is
    the same trio of matrix multiplies as an affine layer, plus one extra
    propagation step that folds dH_agg back onto the nodes:
        dW = H_agg^T @ dout,  db = sum(dout),  dX = A_norm^T @ (dout @ W^T)

    Inputs:
    - dout: (N, F_out) upstream gradient
    - cache: from graph_conv_forward

    Returns a tuple of (dX, dW, db):
    - dX: (N, F_in)
    - dW: (F_in, F_out)
    - db: (F_out,)
    """
    A_norm, X, W, b, H_agg = cache
    raise NotImplementedError
