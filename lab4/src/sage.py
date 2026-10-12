from __future__ import annotations

import numpy as np

from src.gcn import aggregate, aggregate_backward


def max_aggregate(A, X):
    """Max-pooling aggregator: element-wise max over each node's neighbours.

    Unlike mean/sum aggregation, max is permutation invariant AND keeps
    only the strongest signal per feature channel. A node with no
    neighbours gets an all-zero aggregate.

    Inputs:
    - A: (N, N) adjacency (no normalization; only the sparsity pattern
      matters)
    - X: (N, F) node features

    Returns a tuple of:
    - H_agg: (N, F)
    - cache: (A, X)
    """
    raise NotImplementedError


def max_aggregate_backward(dH_agg, cache):
    """Backward pass for max_aggregate.

    As in max pooling (lab2), the gradient of each output entry is routed
    to the neighbour(s) that attained the max; ties split the gradient
    evenly among the maximizers.

    Inputs:
    - dH_agg: (N, F) upstream gradient
    - cache: (A, X) from max_aggregate

    Returns:
    - dX: (N, F)
    """
    A, X = cache
    raise NotImplementedError


def sage_conv_forward(A_adj, X, W, b, aggregator="mean"):
    """GraphSAGE convolution: out = concat([X, AGG(X)]) @ W + b.

    Compared with GCN, the node's own features are concatenated with the
    aggregated neighbourhood instead of being mixed in through a self loop,
    so self-information and neighbour-information get separate weights.
    W therefore has shape (2 * F_in, F_out).

    Inputs:
    - A_adj: (N, N) adjacency for the aggregator -- row_normalize(A) for
      "mean" (NO self loops; the self part is the first half of the
      concat), the raw binary A for "max"
    - X: (N, F_in)
    - W: (2 * F_in, F_out)
    - b: (F_out,)
    - aggregator: "mean" or "max"

    Returns a tuple of:
    - out: (N, F_out)
    - cache: (A_adj, X, W, b, H_agg, agg_cache, aggregator)
    """
    raise NotImplementedError


def sage_conv_backward(dout, cache):
    """Backward pass for the GraphSAGE convolution.

    Inputs:
    - dout: (N, F_out) upstream gradient
    - cache: from sage_conv_forward

    Returns a tuple of (dX, dW, db):
    - dX: (N, F_in)
    - dW: (2 * F_in, F_out)
    - db: (F_out,)
    """
    A_adj, X, W, b, H_agg, agg_cache, aggregator = cache
    F_in = X.shape[1]
    raise NotImplementedError
