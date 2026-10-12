import time

import numpy as np

from src.gcn import (
    aggregate,
    aggregate_backward,
    graph_conv_backward,
    graph_conv_forward,
    graph_conv_forward_naive,
)
from src.graph import add_self_loops, make_adjacency, row_normalize, symmetric_normalize
from tests.gradient_check import numeric_gradient, rel_error


def _small_graph():
    # 0 - 1 - 2 and 1 - 3
    return make_adjacency(4, np.array([[0, 1], [1, 2], [1, 3]]))


def test_aggregate_hand_computed():
    A = _small_graph()
    R = row_normalize(A)
    X = np.array([[1.0], [2.0], [4.0], [8.0]])
    H, _ = aggregate(R, X)
    # node 0: mean of {1} -> 2; node 1: mean of {0,2,3} -> (1+4+8)/3
    # node 2: mean of {1} -> 2; node 3: mean of {1} -> 2
    assert np.allclose(H, np.array([[2.0], [13.0 / 3.0], [2.0], [2.0]]))


def test_aggregate_matches_matmul():
    rng = np.random.RandomState(0)
    A = make_adjacency(6, rng.randint(0, 6, size=(9, 2)))
    A_norm = symmetric_normalize(add_self_loops(A))
    X = rng.randn(6, 5)
    H, cache = aggregate(A_norm, X)
    assert H.shape == (6, 5)
    assert np.allclose(H, A_norm @ X)
    assert len(cache) == 2


def test_aggregate_backward_numeric():
    rng = np.random.RandomState(1)
    A_norm = rng.randn(5, 5) * 0.1
    X = rng.randn(5, 3)
    dH = rng.randn(5, 3)
    _, cache = aggregate(A_norm, X)
    dX = aggregate_backward(dH, cache)
    f = lambda: float(np.sum(aggregate(A_norm, X)[0] * dH))  # noqa: E731
    assert rel_error(dX, numeric_gradient(f, X)) < 1e-7


def test_aggregate_backward_uses_transpose():
    """For an asymmetric A_norm the backward must use A_norm^T, not A_norm."""
    A_norm = np.zeros((3, 3))
    A_norm[0, 1] = 1.0  # node 0 aggregates from node 1 only
    X = np.eye(3)
    dH = np.ones((3, 3))
    _, cache = aggregate(A_norm, X)
    dX = aggregate_backward(dH, cache)
    # only row 1 of X feeds into the output (via node 0), so only node 1
    # receives gradient
    assert np.allclose(dX[0], 0.0)
    assert np.allclose(dX[1], dH[0])
    assert np.allclose(dX[2], 0.0)


def test_graph_conv_forward_naive_hand_computed():
    A = _small_graph()
    R = row_normalize(add_self_loops(A))
    X = np.array([[1.0], [2.0], [4.0], [8.0]])
    W = np.array([[2.0]])
    b = np.array([1.0])
    out, _ = graph_conv_forward_naive(R, X, W, b)
    # node 0: mean of {0,1} = 1.5 -> 2*1.5+1 = 4
    # node 1: mean of {0,1,2,3} = 3.75 -> 8.5
    # node 2: mean of {1,2} = 3 -> 7; node 3: mean of {1,3} = 5 -> 11
    assert np.allclose(out, np.array([[4.0], [8.5], [7.0], [11.0]]))


def test_graph_conv_forward_matches_naive():
    rng = np.random.RandomState(2)
    A = make_adjacency(7, rng.randint(0, 7, size=(12, 2)))
    A_norm = symmetric_normalize(add_self_loops(A))
    X = rng.randn(7, 4)
    W = rng.randn(4, 6)
    b = rng.randn(6)
    out_fast, _ = graph_conv_forward(A_norm, X, W, b)
    out_naive, _ = graph_conv_forward_naive(A_norm, X, W, b)
    assert np.allclose(out_fast, out_naive)


def test_graph_conv_forward_shapes():
    rng = np.random.RandomState(3)
    A_norm = symmetric_normalize(add_self_loops(_small_graph()))
    X = rng.randn(4, 5)
    W = rng.randn(5, 8)
    b = rng.randn(8)
    out, cache = graph_conv_forward(A_norm, X, W, b)
    assert out.shape == (4, 8)
    assert np.allclose(out, (A_norm @ X) @ W + b)


def test_graph_conv_bias_shifts_all_nodes():
    rng = np.random.RandomState(4)
    A_norm = row_normalize(add_self_loops(_small_graph()))
    X = rng.randn(4, 3)
    W = rng.randn(3, 2)
    b = rng.randn(2)
    out1, _ = graph_conv_forward(A_norm, X, W, b)
    out2, _ = graph_conv_forward(A_norm, X, W, b + 10.0)
    assert np.allclose(out2 - out1, 10.0)


def test_graph_conv_backward_numeric_dx():
    rng = np.random.RandomState(5)
    A_norm = symmetric_normalize(add_self_loops(_small_graph()))
    X = rng.randn(4, 3)
    W = rng.randn(3, 2)
    b = rng.randn(2)
    dout = rng.randn(4, 2)
    _, cache = graph_conv_forward(A_norm, X, W, b)
    dX, _, _ = graph_conv_backward(dout, cache)
    f = lambda: float(np.sum(graph_conv_forward(A_norm, X, W, b)[0] * dout))  # noqa: E731
    assert rel_error(dX, numeric_gradient(f, X)) < 1e-7


def test_graph_conv_backward_numeric_dw_db():
    rng = np.random.RandomState(6)
    A_norm = symmetric_normalize(add_self_loops(_small_graph()))
    X = rng.randn(4, 3)
    W = rng.randn(3, 2)
    b = rng.randn(2)
    dout = rng.randn(4, 2)
    _, cache = graph_conv_forward(A_norm, X, W, b)
    _, dW, db = graph_conv_backward(dout, cache)
    fW = lambda: float(np.sum(graph_conv_forward(A_norm, X, W, b)[0] * dout))  # noqa: E731
    fb = lambda: float(np.sum(graph_conv_forward(A_norm, X, W, b)[0] * dout))  # noqa: E731
    assert rel_error(dW, numeric_gradient(fW, W)) < 1e-7
    assert rel_error(db, numeric_gradient(fb, b)) < 1e-7


def test_graph_conv_gradient_reaches_neighbours():
    """Loss on one node's output sends gradient to exactly its closed neighbourhood."""
    A = _small_graph()
    A_norm = row_normalize(add_self_loops(A))
    X = np.random.RandomState(7).randn(4, 3)
    W = np.random.RandomState(8).randn(3, 2)
    b = np.zeros(2)
    dout = np.zeros((4, 2))
    dout[0] = 1.0  # loss depends only on node 0's output
    _, cache = graph_conv_forward(A_norm, X, W, b)
    dX, _, _ = graph_conv_backward(dout, cache)
    # node 0 sees {0, 1} (self loop + edge 0-1); nodes 2, 3 get no gradient
    assert not np.allclose(dX[0], 0.0)
    assert not np.allclose(dX[1], 0.0)
    assert np.allclose(dX[2], 0.0)
    assert np.allclose(dX[3], 0.0)


def test_two_hop_receptive_field():
    """Two stacked graph convs reach 2-hop neighbours."""
    A = _small_graph()
    A_norm = row_normalize(add_self_loops(A))
    rng = np.random.RandomState(9)
    X = rng.randn(4, 3)
    W1 = np.abs(rng.randn(3, 3)) + 0.5
    W2 = np.abs(rng.randn(3, 2)) + 0.5

    h, c1 = graph_conv_forward(A_norm, X, W1, np.zeros(3))
    out, c2 = graph_conv_forward(A_norm, h, W2, np.zeros(2))

    dout = np.zeros((4, 2))
    dout[2] = 1.0  # loss on node 2
    dh, _, _ = graph_conv_backward(dout, c2)
    dX, _, _ = graph_conv_backward(dh, c1)
    # node 2's 2-hop neighbourhood is {0,1,2,3}: node 2 <- {1,2} <- {0,1,2,3}
    assert np.all(~np.isclose(dX, 0.0))


def test_graph_conv_is_faster_than_naive():
    rng = np.random.RandomState(10)
    N = 80
    A = make_adjacency(N, rng.randint(0, N, size=(300, 2)))
    A_norm = symmetric_normalize(add_self_loops(A))
    X = rng.randn(N, 16)
    W = rng.randn(16, 12)
    b = rng.randn(12)

    t0 = time.perf_counter()
    graph_conv_forward_naive(A_norm, X, W, b)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    graph_conv_forward(A_norm, X, W, b)
    t_fast = time.perf_counter() - t0

    assert t_fast < t_naive
