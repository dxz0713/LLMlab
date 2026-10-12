import numpy as np

from src.gcn import aggregate
from src.graph import add_self_loops, make_adjacency, row_normalize
from src.sage import (
    max_aggregate,
    max_aggregate_backward,
    sage_conv_backward,
    sage_conv_forward,
)
from tests.gradient_check import numeric_gradient, rel_error


def _small_graph():
    # 0 - 1 - 2 and 1 - 3
    return make_adjacency(4, np.array([[0, 1], [1, 2], [1, 3]]))


def test_max_aggregate_hand_computed():
    A = _small_graph()
    X = np.array([[1.0, 5.0], [2.0, 1.0], [4.0, 3.0], [8.0, 0.0]])
    H, _ = max_aggregate(A, X)
    # node 1 sees {0, 2, 3}: element-wise max of (1,5), (4,3), (8,0)
    assert np.allclose(H[1], np.array([8.0, 5.0]))
    # node 0 sees {1}; node 2 sees {1}; node 3 sees {1}
    assert np.allclose(H[0], X[1])
    assert np.allclose(H[2], X[1])
    assert np.allclose(H[3], X[1])


def test_max_aggregate_isolated_node_zero():
    A = make_adjacency(3, np.array([[0, 1]]))
    X = np.array([[1.0], [2.0], [3.0]])
    H, _ = max_aggregate(A, X)
    assert np.allclose(H[2], 0.0)


def test_max_aggregate_backward_routes_to_argmax():
    A = _small_graph()
    X = np.array([[1.0, 5.0], [2.0, 1.0], [4.0, 3.0], [8.0, 0.0]])
    _, cache = max_aggregate(A, X)
    dH = np.zeros_like(X)
    dH[1] = 1.0  # gradient into node 1's aggregate only
    dX = max_aggregate_backward(dH, cache)
    # node 1's max came from node 3 (feature 0) and node 0 (feature 1)
    assert np.allclose(dX[3], np.array([1.0, 0.0]))
    assert np.allclose(dX[0], np.array([0.0, 1.0]))
    assert np.allclose(dX[1], 0.0)
    assert np.allclose(dX[2], 0.0)


def test_max_aggregate_backward_numeric():
    rng = np.random.RandomState(0)
    A = make_adjacency(6, rng.randint(0, 6, size=(10, 2)))
    X = rng.randn(6, 4)  # continuous values: ties have probability zero
    dH = rng.randn(6, 4)
    _, cache = max_aggregate(A, X)
    dX = max_aggregate_backward(dH, cache)
    f = lambda: float(np.sum(max_aggregate(A, X)[0] * dH))  # noqa: E731
    assert rel_error(dX, numeric_gradient(f, X)) < 1e-5


def test_sage_conv_forward_hand_computed():
    A = _small_graph()
    A_mean = row_normalize(A)
    X = np.array([[1.0], [2.0], [4.0], [8.0]])
    # W splits into self weight 2 and neighbour weight 3
    W = np.array([[2.0], [3.0]])
    b = np.array([1.0])
    out, _ = sage_conv_forward(A_mean, X, W, b, aggregator="mean")
    # node 0: self 1, mean of {1} = 2 -> 2*1 + 3*2 + 1 = 9
    # node 1: self 2, mean of {0,2,3} = 13/3 -> 4 + 13 + 1 = 18
    assert abs(out[0, 0] - 9.0) < 1e-12
    assert abs(out[1, 0] - 18.0) < 1e-12


def test_sage_conv_forward_shapes():
    rng = np.random.RandomState(1)
    A_mean = row_normalize(_small_graph())
    X = rng.randn(4, 5)
    W = rng.randn(10, 8)
    b = rng.randn(8)
    out, _ = sage_conv_forward(A_mean, X, W, b, aggregator="mean")
    assert out.shape == (4, 8)
    H_agg = A_mean @ X
    assert np.allclose(out, np.concatenate([X, H_agg], axis=1) @ W + b)


def test_sage_conv_equivalent_to_split_weights():
    """concat([X, AGG]) @ W == X @ W_top + AGG @ W_bot for W = [W_top; W_bot]."""
    rng = np.random.RandomState(2)
    A_mean = row_normalize(_small_graph())
    X = rng.randn(4, 3)
    W = rng.randn(6, 2)
    b = rng.randn(2)
    out, _ = sage_conv_forward(A_mean, X, W, b, aggregator="mean")
    manual = X @ W[:3] + (A_mean @ X) @ W[3:] + b
    assert np.allclose(out, manual)


def test_sage_conv_backward_numeric_mean():
    rng = np.random.RandomState(3)
    A_mean = row_normalize(_small_graph())
    X = rng.randn(4, 3)
    W = rng.randn(6, 2)
    b = rng.randn(2)
    dout = rng.randn(4, 2)
    _, cache = sage_conv_forward(A_mean, X, W, b, aggregator="mean")
    dX, dW, db = sage_conv_backward(dout, cache)
    f = lambda: float(np.sum(sage_conv_forward(A_mean, X, W, b, "mean")[0] * dout))  # noqa: E731
    assert rel_error(dX, numeric_gradient(f, X)) < 1e-6
    assert rel_error(dW, numeric_gradient(f, W)) < 1e-6
    assert rel_error(db, numeric_gradient(f, b)) < 1e-6


def test_sage_conv_backward_numeric_max():
    rng = np.random.RandomState(4)
    A = _small_graph()
    X = rng.randn(4, 3)  # continuous values: no max ties
    W = rng.randn(6, 2)
    b = rng.randn(2)
    dout = rng.randn(4, 2)
    _, cache = sage_conv_forward(A, X, W, b, aggregator="max")
    dX, dW, db = sage_conv_backward(dout, cache)
    f = lambda: float(np.sum(sage_conv_forward(A, X, W, b, "max")[0] * dout))  # noqa: E731
    assert rel_error(dX, numeric_gradient(f, X)) < 1e-5
    assert rel_error(dW, numeric_gradient(f, W)) < 1e-5
    assert rel_error(db, numeric_gradient(f, b)) < 1e-5


def test_sage_gradient_reaches_neighbours_and_self():
    """SAGE keeps a dedicated self path: gradient flows to self AND neighbours."""
    A = _small_graph()
    A_mean = row_normalize(A)
    rng = np.random.RandomState(5)
    X = rng.randn(4, 3)
    W = np.abs(rng.randn(6, 2)) + 0.5
    b = np.zeros(2)
    dout = np.zeros((4, 2))
    dout[0] = 1.0  # loss depends only on node 0's output
    _, cache = sage_conv_forward(A_mean, X, W, b, aggregator="mean")
    dX, _, _ = sage_conv_backward(dout, cache)
    # node 0's output depends on itself (concat) and on node 1 (edge 0-1)
    assert not np.allclose(dX[0], 0.0)
    assert not np.allclose(dX[1], 0.0)
    assert np.allclose(dX[2], 0.0)
    assert np.allclose(dX[3], 0.0)


def test_mean_aggregator_matches_row_normalized_aggregate():
    """SAGE's mean aggregator is exactly aggregate() with row_normalize(A)."""
    rng = np.random.RandomState(6)
    A = _small_graph()
    X = rng.randn(4, 3)
    H_agg, _ = aggregate(row_normalize(A), X)
    W = rng.randn(6, 2)
    b = rng.randn(2)
    out, _ = sage_conv_forward(row_normalize(A), X, W, b, aggregator="mean")
    assert np.allclose(out, np.concatenate([X, H_agg], axis=1) @ W + b)
    # and, crucially, the mean aggregator does NOT use self loops
    A_self = row_normalize(add_self_loops(A))
    out_self, _ = sage_conv_forward(A_self, X, W, b, aggregator="mean")
    assert not np.allclose(out, out_self)
