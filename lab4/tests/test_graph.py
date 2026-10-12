import numpy as np

from src.graph import (
    add_self_loops,
    degree_vector,
    dirichlet_energy,
    graph_laplacian,
    make_adjacency,
    propagate,
    row_normalize,
    symmetric_normalize,
)


def test_make_adjacency_symmetric_binary():
    edges = np.array([[0, 1], [1, 2], [3, 0]])
    A = make_adjacency(4, edges)
    assert A.shape == (4, 4)
    assert np.allclose(A, A.T)
    assert set(np.unique(A)) <= {0.0, 1.0}
    assert A[0, 1] == 1.0 and A[1, 2] == 1.0 and A[3, 0] == 1.0
    assert A[0, 2] == 0.0


def test_make_adjacency_dedupes_and_drops_self_loops():
    edges = np.array([[0, 1], [1, 0], [0, 1], [2, 2]])
    A = make_adjacency(3, edges)
    assert A[0, 1] == 1.0 and A[1, 0] == 1.0
    assert np.allclose(np.diag(A), 0.0)


def test_make_adjacency_empty():
    A = make_adjacency(5, np.zeros((0, 2), dtype=np.int64))
    assert A.shape == (5, 5)
    assert np.allclose(A, 0.0)


def test_add_self_loops_adds_identity():
    A = make_adjacency(3, np.array([[0, 1]]))
    At = add_self_loops(A)
    assert np.allclose(np.diag(At), 1.0)
    assert np.allclose(At, A + np.eye(3))
    # the input matrix must not be modified
    assert np.allclose(np.diag(A), 0.0)


def test_add_self_loops_preserves_edges():
    rng = np.random.RandomState(0)
    A = (rng.rand(6, 6) > 0.5).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    At = add_self_loops(A)
    off = At - np.eye(6)
    assert np.allclose(off, A)


def test_degree_vector_values():
    # path graph 0 - 1 - 2 plus isolated node 3
    A = make_adjacency(4, np.array([[0, 1], [1, 2]]))
    assert np.allclose(degree_vector(A), np.array([1.0, 2.0, 1.0, 0.0]))


def test_row_normalize_rows_sum_to_one():
    rng = np.random.RandomState(1)
    A = (rng.rand(8, 8) > 0.4).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    R = row_normalize(A)
    d = degree_vector(A)
    assert np.allclose(R.sum(axis=1)[d > 0], 1.0)


def test_row_normalize_isolated_node_stays_zero():
    A = make_adjacency(3, np.array([[0, 1]]))
    R = row_normalize(A)
    assert np.allclose(R[2], 0.0)


def test_symmetric_normalize_hand_computed():
    # path graph 0 - 1 - 2, degrees 1, 2, 1
    A = make_adjacency(3, np.array([[0, 1], [1, 2]]))
    S = symmetric_normalize(A)
    s = 1.0 / np.sqrt(2.0)
    expected = np.array([[0.0, s, 0.0], [s, 0.0, s], [0.0, s, 0.0]])
    assert np.allclose(S, expected)


def test_symmetric_normalize_is_symmetric_and_spectral_radius_at_most_one():
    rng = np.random.RandomState(2)
    A = (rng.rand(10, 10) > 0.5).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    S = symmetric_normalize(add_self_loops(A))
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert np.max(np.abs(eigvals)) <= 1.0 + 1e-9


def test_graph_laplacian_hand_computed():
    A = make_adjacency(3, np.array([[0, 1], [1, 2]]))
    L = graph_laplacian(A)
    expected = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    assert np.allclose(L, expected)


def test_graph_laplacian_quadratic_form():
    """x^T L x == sum over edges (x_i - x_j)^2 >= 0 for any signal x."""
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    A = make_adjacency(4, edges)
    L = graph_laplacian(A)
    rng = np.random.RandomState(3)
    x = rng.randn(4)
    direct = float(x @ L @ x)
    edge_sum = sum((x[i] - x[j]) ** 2 for i, j in edges)
    assert abs(direct - edge_sum) < 1e-10
    assert direct >= 0.0


def test_dirichlet_energy():
    # constant signal on any graph has zero energy
    A = make_adjacency(4, np.array([[0, 1], [1, 2], [2, 3]]))
    X_const = np.ones((4, 3))
    assert dirichlet_energy(X_const, A) < 1e-12

    # matches the edge-sum definition: tr(X^T L X) = sum_edges ||x_i - x_j||^2
    rng = np.random.RandomState(4)
    X = rng.randn(4, 5)
    edge_sum = sum(np.sum((X[i] - X[j]) ** 2) for i, j in [[0, 1], [1, 2], [2, 3]])
    assert abs(dirichlet_energy(X, A) - edge_sum) < 1e-10
    assert dirichlet_energy(X, A) >= 0.0


def test_propagate():
    A = make_adjacency(4, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    A_norm = symmetric_normalize(add_self_loops(A))
    rng = np.random.RandomState(5)
    X = rng.randn(4, 3)

    assert np.allclose(propagate(A_norm, X, 0), X)
    assert np.allclose(propagate(A_norm, X, 2), A_norm @ A_norm @ X)

    # repeated propagation smooths the signal: energy strictly decreases
    E0 = dirichlet_energy(X, A)
    E10 = dirichlet_energy(propagate(A_norm, X, 10), A)
    assert E10 < E0
