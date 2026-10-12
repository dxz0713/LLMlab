from __future__ import annotations

import numpy as np


def make_adjacency(num_nodes, edges):
    """Build a symmetric binary adjacency matrix from an edge list.

    Inputs:
    - num_nodes: number of nodes N
    - edges: (E, 2) integer array, each row an (i, j) pair

    The graph is treated as undirected: each edge sets both A[i, j] and
    A[j, i] to 1. Duplicate edges and self loops are ignored.

    Returns:
    - A: (N, N) float64 array with entries in {0, 1}, zero diagonal
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edges:
        if i != j:
            A[i, j] = 1
            A[j, i] = 1
    return A


def add_self_loops(A):
    """Return A_tilde = A + I: every node additionally points to itself.

    Without a self loop a node's own features never enter its aggregated
    representation -- the update would replace a node by a summary of its
    neighbourhood instead of mixing the two.

    Returns a new array of the same shape.
    """
    N = A.shape[0]
    A_tilde = A.copy()
    A_tilde[np.arange(N), np.arange(N)] = 1
    return A_tilde


def degree_vector(A):
    """Degree of every node: d_i = sum_j A[i, j].

    Returns:
    - d: (N,) float64
    """
    return np.sum(A, axis=1)


def row_normalize(A):
    """Row (random-walk) normalization: D^{-1} A.

    Each row sums to 1 (except all-zero rows of isolated nodes, which stay
    zero), so A @ X computes a plain average of neighbour features.

    Returns a new (N, N) float64 array.
    """
    d = degree_vector(A)
    d_inv = np.zeros_like(d)
    d_inv[d > 0] = 1.0 / d[d > 0]
    return d_inv[:, None] * A


def symmetric_normalize(A):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}.

    The normalization used by GCN. Neighbour contributions are weighted by
    1 / sqrt(d_i * d_j): high-degree nodes are damped from both sides, and
    the result stays symmetric so its spectral radius remains at most 1.

    Returns a new (N, N) float64 array.
    """
    d = degree_vector(A)
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def graph_laplacian(A):
    """Combinatorial graph Laplacian L = D - A.

    L is symmetric positive semidefinite: for any signal x on the nodes,
    x^T L x = sum_{(i,j) in E} (x_i - x_j)^2 measures how "unsmooth" x is
    over the graph. Its eigenvectors are the graph analogue of Fourier
    modes -- the spectral viewpoint from which graph convolutions are
    derived.

    Returns a new (N, N) float64 array.
    """
    d = degree_vector(A)
    D = np.diag(d)
    return D - A


def dirichlet_energy(X, A):
    """Dirichlet energy of node features: E(X) = (1/2) sum_ij A_ij ||x_i - x_j||^2.

    A scalar measuring how "unsmooth" the features are over the graph: it
    is 0 when every connected pair of nodes has identical features, and
    grows with the discrepancy along edges. Equivalently E(X) = tr(X^T L X)
    with L the graph Laplacian -- a handy way to compute it.

    It is the standard way to quantify over-smoothing: stacking graph
    convolutions drives E(H^(l)) toward 0 layer after layer.

    Inputs:
    - X: (N, F) node features
    - A: (N, N) adjacency

    Returns a float >= 0.
    """
    L = graph_laplacian(A)
    return np.trace(X.T @ L @ X)


def propagate(A_norm, X, k):
    """Propagate features through k hops: A_norm^k @ X.

    This is the "feature-only" view of a k-layer GCN with weights removed
    (cf. SGC, Wu et al. 2019): repeated aggregation without any learning.
    Useful for observing over-smoothing directly -- watch how
    dirichlet_energy decays as k grows.

    Inputs:
    - A_norm: (N, N) normalized adjacency
    - X: (N, F) node features
    - k: number of propagation steps (k = 0 returns X itself)

    Returns an (N, F) array.
    """
    out = X.copy()
    for _ in range(k):
        out = A_norm @ out
    return out
    
