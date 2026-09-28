import numpy as np
import pytest

from src.temporal_layers import temporal_affine_backward, temporal_affine_forward, temporal_softmax_loss
from tests.gradient_check import numeric_gradient, rel_error


def test_temporal_affine_forward_matches_each_step():
    """The same W and b are applied independently at every time step."""
    rng = np.random.RandomState(0)
    N, T, H, M = 3, 5, 4, 2
    h, W, b = rng.randn(N, T, H), rng.randn(H, M), rng.randn(M)

    out, _ = temporal_affine_forward(h, W, b)

    assert out.shape == (N, T, M)
    for t in range(T):
        assert np.allclose(out[:, t], h[:, t] @ W + b)


def test_temporal_affine_backward_gradients():
    rng = np.random.RandomState(1)
    N, T, H, M = 2, 4, 3, 5
    h, W, b = rng.randn(N, T, H), rng.randn(H, M), rng.randn(M)
    out, cache = temporal_affine_forward(h, W, b)
    dout = rng.randn(*out.shape)

    dh, dW, db = temporal_affine_backward(dout, cache)
    assert dh.shape == h.shape and dW.shape == W.shape and db.shape == b.shape

    def f():
        return float(np.sum(temporal_affine_forward(h, W, b)[0] * dout))

    for analytic, param in [(dh, h), (dW, W), (db, b)]:
        assert rel_error(analytic, numeric_gradient(f, param)) < 1e-7


def test_temporal_softmax_loss_uniform_scores():
    N, T, C = 4, 6, 3
    y = np.arange(N * T).reshape(N, T) % C
    loss, dscores = temporal_softmax_loss(np.zeros((N, T, C)), y)
    assert isinstance(loss, float)
    assert loss == pytest.approx(np.log(C))
    assert dscores.shape == (N, T, C)
    assert np.allclose(dscores.sum(axis=2), 0.0, atol=1e-12)


def test_temporal_softmax_loss_averages_over_all_positions():
    """The loss is a mean over N * T positions, so it does not grow with T."""
    rng = np.random.RandomState(2)
    N, T, C = 3, 4, 2
    scores = rng.randn(N, T, C)
    y = rng.randint(0, C, size=(N, T))

    loss, _ = temporal_softmax_loss(scores, y)
    per_position = [
        temporal_softmax_loss(scores[i : i + 1, t : t + 1], y[i : i + 1, t : t + 1])[0]
        for i in range(N)
        for t in range(T)
    ]
    assert loss == pytest.approx(np.mean(per_position))

    doubled, _ = temporal_softmax_loss(np.concatenate([scores, scores], axis=1), np.concatenate([y, y], axis=1))
    assert doubled == pytest.approx(loss)


def test_temporal_softmax_loss_gradient():
    rng = np.random.RandomState(3)
    N, T, C = 3, 4, 5
    scores = rng.randn(N, T, C) * 2
    y = rng.randint(0, C, size=(N, T))
    _, dscores = temporal_softmax_loss(scores, y)
    num = numeric_gradient(lambda: temporal_softmax_loss(scores, y)[0], scores)
    assert rel_error(dscores, num) < 1e-7


def test_temporal_softmax_loss_numerical_stability():
    """Huge scores must neither overflow nor underflow."""
    scores = np.array([[[1e4, 1e4 - 1.0], [0.0, -900.0]]])  # shape (1, 2, 2)
    y = np.array([[0, 1]])
    loss, dscores = temporal_softmax_loss(scores, y)
    assert np.isfinite(loss) and np.all(np.isfinite(dscores))
    # position 0: -log sigmoid(1); position 1: the correct class is 900 below the other
    expected = (np.log1p(np.exp(-1.0)) + 900.0) / 2
    assert loss == pytest.approx(expected, rel=1e-9)
