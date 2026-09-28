import numpy as np
import pytest

from src.optim import clip_grad_norm
from src.rnn_layers import rnn_backward, rnn_forward, rnn_step_forward
from tests.gradient_check import numeric_gradient, rel_error


def random_sequence_inputs(rng, N=2, T=5, D=3, H=4):
    x = rng.randn(N, T, D)
    h0 = rng.randn(N, H)
    Wx = rng.randn(D, H) * 0.5
    Wh = rng.randn(H, H) * 0.5
    b = rng.randn(H) * 0.1
    return x, h0, Wx, Wh, b


def test_rnn_forward_shapes():
    rng = np.random.RandomState(0)
    for N, T, D, H in [(2, 5, 3, 4), (1, 1, 2, 6), (7, 12, 2, 3)]:
        x, h0, Wx, Wh, b = random_sequence_inputs(rng, N, T, D, H)
        h, _ = rnn_forward(x, h0, Wx, Wh, b)
        assert h.shape == (N, T, H)


def test_rnn_forward_matches_step_loop():
    """Unrolling is nothing more than calling the same step T times."""
    rng = np.random.RandomState(1)
    x, h0, Wx, Wh, b = random_sequence_inputs(rng)
    h, _ = rnn_forward(x, h0, Wx, Wh, b)

    prev_h = h0
    for t in range(x.shape[1]):
        prev_h, _ = rnn_step_forward(x[:, t], prev_h, Wx, Wh, b)
        assert np.allclose(h[:, t], prev_h, atol=1e-12)


def test_rnn_forward_is_causal():
    """h_t depends on x_1..x_t but never on later inputs."""
    rng = np.random.RandomState(2)
    x, h0, Wx, Wh, b = random_sequence_inputs(rng, T=6)
    h, _ = rnn_forward(x, h0, Wx, Wh, b)

    x_late = x.copy()
    x_late[:, 4] += 1.0
    h_late, _ = rnn_forward(x_late, h0, Wx, Wh, b)
    assert np.allclose(h_late[:, :4], h[:, :4])
    assert not np.allclose(h_late[:, 4], h[:, 4])

    x_early = x.copy()
    x_early[:, 0] += 1.0
    h_early, _ = rnn_forward(x_early, h0, Wx, Wh, b)
    assert not np.allclose(h_early[:, -1], h[:, -1])


def test_rnn_backward_gradients():
    rng = np.random.RandomState(3)
    x, h0, Wx, Wh, b = random_sequence_inputs(rng)
    h, cache = rnn_forward(x, h0, Wx, Wh, b)
    dh = rng.randn(*h.shape)

    dx, dh0, dWx, dWh, db = rnn_backward(dh, cache)
    assert dx.shape == x.shape and dh0.shape == h0.shape
    assert dWx.shape == Wx.shape and dWh.shape == Wh.shape and db.shape == b.shape

    def f():
        return float(np.sum(rnn_forward(x, h0, Wx, Wh, b)[0] * dh))

    for analytic, param in [(dx, x), (dh0, h0), (dWx, Wx), (dWh, Wh), (db, b)]:
        assert rel_error(analytic, numeric_gradient(f, param)) < 1e-7


def test_rnn_backward_gradient_flows_back_through_time():
    """With a loss on the last step only, earlier inputs must still get gradient.

    Everything that reaches x[:, 0] has to travel backwards through h_T,
    h_{T-1}, ..., h_1. If the gradient arriving from step t + 1 is not added
    in at step t, dx[:, 0] comes out as zero.
    """
    rng = np.random.RandomState(4)
    x, h0, Wx, Wh, b = random_sequence_inputs(rng, T=4)
    h, cache = rnn_forward(x, h0, Wx, Wh, b)
    dh = np.zeros_like(h)
    dh[:, -1] = rng.randn(*dh[:, -1].shape)

    dx, dh0, _, _, _ = rnn_backward(dh, cache)

    def f():
        return float(np.sum(rnn_forward(x, h0, Wx, Wh, b)[0] * dh))

    num_dx = numeric_gradient(f, x)
    assert np.max(np.abs(num_dx[:, 0])) > 1e-3
    assert rel_error(dx[:, 0], num_dx[:, 0]) < 1e-7
    assert rel_error(dh0, numeric_gradient(f, h0)) < 1e-7


def test_gradient_vanishes_over_long_sequences():
    """||dL/dh_0|| for a loss at the last step shrinks exponentially with T.

    Each step multiplies the gradient by the step's Jacobian, and with
    ordinary initial weights these products shrink fast.
    """
    rng = np.random.default_rng(3)
    H, D = 16, 2
    Wx = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, H))
    Wh = rng.normal(0.0, 1.0 / np.sqrt(H), size=(H, H))
    b = np.zeros(H)

    norms = []
    for T in (5, 10, 20, 40):
        x = np.random.default_rng(4).integers(0, 2, size=(32, T, D)).astype(np.float64)
        h, cache = rnn_forward(x, np.zeros((32, H)), Wx, Wh, b)
        dh = np.zeros_like(h)
        dh[:, -1] = 1.0
        _, dh0, _, _, _ = rnn_backward(dh, cache)
        norms.append(np.linalg.norm(dh0))

    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))
    assert norms[-1] / norms[0] < 1e-6


def test_clip_grad_norm_rescales_large_gradients():
    """The norm is taken over all parameters together, and all are scaled alike."""
    grads = {"a": np.array([3.0, 0.0]), "b": np.array([[0.0], [4.0]])}  # joint norm 5
    original = {k: v.copy() for k, v in grads.items()}

    norm = clip_grad_norm(grads, max_norm=2.5)

    assert norm == pytest.approx(5.0)
    assert np.allclose(grads["a"], 0.5 * original["a"])
    assert np.allclose(grads["b"], 0.5 * original["b"])
    assert grads["b"].shape == (2, 1)
    total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
    assert total == pytest.approx(2.5)


def test_clip_grad_norm_leaves_small_gradients_alone():
    rng = np.random.RandomState(5)
    grads = {"Wx": rng.randn(2, 3) * 0.1, "b": rng.randn(3) * 0.1}
    original = {k: v.copy() for k, v in grads.items()}
    expected_norm = np.sqrt(sum(np.sum(g * g) for g in original.values()))

    norm = clip_grad_norm(grads, max_norm=10.0)

    assert norm == pytest.approx(expected_norm)
    for k in grads:
        assert np.array_equal(grads[k], original[k])
