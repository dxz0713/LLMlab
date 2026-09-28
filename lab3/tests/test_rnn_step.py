import numpy as np

from src.rnn_layers import rnn_step_backward, rnn_step_forward
from tests.gradient_check import numeric_gradient, rel_error


def random_step_inputs(rng, N=3, D=4, H=5):
    x = rng.randn(N, D)
    prev_h = rng.randn(N, H)
    Wx = rng.randn(D, H) * 0.5
    Wh = rng.randn(H, H) * 0.5
    b = rng.randn(H) * 0.1
    return x, prev_h, Wx, Wh, b


def test_rnn_step_forward_hand_computed():
    N, D, H = 2, 3, 4
    x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)
    prev_h = np.linspace(-0.2, 0.5, num=N * H).reshape(N, H)
    Wx = np.linspace(-0.1, 0.9, num=D * H).reshape(D, H)
    Wh = np.linspace(-0.3, 0.7, num=H * H).reshape(H, H)
    b = np.linspace(-0.2, 0.4, num=H)

    next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    expected = np.array(
        [
            [-0.06892086, 0.06843830, 0.20325090, 0.33077389],
            [0.54621417, 0.77678477, 0.89793473, 0.95499608],
        ]
    )
    assert next_h.shape == (N, H)
    assert np.allclose(next_h, expected, atol=1e-8)


def test_rnn_step_forward_output_is_bounded():
    """tanh keeps every hidden unit strictly inside (-1, 1), however large the input."""
    rng = np.random.RandomState(0)
    x, prev_h, Wx, Wh, b = random_step_inputs(rng)
    next_h, _ = rnn_step_forward(100.0 * x, prev_h, Wx, Wh, b)
    assert next_h.shape == prev_h.shape
    assert np.all(np.abs(next_h) <= 1.0)


def test_rnn_step_backward_gradients():
    rng = np.random.RandomState(1)
    x, prev_h, Wx, Wh, b = random_step_inputs(rng)
    next_h, cache = rnn_step_forward(x, prev_h, Wx, Wh, b)
    dnext_h = rng.randn(*next_h.shape)

    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
    assert dx.shape == x.shape and dprev_h.shape == prev_h.shape
    assert dWx.shape == Wx.shape and dWh.shape == Wh.shape and db.shape == b.shape

    def f():
        return float(np.sum(rnn_step_forward(x, prev_h, Wx, Wh, b)[0] * dnext_h))

    for analytic, param in [(dx, x), (dprev_h, prev_h), (dWx, Wx), (dWh, Wh), (db, b)]:
        assert rel_error(analytic, numeric_gradient(f, param)) < 1e-7


def test_rnn_step_saturated_units_pass_no_gradient():
    """Where tanh is flat (|pre-activation| large), almost no gradient gets through.

    This is the local mechanism behind vanishing gradients in Task 2.
    """
    x = np.array([[1.0]])
    prev_h = np.zeros((1, 2))
    Wx = np.array([[20.0, 0.1]])  # unit 0 saturates, unit 1 stays in the linear range
    Wh = np.zeros((2, 2))
    b = np.zeros(2)
    _, cache = rnn_step_forward(x, prev_h, Wx, Wh, b)

    _, _, dWx, _, _ = rnn_step_backward(np.ones((1, 2)), cache)
    assert abs(dWx[0, 0]) < 1e-12
    assert abs(dWx[0, 1] - (1.0 - np.tanh(0.1) ** 2)) < 1e-12
