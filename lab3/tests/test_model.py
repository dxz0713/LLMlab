import numpy as np
import pytest

from src.data import make_addition
from src.model import AdderRNN
from tests.gradient_check import numeric_gradient, rel_error


def test_params_shapes():
    net = AdderRNN(hidden_dim=16, seed=0)
    shapes = {name: p.shape for name, p in net.params.items()}
    assert shapes == {"Wx": (2, 16), "Wh": (16, 16), "b": (16,), "Wy": (16, 2), "by": (2,)}


def test_forward_scores_shape():
    net = AdderRNN(hidden_dim=8, seed=0)
    X, _ = make_addition(5, 7, np.random.default_rng(0))
    scores = net.loss(X)
    assert scores.shape == (5, 7, 2)


def test_same_parameters_handle_any_length():
    """Weights are shared across time, so one model reads sequences of any length."""
    net = AdderRNN(hidden_dim=8, seed=0)
    rng = np.random.default_rng(1)
    for T in (1, 8, 50):
        X, y = make_addition(3, T, rng)
        loss, grads = net.loss(X, y)
        assert np.isfinite(loss)
        assert {k: g.shape for k, g in grads.items()} == {k: p.shape for k, p in net.params.items()}


def test_initial_loss_is_about_log2():
    """An untrained model is close to guessing each bit, so the loss is near log 2."""
    net = AdderRNN(hidden_dim=16, seed=0)
    X, y = make_addition(2000, 8, np.random.default_rng(0))
    loss, _ = net.loss(X, y)
    assert loss == pytest.approx(np.log(2), abs=0.1)


def test_gradients_match_numeric():
    net = AdderRNN(hidden_dim=5, seed=3)
    for name in net.params:  # nonzero biases, so their gradients are tested fully
        net.params[name] = net.params[name] + 0.1
    X, y = make_addition(4, 6, np.random.default_rng(2))

    _, grads = net.loss(X, y)
    assert set(grads) == set(net.params)
    for name, param in net.params.items():
        num = numeric_gradient(lambda: net.loss(X, y)[0], param)
        assert rel_error(grads[name], num) < 1e-6, name


def test_predict_returns_bits():
    net = AdderRNN(hidden_dim=8, seed=0)
    X, _ = make_addition(10, 9, np.random.default_rng(3))
    pred = net.predict(X)
    assert pred.shape == (10, 9)
    assert np.issubdtype(pred.dtype, np.integer)
    assert set(np.unique(pred)) <= {0, 1}
    assert np.array_equal(pred, np.argmax(net.loss(X), axis=2))
