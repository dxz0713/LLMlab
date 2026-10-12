import numpy as np
import pytest

from src.graph import add_self_loops, make_adjacency, row_normalize, symmetric_normalize
from src.layers import softmax_loss
from src.model import GraphNet, masked_accuracy, masked_softmax_loss
from tests.gradient_check import numeric_gradient, rel_error


def _small_setup(seed=0, layer_type="gcn", aggregator="mean"):
    """Two triangles connected by a bridge, 2 classes, 4-dim features."""
    A = make_adjacency(6, np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [2, 3]]))
    rng = np.random.RandomState(seed)
    X = rng.randn(6, 4)
    y = np.array([0, 0, 0, 1, 1, 1])
    if layer_type == "gcn":
        A_norm = symmetric_normalize(add_self_loops(A))
    else:
        A_norm = row_normalize(A) if aggregator == "mean" else A
    return A, A_norm, X, y


def test_graphnet_gcn_param_shapes():
    net = GraphNet(num_features=10, num_classes=3, hidden_dim=16, num_layers=2, layer_type="gcn")
    assert net.params["W0"].shape == (10, 16)
    assert net.params["b0"].shape == (16,)
    assert net.params["W1"].shape == (16, 3)
    assert net.params["b1"].shape == (3,)


def test_graphnet_sage_param_shapes():
    """SAGE concatenates self and neighbour features: W0 is (2*F_in, hidden)."""
    net = GraphNet(num_features=10, num_classes=3, hidden_dim=16, num_layers=2, layer_type="sage")
    assert net.params["W0"].shape == (20, 16)
    assert net.params["W1"].shape == (32, 3)


def test_masked_softmax_loss_uniform_gives_log_c():
    N, C = 8, 4
    scores = np.zeros((N, C))
    y = np.arange(N) % C
    mask = np.array([True, False, True, False, True, True, False, True])
    loss, _ = masked_softmax_loss(scores, y, mask)
    assert loss == pytest.approx(np.log(C))


def test_masked_softmax_loss_matches_subset():
    """Loss must equal the plain softmax loss restricted to the masked rows."""
    rng = np.random.RandomState(0)
    scores = rng.randn(10, 4) * 2
    y = rng.randint(0, 4, size=10)
    mask = rng.rand(10) > 0.5
    loss, _ = masked_softmax_loss(scores, y, mask)
    ref, _ = softmax_loss(scores[mask], y[mask])
    assert loss == pytest.approx(ref)


def test_masked_softmax_loss_zero_gradient_outside_mask():
    rng = np.random.RandomState(1)
    scores = rng.randn(10, 4)
    y = rng.randint(0, 4, size=10)
    mask = np.array([True, False] * 5)
    _, dscores = masked_softmax_loss(scores, y, mask)
    assert np.allclose(dscores[~mask], 0.0)
    assert not np.allclose(dscores[mask], 0.0)


def test_masked_accuracy():
    scores = np.array([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.0, 3.0]])
    y = np.array([0, 1, 1, 1])
    mask = np.array([True, True, True, False])
    assert masked_accuracy(scores, y, mask) == pytest.approx(2.0 / 3.0)


def test_forward_scores_shape():
    _, A_norm, X, y = _small_setup()
    net = GraphNet(num_features=4, num_classes=2, hidden_dim=8, num_layers=2, seed=0)
    scores = net.loss(X, A_norm)
    assert scores.shape == (6, 2)
    preds = net.predict(X, A_norm)
    assert preds.shape == (6,)
    assert preds.dtype == np.int64
    assert set(np.unique(preds)) <= {0, 1}


def test_model_numeric_gradients_gcn():
    _, A_norm, X, y = _small_setup(seed=2)
    mask = np.array([True, False, True, True, False, True])
    net = GraphNet(num_features=4, num_classes=2, hidden_dim=5, num_layers=2, seed=3)

    _, grads = net.loss(X, A_norm, y, mask)
    assert set(grads.keys()) == set(net.params.keys())
    for name in net.params:
        assert grads[name].shape == net.params[name].shape

    for name in ["W0", "b0", "W1", "b1"]:
        f = lambda: net.loss(X, A_norm, y, mask)[0]  # noqa: E731
        num = numeric_gradient(f, net.params[name])
        assert rel_error(grads[name], num) < 1e-5, f"gradient mismatch on {name}"


def test_model_numeric_gradients_sage():
    _, A_norm, X, y = _small_setup(seed=4, layer_type="sage")
    mask = np.array([True, False, True, True, False, True])
    net = GraphNet(num_features=4, num_classes=2, hidden_dim=5, num_layers=2,
                   layer_type="sage", aggregator="mean", seed=5)

    _, grads = net.loss(X, A_norm, y, mask)
    for name in net.params:
        f = lambda: net.loss(X, A_norm, y, mask)[0]  # noqa: E731
        num = numeric_gradient(f, net.params[name])
        assert rel_error(grads[name], num) < 1e-5, f"gradient mismatch on {name}"


def test_model_numeric_gradients_sage_max():
    _, A_norm, X, y = _small_setup(seed=6, layer_type="sage", aggregator="max")
    mask = np.array([True, False, True, True, False, True])
    net = GraphNet(num_features=4, num_classes=2, hidden_dim=5, num_layers=2,
                   layer_type="sage", aggregator="max", seed=7)

    _, grads = net.loss(X, A_norm, y, mask)
    for name in net.params:
        f = lambda: net.loss(X, A_norm, y, mask)[0]  # noqa: E731
        num = numeric_gradient(f, net.params[name])
        assert rel_error(grads[name], num) < 1e-4, f"gradient mismatch on {name}"


def test_model_numeric_gradients_deep():
    """A 3-layer GCN still yields exact gradients (receptive field grows)."""
    _, A_norm, X, y = _small_setup(seed=8)
    mask = np.array([True, True, False, False, True, True])
    net = GraphNet(num_features=4, num_classes=2, hidden_dim=5, num_layers=3, seed=9)

    _, grads = net.loss(X, A_norm, y, mask)
    assert "W2" in grads
    for name in net.params:
        f = lambda: net.loss(X, A_norm, y, mask)[0]  # noqa: E731
        num = numeric_gradient(f, net.params[name])
        assert rel_error(grads[name], num) < 1e-5, f"gradient mismatch on {name}"


def test_single_layer_gcn_matches_one_conv():
    """num_layers=1 reduces to a single graph convolution + softmax."""
    _, A_norm, X, y = _small_setup(seed=10)
    mask = np.array([True, True, True, False, False, False])
    net = GraphNet(num_features=4, num_classes=2, num_layers=1, seed=11)
    scores = net.loss(X, A_norm)
    manual = A_norm @ X @ net.params["W0"] + net.params["b0"]
    assert np.allclose(scores, manual)
