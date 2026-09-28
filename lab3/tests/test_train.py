import functools

import numpy as np
import pytest

from src.data import make_addition
from src.model import AdderRNN
from src.train import TrainConfig, sequence_accuracy, train


class FixedPredictor:
    """Stands in for a model; predict() returns a fixed answer."""

    def __init__(self, pred):
        self.pred = pred

    def predict(self, X):
        return self.pred


@functools.cache
def trained_net():
    """Trained once, then shared by the tests that need a trained model."""
    net, _ = train(TrainConfig(num_steps=500, seed=0))
    return net


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.hidden_dim == 16
    assert cfg.num_bits == 8
    assert cfg.batch_size == 64
    assert cfg.num_steps == 1000
    assert cfg.lr == 1e-2
    assert cfg.clip == 5.0
    assert cfg.recurrent is True
    assert cfg.seed == 0


def test_sequence_accuracy_counts_whole_sequences():
    """One wrong bit makes the whole sum wrong."""
    y = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    X = np.zeros((3, 4, 2))
    assert sequence_accuracy(FixedPredictor(y.copy()), X, y) == pytest.approx(1.0)

    pred = y.copy()
    pred[0, 3] = 1  # 3 of 4 bits right is still a wrong answer
    assert sequence_accuracy(FixedPredictor(pred), X, y) == pytest.approx(2 / 3)


def test_train_history_records_values_before_the_update():
    """history[...][0] belongs to the untrained model; grad_norm is measured before clipping."""
    cfg = TrainConfig(num_steps=3, seed=7, clip=1e-3)
    net, history = train(cfg)
    assert isinstance(net, AdderRNN)
    assert set(history) == {"loss", "grad_norm"}
    assert len(history["loss"]) == 3 and len(history["grad_norm"]) == 3

    untrained = AdderRNN(cfg.hidden_dim, seed=cfg.seed)
    X, y = make_addition(cfg.batch_size, cfg.num_bits, np.random.default_rng(cfg.seed))
    loss0, grads0 = untrained.loss(X, y)
    norm0 = np.sqrt(sum(np.sum(g * g) for g in grads0.values()))
    assert history["loss"][0] == pytest.approx(loss0)
    assert history["grad_norm"][0] == pytest.approx(norm0)
    assert min(history["grad_norm"]) > 1e-2  # far above the clip value of 1e-3


def test_train_reduces_loss():
    _, history = train(TrainConfig(num_steps=150, seed=1))
    assert len(history["loss"]) == 150
    assert np.all(np.isfinite(history["loss"]))
    assert np.mean(history["loss"][-20:]) < 0.5 * history["loss"][0]


def test_rnn_learns_binary_addition():
    X, y = make_addition(1000, 8, np.random.default_rng(100))
    assert sequence_accuracy(trained_net(), X, y) > 0.99


def test_rnn_generalizes_to_longer_numbers():
    """Trained on 8-bit numbers only, it still adds 32-bit numbers correctly."""
    X, y = make_addition(1000, 32, np.random.default_rng(101))
    assert sequence_accuracy(trained_net(), X, y) > 0.99


def test_without_recurrence_it_cannot_add():
    """With Wh fixed at zero every step sees only its own two bits and cannot know the carry."""
    net, _ = train(TrainConfig(num_steps=500, seed=0, recurrent=False))
    assert np.all(net.params["Wh"] == 0.0)
    X, y = make_addition(1000, 8, np.random.default_rng(102))
    assert sequence_accuracy(net, X, y) < 0.3
