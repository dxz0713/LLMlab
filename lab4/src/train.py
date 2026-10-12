from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model import GraphNet, masked_accuracy, masked_softmax_loss
from src.optim import adam, sgd, sgd_momentum


@dataclass
class TrainConfig:
    """Hyperparameters for Task 4.

    Students are expected to tweak these (especially lr, hidden_dim,
    num_layers and weight_decay) to push test accuracy higher.
    """

    hidden_dim: int = 16
    num_layers: int = 2
    layer_type: str = "gcn"
    aggregator: str = "mean"
    epochs: int = 200
    lr: float = 0.01
    optimizer: str = "adam"
    weight_decay: float = 5e-4
    seed: int = 0


def evaluate(net, X, A_norm, y, mask):
    """Compute loss and accuracy over the nodes selected by mask.

    Returns a tuple of (loss, accuracy).
    """
    raise NotImplementedError


def train(X, A_norm, y, train_mask, cfg: TrainConfig, val_mask=None, verbose=True):
    """Full-batch training loop for Task 4.

    Node classification on a single graph is transductive: the entire graph
    is ONE training sample. There are no minibatches -- every epoch is one
    forward/backward pass over the whole graph, but the loss (and hence the
    gradient) only sees the nodes selected by train_mask.

    Returns a tuple of (net, history). The history dict holds per-epoch
    lists: train_loss, train_acc, and (if val_mask is given) val_loss,
    val_acc.
    """
    num_features = X.shape[1]
    num_classes = int(y.max()) + 1
    raise NotImplementedError

    # return net, history
