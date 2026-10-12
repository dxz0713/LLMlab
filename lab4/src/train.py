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
    scores = net.loss(X, A_norm)
    loss, _ = masked_softmax_loss(scores, y, mask)
    acc = masked_accuracy(scores, y, mask)
    return loss, acc


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

    net = GraphNet(num_features, num_classes, cfg.hidden_dim, cfg.num_layers,
                   cfg.layer_type, cfg.aggregator, cfg.seed)
    if cfg.optimizer == "adam":
        optim_func = adam
    elif cfg.optimizer == "sgd":
        optim_func = sgd
    elif cfg.optimizer == "sgd_momentum":
        optim_func = sgd_momentum
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")

    history = {"train_loss": [], "train_acc": []}
    if val_mask is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    optim_configs = {
        name: {"learning_rate": cfg.lr}
        for name in net.params
    }
    for epoch in range(cfg.epochs):
        loss, grads = net.loss(X, A_norm, y, train_mask)
        for name, param in net.params.items():
            if name.startswith("W"):
                grads[name] += cfg.weight_decay * param
            net.params[name], optim_configs[name] = optim_func(
                param, grads[name], optim_configs[name]
            )

        train_loss, train_acc = evaluate(net, X, A_norm, y, train_mask)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_mask is not None:
            val_loss, val_acc = evaluate(net, X, A_norm, y, val_mask)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose:
            message = (
                f"Epoch {epoch + 1}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )
            if val_mask is not None:
                message += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            print(message)

    return net, history
