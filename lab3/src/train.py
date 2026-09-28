from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.data import make_addition
from src.model import AdderRNN
from src.optim import adam, clip_grad_norm

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class TrainConfig:
    """Hyperparameters for training the binary adder.

    Attributes:
        hidden_dim: size H of the hidden state.
        num_bits: length T of every training sequence.
        batch_size: number of addition problems per step.
        num_steps: number of parameter updates.
        lr: Adam learning rate.
        clip: max_norm passed to clip_grad_norm; None disables clipping.
        recurrent: if False, Wh is set to zero and kept there for the whole
            run, which cuts the connection between time steps. This is the
            "no memory" baseline: every step then sees only its own two input
            bits, like an MLP applied to each position separately.
        seed: seeds both the model initialization and the data.
    """

    hidden_dim: int = 16
    num_bits: int = 8
    batch_size: int = 64
    num_steps: int = 1000
    lr: float = 1e-2
    clip: float | None = 5.0
    recurrent: bool = True
    seed: int = 0


def sequence_accuracy(net: AdderRNN, X: Array, y: IntArray) -> float:
    """Fraction of problems whose sum is predicted exactly right.

    A problem only counts as correct if every one of its T bits is correct;
    getting 7 of 8 bits right is still a wrong answer to an addition.

    Args:
        net: the model.
        X: input bits, shape (N, T, 2).
        y: correct sum bits, shape (N, T).

    Returns:
        A Python float in [0, 1].
    """
    pred = net.predict(X)  # (N, T)
    # All bits correct per sample
    all_correct = np.all(pred == y, axis=1)  # (N,)
    return float(np.mean(all_correct))


def train(cfg: TrainConfig, verbose: bool = False) -> tuple[AdderRNN, dict[str, list[float]]]:
    """Train an AdderRNN on freshly generated addition problems.

    Every step draws a new batch, so the model never sees the same problem
    twice and there is no fixed training set to overfit. Create one
    generator rng = np.random.default_rng(cfg.seed) at the start and draw
    every batch from it with make_addition(cfg.batch_size, cfg.num_bits, rng).

    One step is: sample a batch, compute loss and gradients, measure (and,
    if cfg.clip is set, clip) the gradient norm, then update every parameter
    with adam, each with its own config dict.

    If cfg.recurrent is False, set Wh to zero before the first step and
    never update it.

    Args:
        cfg: hyperparameters.
        verbose: if True, print the loss every 100 steps.

    Returns:
        net: the trained model, built as AdderRNN(cfg.hidden_dim, seed=cfg.seed).
        history: a dict of per-step lists, each of length cfg.num_steps:
            "loss": the training loss of the batch, before the update;
            "grad_norm": the combined gradient norm, before clipping.
    """
    rng = np.random.default_rng(cfg.seed)
    net = AdderRNN(cfg.hidden_dim, seed=cfg.seed)

    # If non-recurrent, zero out Wh permanently
    if not cfg.recurrent:
        net.params["Wh"] = np.zeros_like(net.params["Wh"])

    # Adam configs for each parameter
    optim_configs = {k: {"learning_rate": cfg.lr} for k in net.params}

    history: dict[str, list[float]] = {"loss": [], "grad_norm": []}

    # Determine max_norm for clipping
    max_norm = cfg.clip if cfg.clip is not None else np.inf

    for step in range(cfg.num_steps):
        # Generate batch
        X, y = make_addition(cfg.batch_size, cfg.num_bits, rng)

        # Compute loss and gradients
        loss, grads = net.loss(X, y)

        # Record loss before update
        history["loss"].append(loss)

        # Measure (and clip) gradient norm
        total_norm = clip_grad_norm(grads, max_norm)
        history["grad_norm"].append(total_norm)

        # For non-recurrent, zero the Wh gradient to keep it at zero
        if not cfg.recurrent:
            grads["Wh"] = np.zeros_like(grads["Wh"])

        # Update each parameter with Adam
        for k in net.params:
            net.params[k], optim_configs[k] = adam(net.params[k], grads[k], optim_configs[k])

        if verbose and (step + 1) % 100 == 0:
            print(f"step {step + 1:4d}, loss = {loss:.4f}, grad_norm = {total_norm:.4f}")

    return net, history
