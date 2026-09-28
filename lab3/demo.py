"""Train the binary-adder RNN and look at what it has learned (not graded).

Usage:
    uv run python demo.py                        # train, report, save outputs/rnn_adder.png
    uv run python demo.py 123456789 987654321    # ...and then add these two numbers with it

The script:
  1. trains the RNN with the default TrainConfig (about a second), and the
     same model with recurrent=False as a baseline;
  2. prints how often each gets the whole sum right, for numbers much longer
     than the 8-bit ones they were trained on;
  3. saves a figure: the training loss, the accuracy table as a plot, and the
     hidden states projected onto their top two principal components;
  4. if two numbers are given, adds them with the trained RNN.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import bits_to_int, int_to_bits, make_addition  # noqa: E402
from src.model import AdderRNN  # noqa: E402
from src.rnn_layers import rnn_forward  # noqa: E402
from src.train import TrainConfig, sequence_accuracy, train  # noqa: E402

LENGTHS = (8, 16, 32, 64, 128)
EVAL_SAMPLES = 1000

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description="Train an RNN to add binary numbers")
    p.add_argument("numbers", type=int, nargs="*", help="two non-negative integers to add")
    p.add_argument("--steps", type=int, default=TrainConfig.num_steps)
    p.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()
    if args.numbers and (len(args.numbers) != 2 or min(args.numbers) < 0):
        p.error("give exactly two non-negative integers")
    return args


def add_with_rnn(net: AdderRNN, a: int, b: int) -> int:
    """Feed a and b to the network bit by bit and read the sum back."""
    num_bits = max(a.bit_length(), b.bit_length()) + 1  # one spare bit for the final carry
    X = np.stack([int_to_bits(a, num_bits), int_to_bits(b, num_bits)], axis=1)
    return bits_to_int(net.predict(X[None].astype(np.float64))[0])


def full_adder_inputs(X):
    """For every position: the two input bits and the carry coming in from below."""
    a, b = X[:, :, 0].astype(int), X[:, :, 1].astype(int)
    carry_in = np.zeros_like(a)
    carry = np.zeros(a.shape[0], dtype=int)
    for t in range(a.shape[1]):
        carry_in[:, t] = carry
        carry = (a[:, t] + b[:, t] + carry) // 2
    return a, b, carry_in


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(net, history, accuracies, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # 1. training loss
    ax = axes[0]
    ax.plot(history["loss"], color=BLUE, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("step", color=INK_2)
    ax.set_ylabel("training loss (log scale)", color=INK_2)
    style(ax, "Training loss")

    # 2. length generalization
    ax = axes[1]
    x = np.arange(len(LENGTHS))
    for key, label, color, marker in [
        ("rnn", "RNN", BLUE, "o"),
        ("baseline", "no recurrence (Wh = 0)", ORANGE, "s"),
    ]:
        ax.plot(x, accuracies[key], color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
    ax.axvspan(-0.3, 0.3, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0, 0.5, "trained\nlength", ha="center", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [str(n) for n in LENGTHS])
    ax.set_xlim(-0.4, len(LENGTHS) - 0.6)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("number of bits", color=INK_2)
    ax.set_ylabel("whole-sum accuracy", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=INK)
    style(ax, "Adding numbers longer than any seen in training")

    # 3. hidden states
    ax = axes[2]
    X, _ = make_addition(400, 8, np.random.default_rng(5))
    p = net.params
    h, _ = rnn_forward(X, np.zeros((X.shape[0], net.hidden_dim)), p["Wx"], p["Wh"], p["b"])
    H = h.reshape(-1, net.hidden_dim)
    Hc = H - H.mean(axis=0)
    Z = Hc @ np.linalg.svd(Hc, full_matrices=False)[2][:2].T

    a, b, cin = (v.reshape(-1) for v in full_adder_inputs(X))
    carry_out = (a + b + cin) // 2
    for c, color, marker in [(0, BLUE, "o"), (1, ORANGE, "^")]:
        m = carry_out == c
        ax.scatter(Z[m, 0], Z[m, 1], s=14, color=color, marker=marker, alpha=0.55,
                   edgecolors="none", label=f"carry out = {c}")
    for combo in range(8):
        m = (4 * a + 2 * b + cin) == combo
        if not m.any():
            continue
        cx, cy = Z[m].mean(axis=0)
        ax.annotate(f"{combo >> 2}+{(combo >> 1) & 1}+{combo & 1}", (cx, cy), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.margins(x=0.08, y=0.16)
    ax.set_xlabel("principal component 1", color=INK_2)
    ax.set_ylabel("principal component 2", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="best", labelcolor=INK)
    style(ax, "Hidden states, labelled  a_t + b_t + carry_in")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = TrainConfig(num_steps=args.steps, hidden_dim=args.hidden_dim, seed=args.seed)

    t0 = time.time()
    net, history = train(cfg, verbose=True)
    print(f"trained in {time.time() - t0:.1f}s, final loss {history['loss'][-1]:.5f}")
    baseline, _ = train(TrainConfig(num_steps=args.steps, hidden_dim=args.hidden_dim,
                                    seed=args.seed, recurrent=False))

    rng = np.random.default_rng(12345)
    accuracies = {"rnn": [], "baseline": []}
    print(f"\nwhole-sum accuracy on {EVAL_SAMPLES} random problems (trained on {cfg.num_bits}-bit numbers only)")
    print(f"{'bits':>6}  {'RNN':>6}  {'no recurrence':>14}")
    for n in LENGTHS:
        X, y = make_addition(EVAL_SAMPLES, n, rng)
        accuracies["rnn"].append(sequence_accuracy(net, X, y))
        accuracies["baseline"].append(sequence_accuracy(baseline, X, y))
        print(f"{n:>6}  {accuracies['rnn'][-1]:>6.3f}  {accuracies['baseline'][-1]:>14.3f}")

    out_path = os.path.join(args.out_dir, "rnn_adder.png")
    plot(net, history, accuracies, out_path)
    print(f"\nsaved {out_path}")

    if args.numbers:
        a, b = args.numbers
        got = add_with_rnn(net, a, b)
        verdict = "correct" if got == a + b else f"wrong, the answer is {a + b}"
        print(f"\nthe RNN says {a} + {b} = {got}   ({verdict})")


if __name__ == "__main__":
    main()
