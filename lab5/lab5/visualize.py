"""Visualize what the tokenizer and the attention mechanism learned.

Run with:  uv run python visualize.py

Produces lab5.png with four panels: how a from-scratch BPE tokenizer
compresses its training corpus as the vocabulary grows, how that compression
compares on in-corpus vs novel text, and two causal-attention heads whose
strictly lower-triangular weight maps show each position attending only to
itself and the past. Nothing here is graded -- it is the payoff for getting
the tasks right.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.attention import MultiHeadAttention
from src.tokenizer import BPETokenizer, compression_ratio

CORPUS_PATH = "data/the-verdict.txt"


def bpe_panels(fig, axes, corpus: str) -> None:
    """Left: corpus token count shrinks as merges grow. Right: compression."""
    # Train tokenizers of growing size and see the corpus need fewer tokens.
    vocab_sizes = [256, 300, 400, 600, 800, 1000]
    token_counts = []
    for vs in vocab_sizes:
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=vs)
        token_counts.append(len(tok.encode(corpus)))

    ax = axes[0]
    ax.plot(vocab_sizes, token_counts, marker="o")
    ax.set_xlabel("vocab size")
    ax.set_ylabel("tokens to encode corpus")
    ax.set_title("BPE learns to compress")

    # A trained tokenizer compresses in-corpus text far better than novel text.
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=1000)
    in_ratio = compression_ratio(corpus[:2000], tok)
    out_ratio = compression_ratio("zqwxv kjrbt nmplf gsdkh vqwc xjzb pq", tok)
    ax = axes[1]
    ax.bar(["in-corpus", "novel text"], [in_ratio, out_ratio], color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("bytes / token")
    ax.set_title("Compression: seen vs unseen")
    for i, v in enumerate([in_ratio, out_ratio]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")


def attention_panels(fig, axes) -> None:
    """Two causal-attention heads: strictly lower-triangular weight maps."""
    rng = np.random.RandomState(0)
    seq, d = 12, 16
    mha = MultiHeadAttention(d_in=d, d_out=d, num_heads=4, context_length=seq,
                             dropout=0.0, rng=rng)
    x = np.random.RandomState(1).randn(1, seq, d)
    _, weights = mha(x, return_weights=True)  # (1, heads, seq, seq)

    for ax, head in zip(axes, [0, 1]):
        im = ax.imshow(weights[0, head], cmap="viridis", vmin=0.0)
        ax.set_title(f"Attention head {head}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        fig.colorbar(im, ax=ax, fraction=0.046)


def main() -> None:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = f.read()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    bpe_panels(fig, axes[0:2], corpus)
    attention_panels(fig, axes[2:4])

    fig.tight_layout()
    fig.savefig("lab5.png", dpi=120)
    print("saved lab5.png")


if __name__ == "__main__":
    main()
