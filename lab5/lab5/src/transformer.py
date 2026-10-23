from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.attention import MultiHeadAttention, _attention_dropout

Array = NDArray[np.float64]


class LayerNorm:
    """Normalize the feature axis of each position to zero mean, unit variance.

    Each position (last axis) is normalized independently, then rescaled by a
    learnable ``scale`` and ``shift``. This keeps activations in a stable
    range no matter how deep the stack grows. The variance uses the biased
    (population) estimator, matching ``torch.nn.LayerNorm``.

    Attributes:
        eps: small constant for numerical stability inside the square root.
        scale: per-feature gain of shape ``(emb_dim,)``, initialized to ones.
        shift: per-feature bias of shape ``(emb_dim,)``, initialized to zeros.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.scale = np.ones(emb_dim, dtype=np.float64)
        self.shift = np.zeros(emb_dim, dtype=np.float64)

    def forward(self, x: Array) -> Array:
        """Normalize ``x`` over its last axis, then apply ``scale``/``shift``.

        Args:
            x: array whose last axis is the feature axis of length
                ``emb_dim``.

        Returns:
            The normalized array, same shape as ``x``. With default
            ``scale``/``shift`` each position has mean ~0 and variance ~1.
        """
        raise NotImplementedError

    __call__ = forward


def gelu(x: Array) -> Array:
    """Gaussian-error-linear-unit activation, tanh approximation.

    This is the smooth, non-monotonic activation used in GPT-2's feed-forward
    network (as opposed to the exact ``x * Phi(x)`` form, the tanh
    approximation is what the original GPT-2 code ships).

    Args:
        x: input array of any shape.

    Returns:
        ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``,
        applied elementwise.
    """
    raise NotImplementedError


class FeedForward:
    """The position-wise two-layer MLP inside every transformer block.

    Expands each position from ``emb_dim`` to ``4 * emb_dim``, applies GELU,
    and projects back. Every position goes through the same two matrices
    independently -- no information flows between positions here, that is the
    attention sublayer's job.

    Attributes:
        W1, b1: expansion projection, shapes ``(emb_dim, 4*emb_dim)`` and
            ``(4*emb_dim,)``.
        W2, b2: contraction projection, shapes ``(4*emb_dim, emb_dim)`` and
            ``(emb_dim,)``.
    """

    def __init__(self, emb_dim: int, rng: np.random.RandomState | None = None) -> None:
        """Create the two projections, initialized from ``rng``.

        Args:
            emb_dim: feature dimension; the hidden width is ``4 * emb_dim``.
            rng: source of randomness, so a fixed seed reproduces the layer.
        """
        rng = np.random.RandomState() if rng is None else rng
        hidden = 4 * emb_dim
        self.W1 = rng.randn(emb_dim, hidden) * (1.0 / np.sqrt(emb_dim))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.randn(hidden, emb_dim) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(emb_dim, dtype=np.float64)

    def forward(self, x: Array) -> Array:
        """Apply ``gelu(x @ W1 + b1) @ W2 + b2`` position-wise.

        Args:
            x: array whose last axis has length ``emb_dim``.

        Returns:
            An array of the same shape as ``x``.
        """
        raise NotImplementedError

    __call__ = forward


class TransformerBlock:
    """One pre-norm transformer block: attention, then feed-forward.

    Each sublayer is wrapped as ``x + sublayer(LayerNorm(x))``: the input is
    normalized, transformed, and added back onto the residual stream. The
    residual connections let gradients flow straight through the stack and let
    each sublayer learn a *correction* to its input rather than a wholesale
    replacement.

    Attributes:
        norm1, norm2: the LayerNorms before the attention and feed-forward
            sublayers.
        att: the causal MultiHeadAttention sublayer.
        ff: the FeedForward sublayer.
        dropout: residual-dropout rate.
        training: whether dropout is active.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        rng: np.random.RandomState | None = None,
    ) -> None:
        rng = np.random.RandomState() if rng is None else rng
        self.norm1 = LayerNorm(emb_dim)
        self.att = MultiHeadAttention(emb_dim, emb_dim, num_heads, context_length, dropout, rng)
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, rng)
        self.dropout = dropout
        self.training = False
        self.rng = rng

    def forward(self, x: Array) -> Array:
        """Run the block: two residual sublayers, attention then feed-forward.

        Args:
            x: array of shape ``(batch, seq_len, emb_dim)``.

        Returns:
            An array of the same shape; a transformer block never changes the
            shape, it only refines the per-position representations.
        """
        raise NotImplementedError

    __call__ = forward
