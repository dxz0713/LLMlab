from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax along ``axis``.

    Subtract the per-row maximum before exponentiating. This does not change
    the result (softmax is shift-invariant) but keeps ``exp`` from overflowing
    to ``inf`` when the logits are large -- the same trick used in Task 2 of
    lab 1's ``softmax_cross_entropy``.

    Args:
        x: input array of any shape.
        axis: the axis to normalize over.

    Returns:
        An array of the same shape whose slices along ``axis`` are non-negative
        and sum to 1.
    """
    raise NotImplementedError


def causal_mask(seq_len: int) -> Array:
    """Additive mask that hides every position after the current one.

    The mask has zeros on and below the diagonal and ``-inf`` strictly above
    it. Adding it to an attention-score matrix sends future scores to
    ``-inf``, which softmax then maps to exactly zero -- so position ``i`` can
    only attend to positions ``0..i``.

    Args:
        seq_len: number of positions in the sequence.

    Returns:
        A ``(seq_len, seq_len)`` array; entry ``[i, j]`` is ``-inf`` when
        ``j > i`` else ``0``.
    """
    raise NotImplementedError


def scaled_dot_product_attention(
    q: Array, k: Array, v: Array, mask: Array | None = None
) -> tuple[Array, Array]:
    """Attention over the last two axes: ``softmax(q @ k^T / sqrt(d)) @ v``.

    Works on plain ``(seq, d)`` inputs and on batched/multi-head inputs of
    shape ``(batch, heads, seq, d)`` alike, since every operation acts on the
    last two axes only.

    Args:
        q: queries, shape ``(..., seq_q, d)``.
        k: keys, shape ``(..., seq_k, d)``.
        v: values, shape ``(..., seq_k, d_v)``.
        mask: optional additive mask broadcastable to ``(..., seq_q, seq_k)``;
            entries of ``-inf`` remove that position from the attention. Pass
            ``causal_mask(seq)`` to stop positions from seeing the future.

    Returns:
        A pair ``(context, weights)``. ``context`` has shape
        ``(..., seq_q, d_v)``; ``weights`` has shape ``(..., seq_q, seq_k)``
        and its last axis sums to 1 (each query's distribution over keys).
    """
    raise NotImplementedError


def _attention_dropout(weights: Array, rate: float, training: bool, rng: np.random.RandomState) -> Array:
    """Inverted dropout on attention weights; a no-op when not training.

    During training, each weight is independently zeroed with probability
    ``rate`` and the survivors are scaled by ``1 / (1 - rate)`` so the
    expected value is unchanged. Given so you can focus on the attention
    logic itself; call it on the attention weights when ``self.training``.

    Args:
        weights: attention weights of any shape.
        rate: dropout probability in ``[0, 1)``.
        training: whether the model is in training mode.
        rng: source of randomness for the dropout mask.

    Returns:
        The dropped-out (or unchanged) weights.
    """
    if not training or rate == 0.0:
        return weights
    keep = rng.rand(*weights.shape) >= rate
    return weights * keep / (1.0 - rate)


class MultiHeadAttention:
    """Causal multi-head self-attention, computed with plain NumPy.

    The input is projected to queries, keys and values, the ``d_out`` features
    are split into ``num_heads`` heads of ``head_dim`` each, attention runs on
    all heads in parallel (as batched matrix operations, not a Python loop),
    and the per-head outputs are concatenated and projected back to ``d_out``.
    A causal mask is always applied, so this is a decoder-style attention.

    Attributes:
        d_out: total output (and value) dimension.
        num_heads: number of attention heads.
        head_dim: per-head dimension, ``d_out // num_heads``.
        W_query, W_key, W_value: projection matrices of shape ``(d_in, d_out)``.
        out_proj: output projection of shape ``(d_out, d_out)``.
        dropout: attention-dropout rate.
        training: whether dropout is active.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        rng: np.random.RandomState | None = None,
    ) -> None:
        """Create the projections and the causal mask.

        Weights are drawn from ``rng`` (never the global generator) scaled by
        ``1 / sqrt(d_in)`` so activations keep a stable scale, and so a fixed
        seed reproduces a fixed module.

        Args:
            d_in: input feature dimension.
            d_out: total output feature dimension; must be divisible by
                ``num_heads``.
            num_heads: number of attention heads.
            context_length: maximum sequence length the causal mask covers.
            dropout: attention-dropout rate; 0 disables it.
            rng: source of randomness for the projection weights and dropout.
        """
        if d_out % num_heads != 0:
            raise ValueError(f"d_out={d_out} must be divisible by num_heads={num_heads}")
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.dropout = dropout
        self.training = False
        self.rng = np.random.RandomState() if rng is None else rng

        scale = 1.0 / np.sqrt(d_in)
        self.W_query = self.rng.randn(d_in, d_out) * scale
        self.W_key = self.rng.randn(d_in, d_out) * scale
        self.W_value = self.rng.randn(d_in, d_out) * scale
        self.out_proj = self.rng.randn(d_out, d_out) * (1.0 / np.sqrt(d_out))

    def forward(self, x: Array, return_weights: bool = False):
        """Run causal multi-head attention on a batch of sequences.

        Args:
            x: input array of shape ``(batch, seq_len, d_in)`` with
                ``seq_len <= context_length``.
            return_weights: if True, also return the attention weights of
                shape ``(batch, num_heads, seq_len, seq_len)`` for inspection
                and visualization.

        Returns:
            The output array of shape ``(batch, seq_len, d_out)``, or the pair
            ``(output, weights)`` when ``return_weights`` is True.
        """
        raise NotImplementedError

    __call__ = forward


def compute_rope_params(
    head_dim: int, theta_base: float = 10_000.0, context_length: int = 4096
) -> tuple[Array, Array]:
    """Precompute the cosine and sine tables for rotary position embedding.

    For each pair of feature dimensions ``(2i, 2i+1)`` RoPE rotates by an
    angle ``position * theta_base ** (-2i / head_dim)``. Both halves of the
    head share the same angle table, so the tables are built at width
    ``head_dim // 2`` and then concatenated to width ``head_dim`` (the
    "rotate-half" layout used by Llama-family models).

    Args:
        head_dim: per-head dimension; must be even.
        theta_base: the frequency base, 10000 in most models.
        context_length: number of positions to tabulate.

    Returns:
        ``(cos, sin)``, each of shape ``(context_length, head_dim)``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim={head_dim} must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (theta_base ** (np.arange(0, head_dim, 2)[:half] / head_dim))
    positions = np.arange(context_length, dtype=np.float64)
    angles = positions[:, None] * inv_freq[None, :]  # (context_length, half)
    angles = np.concatenate([angles, angles], axis=1)  # (context_length, head_dim)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate each position's query/key vector by its position-dependent angle.

    RoPE touches only the queries and keys, not the attention structure: the
    vector is split into two halves, the second half is rotated into the first
    (and vice versa with a sign flip) and combined with the cos/sin tables.
    Because rotation preserves length and the angle depends only on position,
    ``apply_rope(q, m) . apply_rope(k, n)`` depends only on the offset
    ``m - n`` -- the relative-position property RoPE is known for.

    Args:
        x: array whose last two axes are ``(seq_len, head_dim)``; leading
            axes (batch, heads, ...) are broadcast over.
        cos, sin: tables from ``compute_rope_params``, shape
            ``(context_length, head_dim)`` with ``context_length >= seq_len``.

    Returns:
        The rotated array, same shape as ``x``.
    """
    raise NotImplementedError
