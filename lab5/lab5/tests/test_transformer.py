import numpy as np
import pytest

from src.transformer import FeedForward, LayerNorm, TransformerBlock, gelu


# -- LayerNorm ------------------------------------------------------------


def test_layernorm_normalizes_each_position():
    x = np.random.RandomState(0).randn(2, 3, 4) * 5 + 7  # off-center, off-scale
    out = LayerNorm(emb_dim=4)(x)
    assert out.shape == x.shape
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)


def test_layernorm_matches_manual():
    x = np.random.RandomState(1).randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    expected = (x - mean) / np.sqrt(var + ln.eps)
    assert np.allclose(ln(x), expected)


def test_layernorm_applies_scale_and_shift():
    x = np.random.RandomState(2).randn(3, 4)
    ln = LayerNorm(emb_dim=4)
    ln.scale = np.full(4, 2.0)
    ln.shift = np.full(4, 3.0)
    base = LayerNorm(emb_dim=4)(x)
    assert np.allclose(ln(x), 2.0 * base + 3.0)


# -- GELU -----------------------------------------------------------------


def test_gelu_at_zero_is_zero():
    assert gelu(np.array([0.0]))[0] == pytest.approx(0.0)


def test_gelu_matches_tanh_approximation():
    x = np.linspace(-4, 4, 9)
    expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    assert np.allclose(gelu(x), expected)


def test_gelu_asymptotes():
    """For large |x| GELU approaches x on the right and 0 on the left."""
    assert gelu(np.array([10.0]))[0] == pytest.approx(10.0, abs=1e-4)
    assert gelu(np.array([-10.0]))[0] == pytest.approx(0.0, abs=1e-4)


def test_gelu_is_not_relu():
    """GELU is smooth and slightly negative just left of zero, unlike ReLU."""
    small_negative = gelu(np.array([-0.5]))[0]
    assert small_negative < 0.0


# -- FeedForward ----------------------------------------------------------


def test_feedforward_preserves_shape():
    ff = FeedForward(emb_dim=4, rng=np.random.RandomState(3))
    x = np.random.RandomState(4).randn(2, 5, 4)
    assert ff(x).shape == x.shape


def test_feedforward_matches_manual():
    ff = FeedForward(emb_dim=4, rng=np.random.RandomState(5))
    x = np.random.RandomState(6).randn(2, 3, 4)
    expected = gelu(x @ ff.W1 + ff.b1) @ ff.W2 + ff.b2
    assert np.allclose(ff(x), expected)


# -- TransformerBlock -----------------------------------------------------


def test_transformer_block_preserves_shape():
    block = TransformerBlock(emb_dim=8, num_heads=2, context_length=6,
                             dropout=0.0, rng=np.random.RandomState(7))
    x = np.random.RandomState(8).randn(2, 6, 8)
    assert block(x).shape == x.shape


def test_transformer_block_residual_is_identity_when_sublayers_zero():
    """Zero the sublayer outputs and the block must reduce to the identity.

    Setting out_proj and the FFN's output projection to zero forces both
    sublayers to emit zeros, leaving only the residual stream x + 0 + 0.
    """
    block = TransformerBlock(emb_dim=8, num_heads=2, context_length=6,
                             dropout=0.0, rng=np.random.RandomState(9))
    block.att.out_proj[:] = 0.0
    block.ff.W2[:] = 0.0
    block.ff.b2[:] = 0.0
    x = np.random.RandomState(10).randn(2, 6, 8)
    assert np.allclose(block(x), x)


def test_transformer_block_matches_manual_prenorm():
    """The block must equal hand-assembled pre-norm residual composition."""
    block = TransformerBlock(emb_dim=8, num_heads=2, context_length=6,
                             dropout=0.0, rng=np.random.RandomState(11))
    x = np.random.RandomState(12).randn(2, 6, 8)
    expected = x + block.att(block.norm1(x))
    expected = expected + block.ff(block.norm2(expected))
    assert np.allclose(block(x), expected)


def test_transformer_block_stacks_without_shape_drift():
    """Two blocks in a row must keep the (batch, seq, emb_dim) shape."""
    rng = np.random.RandomState(13)
    b1 = TransformerBlock(emb_dim=8, num_heads=2, context_length=6, dropout=0.0, rng=rng)
    b2 = TransformerBlock(emb_dim=8, num_heads=2, context_length=6, dropout=0.0, rng=rng)
    x = np.random.RandomState(14).randn(2, 6, 8)
    assert b2(b1(x)).shape == x.shape
