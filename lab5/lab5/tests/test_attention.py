import numpy as np
import pytest

from src.attention import (
    MultiHeadAttention,
    apply_rope,
    causal_mask,
    compute_rope_params,
    scaled_dot_product_attention,
    softmax,
)


# -- softmax -------------------------------------------------------------


def test_softmax_rows_sum_to_one():
    x = np.random.RandomState(0).randn(4, 7)
    assert np.allclose(softmax(x, axis=-1).sum(axis=-1), 1.0)


def test_softmax_is_numerically_stable():
    """Naive exp() would overflow on logits this large."""
    x = np.array([[1000.0, 1001.0, 1002.0]])
    out = softmax(x)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(), 1.0)


def test_softmax_is_shift_invariant():
    """Adding a constant to every logit must not change the distribution."""
    x = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(x), softmax(x + 100.0))


def test_softmax_matches_manual():
    x = np.array([[0.5, -1.0, 2.0]])
    e = np.exp(x - x.max())
    assert np.allclose(softmax(x), e / e.sum())


# -- causal mask ---------------------------------------------------------


def test_causal_mask_blocks_upper_triangle():
    mask = causal_mask(4)
    for i in range(4):
        for j in range(4):
            if j > i:
                assert mask[i, j] == -np.inf
            else:
                assert mask[i, j] == 0.0


# -- scaled dot-product attention ----------------------------------------


def test_sdpa_weights_are_a_distribution_per_query():
    rng = np.random.RandomState(1)
    q, k, v = rng.randn(3, 5), rng.randn(3, 5), rng.randn(3, 4)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert weights.shape == (3, 3)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_sdpa_matches_manual_computation():
    rng = np.random.RandomState(2)
    q, k, v = rng.randn(3, 5), rng.randn(3, 5), rng.randn(3, 4)
    context, weights = scaled_dot_product_attention(q, k, v)
    scores = q @ k.T / np.sqrt(5)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected_w = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(weights, expected_w)
    assert np.allclose(context, expected_w @ v)


def test_sdpa_supports_batched_heads():
    """The same function must act on (batch, heads, seq, d) via the last 2 axes."""
    rng = np.random.RandomState(3)
    q, k, v = rng.randn(2, 4, 3, 5), rng.randn(2, 4, 3, 5), rng.randn(2, 4, 3, 6)
    context, weights = scaled_dot_product_attention(q, k, v)
    assert context.shape == (2, 4, 3, 6)
    assert weights.shape == (2, 4, 3, 3)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_sdpa_causal_blocks_future():
    """With a causal mask, position i puts zero weight on any j > i."""
    rng = np.random.RandomState(4)
    seq = 5
    q, k, v = rng.randn(seq, 4), rng.randn(seq, 4), rng.randn(seq, 4)
    _, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(seq))
    iu = np.triu_indices(seq, k=1)
    assert np.allclose(weights[iu], 0.0)
    # every row is still a proper distribution
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_sdpa_first_position_attends_only_to_itself():
    rng = np.random.RandomState(5)
    seq = 6
    q, k, v = rng.randn(seq, 4), rng.randn(seq, 4), rng.randn(seq, 3)
    context, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(seq))
    assert np.allclose(weights[0], [1.0, 0, 0, 0, 0, 0])
    # and the first context vector is exactly the first value vector
    assert np.allclose(context[0], v[0])


# -- multi-head attention -------------------------------------------------


def _manual_mha(mha: MultiHeadAttention, x: np.ndarray) -> np.ndarray:
    """Reference: run each head on its slice of the projections, then combine."""
    batch, seq, _ = x.shape
    q = x @ mha.W_query
    k = x @ mha.W_key
    v = x @ mha.W_value
    head_dim = mha.head_dim
    head_outs = []
    for h in range(mha.num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        ctx, _ = scaled_dot_product_attention(
            q[:, :, sl], k[:, :, sl], v[:, :, sl], mask=causal_mask(seq)
        )
        head_outs.append(ctx)
    combined = np.concatenate(head_outs, axis=-1)
    return combined @ mha.out_proj


def test_mha_output_shape():
    mha = MultiHeadAttention(d_in=8, d_out=8, num_heads=2, context_length=6,
                             dropout=0.0, rng=np.random.RandomState(0))
    x = np.random.RandomState(1).randn(2, 6, 8)
    assert mha(x).shape == (2, 6, 8)


def test_mha_rejects_indivisible_head_split():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_in=8, d_out=8, num_heads=3, context_length=4)


def test_mha_matches_manual_per_head_computation():
    """The parallel head split must equal looping over heads by hand."""
    mha = MultiHeadAttention(d_in=6, d_out=6, num_heads=3, context_length=5,
                             dropout=0.0, rng=np.random.RandomState(7))
    x = np.random.RandomState(8).randn(2, 5, 6)
    assert np.allclose(mha(x), _manual_mha(mha, x))


def test_mha_single_head_equals_plain_attention():
    """With one head, MHA reduces to single-head attention plus a projection."""
    mha = MultiHeadAttention(d_in=5, d_out=5, num_heads=1, context_length=4,
                             dropout=0.0, rng=np.random.RandomState(9))
    x = np.random.RandomState(10).randn(2, 4, 5)
    context, _ = scaled_dot_product_attention(
        x @ mha.W_query, x @ mha.W_key, x @ mha.W_value, mask=causal_mask(4)
    )
    assert np.allclose(mha(x), context @ mha.out_proj)


def test_mha_returned_weights_have_head_axis():
    mha = MultiHeadAttention(d_in=8, d_out=8, num_heads=4, context_length=6,
                             dropout=0.0, rng=np.random.RandomState(11))
    x = np.random.RandomState(12).randn(2, 6, 8)
    out, weights = mha(x, return_weights=True)
    assert weights.shape == (2, 4, 6, 6)
    iu = np.triu_indices(6, k=1)
    assert np.allclose(weights[:, :, iu[0], iu[1]], 0.0)


# -- RoPE -----------------------------------------------------------------


def test_rope_params_shape():
    cos, sin = compute_rope_params(head_dim=8, theta_base=10_000, context_length=16)
    assert cos.shape == sin.shape == (16, 8)


def test_rope_requires_even_head_dim():
    with pytest.raises(ValueError):
        compute_rope_params(head_dim=7)


def test_rope_preserves_vector_norm():
    """Rotation never changes a vector's length."""
    rng = np.random.RandomState(13)
    cos, sin = compute_rope_params(head_dim=8, context_length=16)
    v = rng.randn(1, 1, 16, 8)
    assert np.allclose(np.linalg.norm(apply_rope(v, cos, sin), axis=-1),
                       np.linalg.norm(v, axis=-1))


def test_rope_position_zero_is_identity():
    """At position 0 the angle is 0, so the vector is left unchanged."""
    rng = np.random.RandomState(14)
    cos, sin = compute_rope_params(head_dim=8, context_length=16)
    v = rng.randn(1, 1, 16, 8)
    assert np.allclose(apply_rope(v, cos, sin)[0, 0, 0], v[0, 0, 0])


def test_rope_dot_product_depends_only_on_relative_offset():
    """Rotating the same q and k into positions m, n gives a dot product that
    depends only on the offset m - n.

    The query and key must be *the same vector* at every position for this to
    hold: RoPE guarantees ``R_m(q) . R_n(k) = q . R_{n-m}(k)``, which is a
    statement about one fixed pair of vectors, so we replicate a single q and
    a single k across all positions before rotating.
    """
    rng = np.random.RandomState(15)
    cos, sin = compute_rope_params(head_dim=8, theta_base=10_000, context_length=16)
    q = np.repeat(rng.randn(1, 1, 1, 8), 16, axis=2)
    k = np.repeat(rng.randn(1, 1, 1, 8), 16, axis=2)
    q_rot, k_rot = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def dot(m, n):
        return float(q_rot[0, 0, m] @ k_rot[0, 0, n])

    # three pairs with the same relative distance of 3 must agree
    assert dot(5, 2) == pytest.approx(dot(9, 6), rel=1e-9)
    assert dot(5, 2) == pytest.approx(dot(12, 9), rel=1e-9)
    # a different relative distance gives a different value in general
    assert abs(dot(5, 2) - dot(5, 1)) > 1e-6
