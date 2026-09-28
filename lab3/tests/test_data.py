import numpy as np

from src.data import add_bits, make_addition


def to_ints(bits):
    """Decode (N, T) bit arrays, least significant first. Only valid for T < 63."""
    return bits.astype(np.int64) @ (2 ** np.arange(bits.shape[1], dtype=np.int64))


def test_add_bits_hand_computed():
    # Least significant bit first:
    #   7 + 1 = 8   ->  [1,1,1,0] + [1,0,0,0] = [0,0,0,1]   (the carry ripples 3 places)
    #   5 + 2 = 7   ->  [1,0,1,0] + [0,1,0,0] = [1,1,1,0]   (no carry at all)
    #   6 + 6 = 12  ->  [0,1,1,0] + [0,1,1,0] = [0,0,1,1]
    a = np.array([[1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
    b = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
    expected = np.array([[0, 0, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1]])
    assert np.array_equal(add_bits(a, b), expected)


def test_add_bits_matches_integer_addition():
    rng = np.random.default_rng(0)
    T = 20
    a_int = rng.integers(0, 2 ** (T - 1), size=200)
    b_int = rng.integers(0, 2 ** (T - 1), size=200)
    bits = lambda v: (v[:, None] >> np.arange(T)) & 1  # noqa: E731

    s = add_bits(bits(a_int), bits(b_int))

    assert s.shape == (200, T)
    assert np.array_equal(to_ints(s), a_int + b_int)


def test_make_addition_format_and_correctness():
    X, y = make_addition(500, 12, np.random.default_rng(1))

    assert X.shape == (500, 12, 2) and X.dtype == np.float64
    assert y.shape == (500, 12) and np.issubdtype(y.dtype, np.integer)
    assert set(np.unique(X)) <= {0.0, 1.0}
    assert set(np.unique(y)) <= {0, 1}

    assert np.all(X[:, -1, :] == 0), "the top bit of both inputs must be 0"
    a, b = to_ints(X[:, :, 0]), to_ints(X[:, :, 1])
    assert np.array_equal(to_ints(y), a + b)
    assert len(np.unique(a)) > 100, "inputs should be random, not a fixed pattern"


def test_make_addition_is_reproducible():
    X1, y1 = make_addition(50, 8, np.random.default_rng(42))
    X2, y2 = make_addition(50, 8, np.random.default_rng(42))
    X3, _ = make_addition(50, 8, np.random.default_rng(43))
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)
