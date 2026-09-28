from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def add_bits(a: IntArray, b: IntArray) -> IntArray:
    """Add two batches of binary numbers, one bit position at a time.

    Bits are stored least significant first: a[i, 0] is the ones digit of
    the i-th number, a[i, 1] the twos digit, and so on. Scanning t from 0 to
    T - 1 and keeping a running carry is exactly the algorithm the RNN will
    later have to learn.

    A carry out of the last position has nowhere to go and is dropped, so
    the result is the sum modulo 2 ** T. make_addition avoids this case by
    keeping both inputs small enough.

    Args:
        a: integer array of shape (N, T) holding 0/1 bits.
        b: integer array of shape (N, T) holding 0/1 bits.

    Returns:
        Integer array of shape (N, T) holding the bits of a + b, least
        significant first.
    """
    N, T = a.shape
    result = np.zeros((N, T), dtype=np.int64)
    carry = np.zeros(N, dtype=np.int64)
    for t in range(T):
        # Full adder: sum = (a + b + carry) % 2, carry = (a + b + carry) // 2
        s = a[:, t] + b[:, t] + carry
        result[:, t] = s % 2
        carry = s // 2
    # Carry out of the last bit is dropped
    return result


def make_addition(num_samples: int, num_bits: int, rng: np.random.Generator) -> tuple[Array, IntArray]:
    """Sample a batch of binary addition problems.

    Each problem is a pair of random num_bits-bit numbers a and b whose
    highest bit (index num_bits - 1) is forced to 0, so that a + b always
    fits in num_bits bits and add_bits never drops a carry.

    Args:
        num_samples: N, the number of problems.
        num_bits: T, the sequence length.
        rng: source of randomness. Draw every random bit from it (and from
            nothing else), so that the same seed gives the same batch.

    Returns:
        X: float array of shape (N, T, 2). X[i, t, 0] and X[i, t, 1] are bit
            t of a and of b for the i-th problem, least significant first.
        y: integer array of shape (N, T) holding the bits of a + b, least
            significant first.
    """
    # Generate random bits for all but the highest position
    # Shape: (N, num_bits - 1) for each of a and b
    a_bits = rng.integers(0, 2, size=(num_samples, num_bits - 1), dtype=np.int64)
    b_bits = rng.integers(0, 2, size=(num_samples, num_bits - 1), dtype=np.int64)

    # Append zero at the highest bit position
    a_full = np.concatenate([a_bits, np.zeros((num_samples, 1), dtype=np.int64)], axis=1)
    b_full = np.concatenate([b_bits, np.zeros((num_samples, 1), dtype=np.int64)], axis=1)

    # Build X: (N, T, 2)
    X = np.stack([a_full, b_full], axis=2).astype(np.float64)

    # Build y: result of a + b
    y = add_bits(a_full, b_full)

    return X, y


# ---------------------------------------------------------------------------
# The helpers below are already written and need no changes. They convert
# between Python integers (of any size) and bit vectors, least significant
# bit first.


def int_to_bits(value: int, num_bits: int) -> IntArray:
    """Bits of a non-negative Python int, least significant first, shape (num_bits,)."""
    if value < 0 or value >= 2**num_bits:
        raise ValueError(f"{value} does not fit in {num_bits} bits")
    return np.array([(value >> t) & 1 for t in range(num_bits)], dtype=np.int64)


def bits_to_int(bits: IntArray) -> int:
    """Inverse of int_to_bits: a 1-D array of 0/1 bits, least significant first."""
    return sum(int(bit) << t for t, bit in enumerate(bits))
