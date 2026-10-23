import numpy as np
import pytest

from src.tokenizer import NUM_BYTES, BPETokenizer, compression_ratio, sliding_window


def _trained(text="abcabc", vocab_size=258, allowed_special=frozenset()):
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size, allowed_special=allowed_special)
    return tok


def test_pair_stats_counts_overlapping_pairs():
    """[1, 2, 1, 2] has two copies of (1, 2) and one of (2, 1)."""
    stats = BPETokenizer._pair_stats([1, 2, 1, 2])
    assert stats[(1, 2)] == 2
    assert stats[(2, 1)] == 1


def test_most_frequent_pair_breaks_ties_deterministically():
    """Equal counts must resolve the same way every run: smallest pair wins."""
    stats = BPETokenizer._pair_stats([5, 9, 3, 9])  # (5,9),(9,3),(3,9) all once
    assert BPETokenizer._most_frequent_pair(stats) == (3, 9)


def test_merge_replaces_pair():
    assert BPETokenizer()._merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_is_non_overlapping_left_to_right():
    """[a, a, a] with pair (a, a) becomes [new, a], not [a, new]."""
    assert BPETokenizer()._merge([7, 7, 7], (7, 7), 42) == [42, 7]


def test_merge_leaves_absent_pair_untouched():
    assert BPETokenizer()._merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]


def test_train_reaches_vocab_size():
    tok = _trained("abcabc", vocab_size=258)
    assert len(tok.vocab) == 258
    assert NUM_BYTES == 256


def test_train_learns_frequent_pair_bytes():
    """On 'abcabc' the first merge is (a, b) -> 'ab', then ('ab', c) -> 'abc'."""
    tok = _trained("abcabc", vocab_size=258)
    assert tok.vocab[NUM_BYTES] == b"ab"
    assert tok.vocab[NUM_BYTES + 1] == b"abc"


def test_encode_replays_merges_to_longest_token():
    tok = _trained("abcabc", vocab_size=258)
    assert tok.encode("abc") == [NUM_BYTES + 1]


def test_encode_decode_roundtrip():
    tok = _trained("the quick brown fox " * 30, vocab_size=300)
    for s in ["the quick brown fox", "quick brown", "the fox"]:
        assert tok.decode(tok.encode(s)) == s


def test_encode_respects_special_token_boundary():
    """A special token must come out as its own id, never byte-split."""
    tok = _trained("hello world hello", vocab_size=270,
                   allowed_special=frozenset({"<|endoftext|>"}))
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("hi<|endoftext|>hi", allowed_special=frozenset({"<|endoftext|>"}))
    assert ids.count(eot) == 1
    # decoding reconstructs the literal special token text
    assert tok.decode(ids) == "hi<|endoftext|>hi"


def test_train_reserves_room_for_specials_in_vocab_size():
    # long enough that all 13 merges run to completion
    tok = _trained("the quick brown fox jumps over the lazy dog " * 5,
                   vocab_size=270, allowed_special=frozenset({"<|endoftext|>"}))
    # 256 bytes + 13 merges + 1 special == vocab_size
    assert len(tok.vocab) + len(tok.special_tokens) == 270


def test_compression_higher_in_corpus_than_out():
    """A tokenizer compresses the text it was trained on better than novel text."""
    tok = _trained("the quick brown fox jumps over the lazy dog " * 50, vocab_size=300)
    in_corpus = compression_ratio("the quick brown fox jumps over the lazy dog", tok)
    out_corpus = compression_ratio("zqwxv kjrbt nmplf gsdkh", tok)
    assert in_corpus > out_corpus


def test_sliding_window_shapes():
    ids = np.arange(20)
    inputs, targets = sliding_window(ids, max_length=4, stride=4)
    assert inputs.shape == targets.shape == (4, 4)
    assert inputs.dtype == targets.dtype == np.int64


def test_sliding_window_targets_shifted_by_one():
    ids = np.arange(20)
    inputs, targets = sliding_window(ids, max_length=4, stride=4)
    assert np.array_equal(inputs[:, 1:], targets[:, :-1])


def test_sliding_window_first_sample_values():
    ids = np.arange(10)
    inputs, targets = sliding_window(ids, max_length=4, stride=4)
    assert np.array_equal(inputs[0], [0, 1, 2, 3])
    assert np.array_equal(targets[0], [1, 2, 3, 4])


def test_sliding_window_overlap_when_stride_small():
    """stride < max_length makes consecutive windows share tokens."""
    ids = np.arange(12)
    inputs, _ = sliding_window(ids, max_length=4, stride=2)
    # samples start at 0, 2, 4, ...
    assert inputs.shape[0] == 4
    assert np.array_equal(inputs[0][2:], inputs[1][:2])  # overlap region


def test_sliding_window_no_overlap_when_stride_equal():
    """stride == max_length tiles the sequence into disjoint windows."""
    ids = np.arange(12)
    inputs, _ = sliding_window(ids, max_length=4, stride=4)
    # windows start at 0 and 4, so consecutive inputs sit back to back
    assert np.array_equal(inputs[:, 0], [0, 4])
    assert inputs[0][-1] + 1 == inputs[1][0]  # disjoint and adjacent


def test_sliding_window_too_short_raises():
    with pytest.raises(ValueError):
        sliding_window(np.arange(4), max_length=4, stride=1)
