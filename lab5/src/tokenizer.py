from __future__ import annotations

import re
from collections import Counter

import numpy as np

# A byte-level BPE vocabulary always starts from the 256 possible byte values.
NUM_BYTES = 256


class BPETokenizer:
    """A byte-level byte-pair-encoding tokenizer, trained from scratch.

    The vocabulary begins as the 256 raw byte values (ids 0..255). Training
    then repeatedly merges the most frequent adjacent pair of ids into a new
    id, growing the vocabulary one token at a time, and finally appends any
    special tokens (such as ``<|endoftext|>``) at the end of the vocabulary.

    Encoding works in reverse: text is turned into its byte ids, and the
    learned merges are replayed in the order they were learned (earliest merge
    first) until no adjacent pair can be merged any further.

    Attributes:
        vocab: mapping from token id to the bytes it stands for.
        merges: mapping from an adjacent id pair ``(a, b)`` to the id that
            replaces it. Insertion order is the order the merges were learned,
            which is also their priority during encoding.
        special_tokens: mapping from a special token string to its id.
        inverse_special_tokens: the reverse of ``special_tokens``.
    """

    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
    special_tokens: dict[str, int]
    inverse_special_tokens: dict[int, str]

    def __init__(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(NUM_BYTES)}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    # -- helpers given to you -------------------------------------------

    @staticmethod
    def _pair_stats(ids: list[int]) -> Counter:
        """Count how often each adjacent pair occurs in a token-id sequence.

        Args:
            ids: the current sequence of token ids.

        Returns:
            A Counter mapping each adjacent pair ``(ids[i], ids[i+1])`` to
            the number of times it appears. Overlapping pairs all count, so
            ``[1, 2, 1, 2]`` contributes two to ``(1, 2)`` and one to
            ``(2, 1)``.
        """
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _most_frequent_pair(stats: Counter) -> tuple[int, int]:
        """Pick the pair to merge next from a frequency table.

        The most frequent pair wins. Ties are broken deterministically by
        taking the lexicographically smallest pair, so that a fixed corpus
        always yields the same tokenizer.

        Args:
            stats: a non-empty Counter of adjacent-pair frequencies.

        Returns:
            The ``(a, b)`` pair with the highest count.
        """
        return min(stats, key=lambda p: (-stats[p], p))

    # -- tasks ----------------------------------------------------------

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of ``pair`` with ``new_id``.

        Scan left to right; whenever the next two ids equal ``pair``, emit
        ``new_id`` and skip both. Otherwise emit the current id and advance by
        one. Left-to-right scanning means ``[a, a, a]`` with pair ``(a, a)``
        becomes ``[new_id, a]``, not ``[a, new_id]``.

        Args:
            ids: the token-id sequence to rewrite.
            pair: the adjacent pair ``(a, b)`` to replace.
            new_id: the id that stands for the merged pair.

        Returns:
            A new list of token ids with the pair merged throughout.
        """
        merged: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, text: str, vocab_size: int, allowed_special: frozenset = frozenset()) -> None:
        """Learn merge rules from ``text`` until the vocabulary is full.

        Starting from the 256 byte ids, repeatedly find the most frequent
        adjacent pair, register it as a new token (whose byte string is the
        concatenation of its two parts), and rewrite the training sequence.
        Stop once ``vocab_size`` tokens exist -- counting the ``NUM_BYTES``
        initial tokens and one slot per special token -- or when no adjacent
        pair remains to merge.

        Args:
            text: the training corpus. Assumed not to contain the special
                tokens; they are reserved, not learned from data.
            vocab_size: total vocabulary size including bytes and specials.
            allowed_special: special token strings to reserve ids for, placed
                at the very end of the vocabulary.

        Raises:
            ValueError: if ``vocab_size`` is too small to hold the bytes and
                the requested special tokens.
        """
        if vocab_size < NUM_BYTES + len(allowed_special):
            raise ValueError(
                f"vocab_size {vocab_size} is too small for {len(allowed_special)} special tokens"
            )
        # Special tokens sit at the very end of the vocabulary, so merges may
        # only use ids up to (but not including) this first reserved id.
        special_start = vocab_size - len(allowed_special)
        self.special_tokens = {
            s: i for i, s in enumerate(sorted(allowed_special), start=special_start)
        }
        self.inverse_special_tokens = {i: s for s, i in self.special_tokens.items()}
        ids = list(text.encode("utf-8"))
        while len(self.vocab) < special_start:
            stats = self._pair_stats(ids)
            if not stats:
                break
            pair = self._most_frequent_pair(stats)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = new_id
            ids = self._merge(ids, pair, new_id)

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text that is known to contain no special tokens.

        Turn the text into its UTF-8 byte ids, then replay the learned merges:
        repeatedly find the adjacent pair whose merge was learned earliest
        (lowest merge id) and merge it, until no adjacent pair appears in
        ``self.merges``. Replaying in learning order is what makes encoding
        consistent with training.

        Args:
            text: a string with no special tokens in it.

        Returns:
            The list of token ids.
        """
        ids = list(text.encode("utf-8"))
        while True:
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            merge_candidates = [(pair, self.merges[pair]) for pair in pairs if pair in self.merges]
            if not merge_candidates:
                break
            # Find the pair with the lowest new_id (earliest learned)
            pair_to_merge, _ = min(merge_candidates, key=lambda x: x[1])
            new_id = self.merges[pair_to_merge]
            ids = self._merge(ids, pair_to_merge, new_id)
        return ids

    # -- given: special-token dispatch and decoding ----------------------

    def encode(self, text: str, allowed_special: frozenset = frozenset()) -> list[int]:
        """Encode text, routing special tokens around the BPE machinery.

        Special tokens are never split into bytes: each occurrence is emitted
        as its reserved id, and only the ordinary text between them goes
        through ``_encode_ordinary``.

        Args:
            text: the string to encode.
            allowed_special: which special tokens to recognize. A special
                token that is not allowed is treated as ordinary text.

        Returns:
            The list of token ids.
        """
        specials = [s for s in allowed_special if s in self.special_tokens]
        if not specials:
            return self._encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into a string.

        Each id is looked up in ``vocab`` (or in ``inverse_special_tokens``
        for specials), the byte strings are concatenated, and the result is
        decoded as UTF-8. Invalid byte sequences are replaced rather than
        raising, mirroring ``errors="replace"`` in the reference stack.

        Args:
            ids: the token ids to decode.

        Returns:
            The decoded string.
        """
        chunks = []
        for i in ids:
            if i in self.vocab:
                chunks.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                chunks.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                raise KeyError(f"unknown token id: {i}")
        return b"".join(chunks).decode("utf-8", errors="replace")


def sliding_window(token_ids, max_length: int, stride: int):
    """Build (input, target) training pairs by sliding a window over the ids.

    A language model learns to predict the next token, so every window of
    ``max_length + 1`` consecutive ids yields one sample: the first
    ``max_length`` ids are the input, and the same window shifted right by one
    (the last ``max_length`` ids) is the target. Consecutive windows start
    ``stride`` positions apart, so ``stride < max_length`` makes samples
    overlap.

    Args:
        token_ids: the full token-id sequence, shape ``(T,)``.
        max_length: length of each input and target sequence.
        stride: how far the window advances between samples.

    Returns:
        A pair ``(inputs, targets)`` of integer arrays, each of shape
        ``(num_samples, max_length)``. ``targets[i]`` is exactly
        ``inputs[i]`` shifted one position to the right, so
        ``inputs[i][1:] == targets[i][:-1]`` must hold.

    Raises:
        ValueError: if the token sequence is not longer than ``max_length``,
            so not even one window fits.
    """
    if len(token_ids) <= max_length:
        raise ValueError("token sequence must be longer than max_length")

    inputs: list = []
    targets: list = []
    for s in range(0, len(token_ids) - max_length, stride):
        inputs.append(token_ids[s : s + max_length])
        targets.append(token_ids[s + 1 : s + max_length + 1])
    return (
        np.array(inputs, dtype=np.int64),
        np.array(targets, dtype=np.int64),
    )


def compression_ratio(text: str, tokenizer) -> float:
    """Measure how many UTF-8 bytes a tokenizer packs into each token.

    Args:
        text: the string to encode.
        tokenizer: any object with an ``encode(text) -> ids`` method.

    Returns:
        ``len(text.encode("utf-8")) / len(ids)``: larger means better
        compression on this text.
    """
    ids = tokenizer.encode(text)
    return len(text.encode("utf-8")) / len(ids)
