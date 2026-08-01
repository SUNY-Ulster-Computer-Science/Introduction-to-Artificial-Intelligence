"""A small byte-level BPE (byte-pair encoding) tokenizer with caching and padding support."""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import tqdm

# Splits text into alternating runs of non-whitespace and whitespace ("word" chunks and "gap" chunks). BPE merges are
# applied within each chunk independently, never across a chunk boundary.
_PRETOKEN_RE = re.compile(r"\S+|\s+")


class BPETokenizer:
    """Byte-level BPE tokenizer: learns a fixed number of merges over the training corpus's byte pairs, then applies
    those merges (in the order they were learned) to tokenize new text."""

    def __init__(self) -> None:
        # (id_a, id_b) -> merged_id, in the order merges were learned.
        self.merges: dict[tuple[int, int], int] = {}
        # 1 pad token + 256 raw byte values = 257 base tokens before merges
        self.vocab_size: int = 257
        self.padding_side = "right"
        self._encode_cache: dict[str, list[int]] = {}

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        """Train the tokenizer by learning BPE merges over a corpus.

        Args:
            texts: Iterable of training strings to learn merges from.
            vocab_size: Target vocabulary size, must be >= 257.
        """

        self._encode_cache.clear()  # Old cached results are invalid once self.merges changes
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (1 pad token + 256 base bytes)")

        # Split each document into pre-token chunks (same regex encode() uses) before building the working sequences.
        sequences: list[list[int]] = []
        for text in texts:
            for chunk in _PRETOKEN_RE.findall(text):
                sequences.append([b + 1 for b in chunk.encode("utf-8")])

        next_id = self._merge_all_pairs(vocab_size, sequences)
        self.vocab_size = next_id

    @staticmethod
    def _count_pairs(sequences: list[list[int]]) -> Counter:
        """Count all adjacent token id pairs across a list of sequences.

        Args:
            sequences: List of token id sequences to count pairs in.
        Returns:
            Counter mapping each (id_a, id_b) pair to its total count.
        """

        counts: Counter = Counter()
        for seq in sequences:
            for a, b in itertools.pairwise(seq):
                counts[(a, b)] += 1
        return counts

    @staticmethod
    def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all non-overlapping occurrences of a pair in a sequence with a new id.

        Args:
            seq: Input token id sequence.
            pair: The (id_a, id_b) pair to merge.
            new_id: The id to replace each occurrence of the pair with.
        Returns:
            New sequence with all occurrences of pair replaced by new_id.
        """

        merged: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        return merged

    def _build_global_pairs(self, sequences: list[list[int]]) -> tuple[Counter, dict[tuple[int, int], set[int]]]:
        """Build global pair counts and a reverse index of which sequences contain each pair.

        Args:
            sequences: List of token id sequences to index.
        Returns:
            Tuple of (pair_counts, pair_to_seqs) where pair_counts maps each pair
            to its total count, and pair_to_seqs maps each pair to the set of
            sequence indices that contain it.
        """

        pair_counts: Counter = Counter()
        pair_to_seqs: dict[tuple[int, int], set[int]] = {}
        for seq_idx, seq in enumerate(sequences):
            for pair, count in self._count_pairs([seq]).items():
                pair_counts[pair] += count
                pair_to_seqs.setdefault(pair, set()).add(seq_idx)

        return pair_counts, pair_to_seqs

    def _merge_all_pairs(self, vocab_size: int, sequences: list[list[int]]) -> int:
        """Iteratively learn and apply the most frequent merge until the target vocab size is reached.

        Args:
            vocab_size: Target vocabulary size to train up to.
            sequences: Pre-tokenized training sequences (already shifted to 1-indexed byte ids).
        Returns:
            The next available token id after all merges have been learned.
        """

        next_id = 257 + len(self.merges)
        num_merges = vocab_size - next_id

        # One-time full pass: build global pair counts, plus an index of which documents contain each pair.
        pair_counts, pair_to_seqs = self._build_global_pairs(sequences)

        for _ in tqdm.tqdm(range(num_merges), desc="Training BPE", unit="merge"):
            if not pair_counts:
                break  # Corpus is fully merged down to single tokens: nothing left to merge

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            affected_seqs = set(pair_to_seqs.get(best_pair, set()))

            for seq_idx in affected_seqs:
                old_seq = sequences[seq_idx]
                old_local_counts = self._count_pairs([old_seq])

                new_seq = self._apply_merge(old_seq, best_pair, next_id)
                sequences[seq_idx] = new_seq

                new_local_counts = self._count_pairs([new_seq])

                # Undo this document's old contribution to the global counts/index, then add its new contribution
                for pair, count in old_local_counts.items():
                    pair_counts[pair] -= count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_seqs[pair].discard(seq_idx)
                    if not pair_to_seqs[pair]:
                        del pair_to_seqs[pair]
                for pair, count in new_local_counts.items():
                    pair_counts[pair] += count
                    pair_to_seqs.setdefault(pair, set()).add(seq_idx)

            self.merges[best_pair] = next_id
            next_id += 1

        return next_id

    def encode(self, text: str, max_length: int | None = None, padding: bool = False) -> list[int]:
        """Encode a string into a list of token ids.

        Args:
            text: Input string to encode.
            max_length: If set, truncates or pads the output to exactly this length.
            padding: If True and the output is shorter than max_length, pad to max_length using id 0. Padding side is
                controlled by self.padding_side. Has no effect if max_length is None.
        Returns:
            List of integer token ids.
        """

        ids: list[int] = []

        for chunk in _PRETOKEN_RE.findall(text):
            ids.extend(self._encode_chunk(chunk))

        # Handle truncation and padding alignment
        if max_length is None:
            return ids

        if len(ids) > max_length:
            ids = ids[:max_length]
        elif len(ids) < max_length and padding:
            # Left or right pad out to max length using ID 0 (<PAD>)
            if self.padding_side == "right":
                ids += [0] * (max_length - len(ids))
            else:
                ids = [0] * (max_length - len(ids)) + ids

        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a single pre-tokenized chunk string into BPE token ids, using a cache.

        Args:
            chunk: A single whitespace-free or whitespace-only string chunk.
        Returns:
            List of BPE token ids for this chunk.
        """

        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached

        chunk_ids = [b + 1 for b in chunk.encode("utf-8")]
        while len(chunk_ids) >= 2:
            pairs_present = {(a, b) for a, b in itertools.pairwise(chunk_ids)}
            # Apply whichever merge was learned earliest (lowest assigned id among adjacent pairs still present).
            candidate = min(
                (p for p in pairs_present if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            chunk_ids = self._apply_merge(chunk_ids, candidate, self.merges[candidate])

        self._encode_cache[chunk] = chunk_ids
        return chunk_ids

    def _id_to_bytes_table(self) -> dict[int, bytes]:
        """Build a lookup table mapping every token id back to its raw byte sequence.

        Returns:
            Dict mapping token id to the bytes it represents. Id 0 maps to empty
            bytes so pad tokens naturally vanish when joined.
        """

        # ID 0 is explicitly empty bytes so it naturally vanishes if joined raw
        table = {0: b""}
        for i in range(256):
            table[i + 1] = bytes([i])

        for (a, b), new_id in self.merges.items():
            table[new_id] = table[a] + table[b]
        return table

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back into a string.

        Args:
            ids: List of token ids to decode.
            skip_special_tokens: If True, pad tokens (id 0) are removed before decoding.
        Returns:
            Decoded string. Any undecodeable byte sequences are replaced with the
            Unicode replacement character.
        """

        table = self._id_to_bytes_table()

        # Filter out the pad token ID (0) if requested
        if skip_special_tokens:
            ids = [i for i in ids if i != 0]

        raw = b"".join(table.get(i, b"") for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        """Serialize the tokenizer's merges and vocab size to a JSON file.

        Args:
            path: Destination file path to write to.
        """

        data = {
            "vocab_size": self.vocab_size,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        """Load a previously saved tokenizer from a JSON file.

        Args:
            path: Path to the JSON file written by save().
        Returns:
            A BPETokenizer with merges and vocab_size restored.
        """

        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        return tokenizer
