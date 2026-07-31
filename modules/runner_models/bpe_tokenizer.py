"""A small byte-level BPE (byte-pair encoding) tokenizer."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import tqdm


class BPETokenizer:
    """Byte-level BPE tokenizer with padding.

    Reserves ID 0 for <PAD>, maps raw bytes to 1-256, and then learns fixed merges over training corpus byte pairs.
    """

    def __init__(self) -> None:
        # (id_a, id_b) -> merged_id, in the order merges were learned.
        self.merges: dict[tuple[int, int], int] = {}
        # 1 pad token + 256 raw byte values = 257 base tokens before merges
        self.vocab_size: int = 257

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (1 pad token + 256 base bytes)")

        # Convert text to UTF-8 bytes and shift up by 1 to avoid overlapping with PAD (0)
        sequences = [[b + 1 for b in text.encode("utf-8")] for text in texts]
        num_merges = vocab_size - 257
        next_id = 257

        for _ in tqdm.tqdm(range(num_merges), desc="Training BPE", unit="merge"):
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break  # Corpus is fully merged down to single tokens
            best_pair = max(pair_counts, key=pair_counts.get)
            sequences = [self._apply_merge(seq, best_pair, next_id) for seq in sequences]
            self.merges[best_pair] = next_id
            next_id += 1

        self.vocab_size = next_id

    @staticmethod
    def _count_pairs(sequences: list[list[int]]) -> Counter:
        counts: Counter = Counter()
        for seq in sequences:
            for a, b in itertools.pairwise(seq):
                counts[(a, b)] += 1
        return counts

    @staticmethod
    def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
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

    def encode(self, text: str, max_length: int | None = None, padding: bool = False) -> list[int]:
        # Shift initial bytes up by 1
        ids = [b + 1 for b in text.encode("utf-8")]

        while len(ids) >= 2:
            pairs_present = {(a, b) for a, b in itertools.pairwise(ids)}
            # Apply whichever merge was learned earliest (lowest assigned id among adjacent pairs still present).
            candidate = min(
                (p for p in pairs_present if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._apply_merge(ids, candidate, self.merges[candidate])

        # Handle truncation and padding alignment
        if max_length is None:
            return ids

        if len(ids) > max_length:
            ids = ids[:max_length]
        elif len(ids) < max_length and padding:
            ids += [0] * (max_length - len(ids))

        return ids

    def _id_to_bytes_table(self) -> dict[int, bytes]:
        # Initialize table mapping IDs back to their original byte arrays
        # ID 0 is explicitly empty bytes so it naturally vanishes if joined raw
        table = {0: b""}
        for i in range(256):
            table[i + 1] = bytes([i])

        for (a, b), new_id in self.merges.items():
            table[new_id] = table[a] + table[b]
        return table

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        table = self._id_to_bytes_table()

        # Filter out the pad token ID (0) if requested
        if skip_special_tokens:
            filtered_ids = [token_id for token_id in ids if token_id != 0]
        else:
            filtered_ids = ids

        raw = b"".join(table.get(i, b"") for i in filtered_ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        return tokenizer
