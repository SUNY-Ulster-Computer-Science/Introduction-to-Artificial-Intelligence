"""A small byte-level BPE (byte-pair encoding) tokenizer."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import tqdm


class BPETokenizer:
    """Byte-level BPE tokenizer: learns a fixed number of merges over the training corpus's byte pairs, then applies
    those merges (in the order they were learned) to tokenize new text until no more are available."""

    def __init__(self) -> None:
        # (id_a, id_b) -> merged_id, in the order merges were learned. Since Python dicts preserve insertion order,
        # iterating self.merges also gives the learned merge priority (earlier = higher priority), for encode().
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab_size: int = 256  # the 256 raw byte values, before any merges

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the base byte vocabulary)")

        sequences = [list(text.encode("utf-8")) for text in texts]
        num_merges = vocab_size - 256
        next_id = 256

        for _ in tqdm.tqdm(range(num_merges), desc="Training BPE", unit="merge"):
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break  # Corpus is fully merged down to single tokens: nothing left to merge
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

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
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
        return ids

    def _id_to_bytes_table(self) -> dict[int, bytes]:
        table = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            table[new_id] = table[a] + table[b]
        return table

    def decode(self, ids: list[int]) -> str:
        table = self._id_to_bytes_table()
        raw = b"".join(table[i] for i in ids)
        # errors="replace" rather than raising: generated/sampled id seq aren't guaranteed to be on a UTF-8 boundary.
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
