"""A small word-level tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


class WordTokenizer:
    """Word-level tokenizer: builds a vocabulary from the most frequent words in a training corpus.

    Unknown words are mapped to <UNK>.
    """

    def __init__(self) -> None:
        # Maps word string to a unique token integer ID
        self.word_to_id: dict[str, int] = {"<UNK>": 0}
        # Maps token integer ID to a word string
        self.id_to_word: dict[int, str] = {0: "<UNK>"}
        self.vocab_size: int = 1

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")

        # Count frequencies of all whitespace-separated words across texts
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(text.split())

        # Select the most common words to fill up the remaining vocabulary slots
        max_words_to_add = vocab_size - 1
        most_common = word_counts.most_common(max_words_to_add)

        # Build vocabulary mappings
        self.word_to_id = {"<UNK>": 0}
        self.id_to_word = {0: "<UNK>"}

        next_id = 1
        for word, _ in most_common:
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        self.vocab_size = next_id

    def encode(self, text: str) -> list[int]:
        # Split text by whitespace and map to IDs, defaulting to <UNK> (0)
        return [self.word_to_id.get(word, 0) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        # Map IDs back to words and join them with spaces
        # Out-of-bounds IDs are safely fallback to <UNK>
        return " ".join(self.id_to_word.get(token_id, "<UNK>") for token_id in ids)

    def save(self, path: Path) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> WordTokenizer:
        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.word_to_id = data["word_to_id"]
        # Reconstruct the reverse mapping, ensuring the dictionary keys are ints
        tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}
        return tokenizer
