from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


class WordTokenizer:
    """Word-level tokenizer: builds a vocabulary from the most frequent case-insensitive words in a training corpus.
    Unknown words are mapped to <UNK> and sequences can be padded with <PAD>.
    """

    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size: int = 2
        self.padding_side = "right"

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2 (for <PAD> and <UNK>)")

        # Count frequencies of all whitespace-separated words across texts
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(text.lower().split())

        # Subtract 2 to leave room for both special tokens
        max_words_to_add = vocab_size - 2
        most_common = word_counts.most_common(max_words_to_add)

        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}

        next_id = 2
        for word, _ in most_common:
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        self.vocab_size = next_id

    def encode(self, text: str, max_length: int | None = None, padding: bool = False) -> list[int]:
        # Fallback defaults to 1 (<UNK>)
        ids = [self.word_to_id.get(word, 1) for word in text.lower().split()]

        if max_length is None:
            return ids

        if len(ids) > max_length:
            # Truncate down to max length
            ids = ids[:max_length]
        elif len(ids) < max_length and padding:
            # Left pad out to max length using ID 0 (<PAD>)
            if self.padding_side == "right":
                ids += [0] * (max_length - len(ids))
            else:
                ids = [0] * (max_length - len(ids)) + ids

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Map IDs back to words and join them with spaces
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, "<UNK>")
            if skip_special_tokens and word == "<PAD>":
                continue  # Skip padding tokens entirely during reconstruction
            words.append(word)
        return " ".join(words)

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
