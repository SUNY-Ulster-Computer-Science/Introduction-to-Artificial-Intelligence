"""A simple word-level tokenizer with padding support."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Lowercased runs of letters and apostrophes.
_TOKEN_RE = re.compile(r"[a-z']+")


class WordTokenizer:
    """Word-level tokenizer: builds a vocabulary from the most frequent case-insensitive words in a training corpus.
    Unknown words are mapped to <UNK> and sequences can be padded with <PAD>.
    """

    def __init__(self) -> None:
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word_to_id: dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_word: dict[int, str] = {0: self.pad_token, 1: self.unk_token}
        self.vocab_size: int = 2
        self.padding_side = "right"

    def train(self, texts: Iterable[str], vocab_size: int) -> None:
        """Build a vocabulary from the most frequent words in a corpus.

        Words are lowercased and extracted with the _TOKEN_RE pattern. The two lowest ids are always reserved for
        <PAD> (0) and <UNK> (1), so the maximum number of real words in the vocabulary is vocab_size - 2.

        Args:
            texts: Iterable of training strings to count word frequencies from.
            vocab_size: Target vocabulary size, must be >= 2.
        """

        if vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2 (for {self.pad_token} and {self.unk_token})")

        # Count frequencies of all words across texts
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(_TOKEN_RE.findall(text.lower()))

        # Subtract 2 to leave room for both special tokens
        max_words_to_add = vocab_size - 2
        most_common = word_counts.most_common(max_words_to_add)

        self.word_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_word = {0: self.pad_token, 1: self.unk_token}

        next_id = 2
        for word, _ in most_common:
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        self.vocab_size = next_id

    def encode(self, text: str, max_length: int | None = None, padding: bool = False) -> list[int]:
        """Encode a string into a list of token ids.

        Args:
            text: Input string to encode.
            max_length: If set, truncates or pads the output to exactly this length.
            padding: If True and the output is shorter than max_length, pad to max_length using id 0. Padding side is
                controlled by self.padding_side. Has no effect if max_length is None.
        Returns:
            List of integer token ids, with unknown words mapped to id 1.
        """

        # Fallback defaults to 1 (<UNK>)
        ids = [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in _TOKEN_RE.findall(text.lower())]

        if max_length is None:
            return ids

        if len(ids) > max_length:
            # Truncate down to max length
            ids = ids[:max_length]
        elif len(ids) < max_length and padding:
            # Left pad out to max length using ID 0 (<PAD>)
            if self.padding_side == "right":
                ids += [self.word_to_id[self.pad_token]] * (max_length - len(ids))
            else:
                ids = [self.word_to_id[self.pad_token]] * (max_length - len(ids)) + ids

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back into a whitespace-separated string.

        Args:
            ids: List of token ids to decode.
            skip_special_tokens: If True, pad tokens are removed before joining.
                Unknown token ids not present in the vocabulary are decoded as <UNK>.
        Returns:
            Whitespace-separated string of decoded words.
        """

        # Map IDs back to words and join them with spaces
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            if skip_special_tokens and word == self.pad_token:
                continue  # Skip padding tokens entirely during reconstruction
            words.append(word)
        return " ".join(words)

    def save(self, path: Path) -> None:
        """Serialize the tokenizer's vocabulary to a JSON file.

        Args:
            path: Destination file path to write to.
        """

        data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> WordTokenizer:
        """Load a previously saved tokenizer from a JSON file.

        Args:
            path: Path to the JSON file written by save().
        Returns:
            A WordTokenizer with vocabulary and vocab_size restored.
        """

        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.word_to_id = data["word_to_id"]
        # Reconstruct the reverse mapping, ensuring the dictionary keys are ints
        tokenizer.id_to_word = {int(v): k for k, v in tokenizer.word_to_id.items()}
        return tokenizer
