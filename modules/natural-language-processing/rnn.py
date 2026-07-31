"""Example module for the runner: a simple RNN trained on IMDB movie review sentiment."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modules.runner_modules.imdb_runner import BaseIMDBRunner

MODULE_DIR = Path(__file__).parent


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed the sparse token data into dense embeddings
        # x: (batch, seq_len) token ids -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # Run the embedded sequence through the RNN one token at a time. We only need the final hidden state, used here
        # as the input to the classifier head.
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)  # (1, batch, hidden_dim) -> (batch, hidden_dim)
        # Linear layer from the final hidden state to per-class scores
        logits = self.fc(hidden)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(logits, dim=1)


class IMDBRunner(BaseIMDBRunner):
    _model_class: type = RNN
    _model_path: Path = MODULE_DIR / "rnn_imdb.pt"
    _vocab_path: Path = MODULE_DIR / "imdb_vocab.json"
    _label_names: list[str] = ["negative", "positive"]
