import math

import torch
import torch.nn.functional as F
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Single projection weights and biases for the hidden state update
        self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(hidden_dim))

        self.init_weights()

    def init_weights(self):
        # Uniform initialization
        std = 1.0 / math.sqrt(self.hidden_dim)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Standard Elman RNN update: hidden = tanh(W_ih * x + b_ih + W_hh * h_prev + b_hh)
        h_next = torch.tanh(F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h_prev, self.weight_hh, self.bias_hh))
        return h_next


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, batch_first: bool = True):
        super().__init__()
        self.cell = RNNCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor, init_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle batch_first structural permutation
        if self.batch_first:
            # Shape: (batch, seq_len, input_dim)
            batch_size, seq_len, _ = x.shape
        else:
            # Shape: (seq_len, batch, input_dim)
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)  # Temporarily flip to iterate over time safely

        # Initialize the hidden state with zeros if not provided
        if init_state is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h = init_state
            # Remove the layer dimension if passed from an external PyTorch-like loop
            if h.dim() == 3:
                h = h.squeeze(0)

        outputs = []

        # Loop through each sequential time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Fetch token features for the current step
            h = self.cell(x_t, h)
            outputs.append(h.unsqueeze(1))  # Track the state history

        # Stack outputs back together into a single tensor
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        # Re-add layer dimension to mirror standard PyTorch API behavior
        hn = h.unsqueeze(0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, hn


class ClassifierRNN(nn.Module):
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
        self.rnn = RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed the sparse token data into dense embeddings
        # x: (batch, seq_len) token ids -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # Run the embedded sequence through the RNN one token at a time. We only need the final hidden state, used here
        # as the input to the classifier head.
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)  # (1, batch, hidden_dim) -> (batch, hidden_dim)
        # Linear layer from the final hidden state to per-class scores
        logits = self.linear(hidden)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(logits, dim=1)


class LanguageModelRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, pad_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        # Weight tying: reuse the embedding matrix as the output projection
        self.linear.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token ids -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # Unlike the classifier RNN, we need the output at every timestep, not just the final hidden state.
        output, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim)
        logits = self.linear(output)  # (batch, seq_len, vocab_size)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(logits, dim=-1)
