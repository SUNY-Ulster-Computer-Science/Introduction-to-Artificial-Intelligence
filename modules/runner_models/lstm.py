import math

import torch
import torch.nn.functional as F
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combine weights for all 4 gates (input, forget, cell, output) to run in parallel
        # 4 * hidden_dim handles: i_t, f_t, g_t, o_t
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))

        self.init_weights()

    def init_weights(self):
        # Xavier/Glorot initialization of parameters
        std = 1.0 / math.sqrt(self.hidden_dim)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = states

        # Parallel projection of input and previous hidden state for all gates
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h_prev, self.weight_hh, self.bias_hh)

        # Split the combined projections into individual gate components
        i_gates, f_gates, g_gates, o_gates = gates.chunk(4, dim=-1)

        # Apply gate activation functions
        i = torch.sigmoid(i_gates)  # Input gate
        f = torch.sigmoid(f_gates)  # Forget gate
        g = torch.tanh(g_gates)  # Cell gate (candidate update)
        o = torch.sigmoid(o_gates)  # Output gate

        # Update long-term memory (cell state) and short-term memory (hidden state)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, batch_first: bool = True):
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, init_states: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Handle batch_first routing shapes
        if self.batch_first:
            # Shape: (batch, seq_len, input_dim)
            batch_size, seq_len, _ = x.shape
        else:
            # Shape: (seq_len, batch, input_dim)
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)  # Convert temporarily to batch-first for easy sequence iteration

        # Initialize hidden state (h) and cell state (c) with zeros if not provided
        if init_states is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h, c = init_states
            # PyTorch expects (num_layers, batch, hidden), remove layer dim if passed
            if h.dim() == 3:
                h = h.squeeze(0)
            if c.dim() == 3:
                c = c.squeeze(0)

        outputs = []

        # Step sequentially through time
        for t in range(seq_len):
            x_t = x[:, t, :]  # Fetch token features for the current time step
            h, c = self.cell(x_t, (h, c))
            outputs.append(h.unsqueeze(1))  # Keep track of output history

        # Recombine individual time step outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        # Restore standard PyTorch shapes for hidden states (adding layer dimension)
        hn = h.unsqueeze(0)
        cn = c.unsqueeze(0)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, (hn, cn)


class ClassifierLSTM(nn.Module):
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
        self.lstm = LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed the sparse token data into dense embeddings
        # x: (batch, seq_len) token ids -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # LSTM returns: output, (hidden, cell), vanilla RNNs do not have the extra call state returned
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)  # (1, batch, hidden_dim) -> (batch, hidden_dim)
        # Linear layer from the final hidden state to per-class scores
        logits = self.linear(hidden)  # (batch, num_classes)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(logits, dim=1)


class LanguageModelLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, pad_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        # Weight tying: reuse the embedding matrix as the output projection
        self.linear.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed the sparse token data into dense embeddings
        # x: (batch, seq_len) token ids -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        # Linear layer from the final hidden state to token predictions
        logits = self.linear(output)  # (batch, seq_len, vocab_size)
        # Log softmax to convert the output into log probabilities for each class (log useful during loss evaluation)
        return F.log_softmax(logits, dim=-1)
