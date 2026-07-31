import math

import torch
import torch.nn.functional as F
from cuda.bindings.nvml import InvalidArgumentError
from torch import nn


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional information to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, : x.size(1)]


class CausalMultiHeadAttention(nn.Module):
    """Custom Multi-Head Attention layer with autoregressive masking."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()

        if d_model % nhead != 0:
            raise InvalidArgumentError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # Scaling factor for attention

        # Combined projection for Query, Key, and Value
        self.qkv_projection = nn.Linear(d_model, d_model * 3)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape  # Batch size, Sequence length, Embedding dimension

        # Project to Q, K, V and split into heads
        qkv = self.qkv_projection(x)  # (B, S, 3 * E)
        q, k, v = qkv.chunk(3, dim=-1)  # Three tensors of (B, S, E) each

        # Reshape to (B, nhead, S, d_k) for parallel attention computation
        q = q.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # Shape of scores: (B, nhead, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (look-ahead mask)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask  # Broadcasts automatically across Batch and Heads

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Context aggregation
        # Shape: (B, nhead, S, d_k)
        context = torch.matmul(attention_weights, v)

        # Concatenate heads back together
        # Shape: (B, S, E)
        context = context.transpose(1, 2).contiguous().view(B, S, E)

        return self.out_projection(context)


class TransformerDecoderLayer(nn.Module):
    """A single Transformer Decoder block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, nhead)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm residual paths
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    """The full transformer decoder architecture stack."""

    def __init__(
        self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 512
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Stack decoder layers using ModuleList
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)]
        )

        self.ln_final = nn.LayerNorm(d_model)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed and encode positions
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # Sequential forward pass through custom blocks
        for layer in self.layers:
            x = layer(x)

        # Final layer normalization and projection
        x = self.ln_final(x)
        logits = self.linear_out(x)
        return F.log_softmax(logits, dim=-1)  # (Batch, Seq_len, Vocab_size)

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive text generation loop."""

        self.eval()
        generated = start_tokens

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated
