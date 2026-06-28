"""LSTM decoder with two init projections (h0, c0) from the image vector.

See PLAN.md § Decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Single-layer LSTM, hidden=512, embedding=512, dropout=0.5.

    h0 = tanh(W_h @ features), c0 = tanh(W_c @ features).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        embedding_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(2048, hidden_dim)
        self.init_c = nn.Linear(2048, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        caption_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """features: (B, 2048); caption_idxs: (B, T) — first token is <start>.
        Returns logits (B, T, vocab_size).
        """
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1, B, H)
        c0 = torch.tanh(self.init_c(features)).unsqueeze(0)
        emb = self.embed(caption_idxs)  # (B, T, E)
        out, _ = self.lstm(emb, (h0, c0))  # (B, T, H)
        return self.head(out)  # (B, T, V)
