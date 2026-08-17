"""Shared feature encoders."""
from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim), nn.LeakyReLU(0.1),
    )


class ActionEncoder(nn.Module):
    def __init__(self, num_actions: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, hidden_dim)
        self.ratio_encoder = mlp(2, hidden_dim, hidden_dim, dropout)
        self.fusion = mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, action_id: torch.Tensor, width_ratio: torch.Tensor, spacing_ratio: torch.Tensor) -> torch.Tensor:
        ratios = torch.stack((width_ratio, spacing_ratio), dim=-1)
        return self.fusion(torch.cat((self.embedding(action_id), self.ratio_encoder(ratios)), dim=-1))

