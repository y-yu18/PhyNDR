"""Partition and chip regression readouts."""
from __future__ import annotations

import torch
from torch import nn


class PartitionReadout(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1))

    def forward(self, u, z_h, z_v):
        return self.net(torch.cat((u, z_h, z_v), dim=-1))


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class ScalarHead(RegressionHead):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__(hidden_dim, 1, dropout)


class ChipHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.drc = ScalarHead(2 * hidden_dim, dropout)
        self.wns = ScalarHead(2 * hidden_dim, dropout)
        self.tns = ScalarHead(2 * hidden_dim, dropout)

    def forward(self, z_chip, z_critical):
        z = torch.cat((z_chip, z_critical), dim=-1)
        return torch.cat((self.drc(z), self.wns(z), self.tns(z)), dim=-1)
