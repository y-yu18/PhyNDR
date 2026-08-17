"""Segmented attention pooling."""
from __future__ import annotations

import torch
from torch import nn


class SegmentAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, segment: torch.Tensor, num_segments: int, mask: torch.Tensor | None = None):
        out = x.new_zeros((num_segments, x.shape[-1]))
        weights = x.new_zeros((x.shape[0], 1))
        valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        logits = self.score(x).squeeze(-1)
        for i in range(num_segments):
            idx = torch.nonzero(valid & (segment == i), as_tuple=False).flatten()
            if idx.numel():
                alpha = torch.softmax(logits[idx], dim=0).unsqueeze(-1)
                out[i] = torch.sum(alpha * x[idx], dim=0)
                weights[idx] = alpha
        return out, weights


class GlobalPartitionPooling(SegmentAttentionPooling):
    pass

