"""Supply/action interaction without a capacity-loss assumption."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from phyndr.models.encoders import mlp


@dataclass
class ResourceActionResult:
    base_supply: torch.Tensor
    effective_supply: torch.Tensor
    action_pressure: torch.Tensor
    embedding: torch.Tensor


class ResourceActionInteraction(nn.Module):
    """Encode NDR pressure while leaving physical routing supply unchanged.

    Width/spacing actions are categorical design choices.  They are not
    converted into an invented capacity-loss target.  A gated learned scalar
    provides an action-pressure diagnostic and an embedding for message
    passing; the PhysicsBridge always receives the baseline physical supply.
    """

    def __init__(self, hidden_dim: int, baseline_action_id: int = 0):
        super().__init__()
        self.baseline_action_id = baseline_action_id
        self.pressure = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.embedding = mlp(4, hidden_dim, hidden_dim)

    def forward(self, base_supply, h_supply, h_action, action_id, width_ratio, spacing_ratio):
        gate = (action_id != self.baseline_action_id).to(base_supply.dtype).unsqueeze(-1)
        action_pressure = gate * self.pressure(torch.cat((h_supply, h_action), dim=-1))
        effective_supply = base_supply
        embedding = self.embedding(
            torch.cat(
                (
                    base_supply,
                    action_pressure,
                    width_ratio.unsqueeze(-1),
                    spacing_ratio.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        return ResourceActionResult(base_supply, effective_supply, action_pressure, embedding)


__all__ = ["ResourceActionInteraction", "ResourceActionResult"]
