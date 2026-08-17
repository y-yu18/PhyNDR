"""Directional partition supply-demand diagnostics broadcast back to R nodes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from phyndr.constants import H_DIRECTION, V_DIRECTION
from phyndr.models.encoders import mlp


def _sum_by(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((size, values.shape[-1]))
    if values.numel():
        out.index_add_(0, index, values)
    return out


@dataclass
class PhysicsResult:
    r_context: torch.Tensor
    capacity_h: torch.Tensor
    capacity_v: torch.Tensor
    rho_h: torch.Tensor
    rho_v: torch.Tensor
    margin_h: torch.Tensor
    margin_v: torch.Tensor
    overflow_h: torch.Tensor
    overflow_v: torch.Tensor


class PhysicsBridge(nn.Module):
    def __init__(self, hidden_dim: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.epsilon = epsilon
        self.context = mlp(4, hidden_dim, hidden_dim)

    def forward(self, c_eff, r_to_u, direction, demand_h, demand_v, num_u):
        hmask = (direction == H_DIRECTION).to(c_eff.dtype).unsqueeze(-1)
        vmask = (direction == V_DIRECTION).to(c_eff.dtype).unsqueeze(-1)
        cap_h = _sum_by(c_eff * hmask, r_to_u, num_u)
        cap_v = _sum_by(c_eff * vmask, r_to_u, num_u)
        demand_h = demand_h.reshape(num_u, 1)
        demand_v = demand_v.reshape(num_u, 1)
        rho_h, rho_v = demand_h / (cap_h + self.epsilon), demand_v / (cap_v + self.epsilon)
        margin_h, margin_v = cap_h - demand_h, cap_v - demand_v
        overflow_h, overflow_v = torch.relu(-margin_h), torch.relu(-margin_v)
        ctx_h = self.context(torch.cat((rho_h, margin_h, overflow_h, demand_h), dim=-1))
        ctx_v = self.context(torch.cat((rho_v, margin_v, overflow_v, demand_v), dim=-1))
        r_context = ctx_h[r_to_u] * hmask + ctx_v[r_to_u] * vmask
        return PhysicsResult(r_context, cap_h, cap_v, rho_h, rho_v, margin_h, margin_v, overflow_h, overflow_v)

