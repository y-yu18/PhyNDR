"""One complete relation-aware heterogeneous message-passing block."""
from __future__ import annotations

import torch
from torch import nn

from phyndr.constants import CANONICAL_ETYPES, N_NODE, R_NODE, U_NODE


def _key(etype) -> str:
    return "__".join(etype)


class HeteroBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dims: dict, dropout: float = 0.0):
        super().__init__()
        self.relation_source = nn.ModuleDict({_key(e): nn.Linear(hidden_dim, hidden_dim) for e in CANONICAL_ETYPES})
        self.relation_edge = nn.ModuleDict({_key(e): nn.Linear(edge_dims[e], hidden_dim, bias=False) for e in CANONICAL_ETYPES if edge_dims.get(e, 0) > 0})
        self.relation_gate = nn.ParameterDict({_key(e): nn.Parameter(torch.zeros(())) for e in CANONICAL_ETYPES})
        self.update = nn.ModuleDict({n: nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)) for n in (R_NODE, U_NODE, N_NODE)})
        self.norm = nn.ModuleDict({n: nn.LayerNorm(hidden_dim) for n in (R_NODE, U_NODE, N_NODE)})

    def forward(self, graph, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        incoming = {ntype: states[ntype].new_zeros(states[ntype].shape) for ntype in states}
        for etype in CANONICAL_ETYPES:
            src_type, _, dst_type = etype
            src, dst = graph.edges(etype=etype)
            if src.numel() == 0:
                continue
            key = _key(etype)
            msg = self.relation_source[key](states[src_type][src])
            if key in self.relation_edge:
                if "x" not in graph.edges[etype].data:
                    raise ValueError(f"missing edge feature 'x' for {etype}")
                msg = msg + self.relation_edge[key](graph.edges[etype].data["x"])
            aggregate = incoming[dst_type].new_zeros(incoming[dst_type].shape)
            aggregate.index_add_(0, dst, msg)
            degree = torch.bincount(dst, minlength=aggregate.shape[0]).clamp_min(1).to(msg.dtype).unsqueeze(-1)
            incoming[dst_type] = incoming[dst_type] + torch.sigmoid(self.relation_gate[key]) * aggregate / degree
        return {ntype: self.norm[ntype](h + self.update[ntype](torch.cat((h, incoming[ntype]), dim=-1))) for ntype, h in states.items()}

