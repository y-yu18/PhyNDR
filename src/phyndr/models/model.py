"""End-to-end PhyNDR v0.3.3 model."""
from __future__ import annotations

import torch
from torch import nn

from phyndr.config import ModelConfig
from phyndr.constants import (
    BELONGS_TO, BOUNDARY_H, BOUNDARY_V, CANONICAL_ETYPES, CHIP_TARGETS,
    CONTAINS, CROSS_LAYER, H_DIRECTION, INCIDENT_FROM, INCIDENT_TO, N_NODE,
    PARTITION_TARGETS, PHYSICAL_H, PHYSICAL_V, R_NODE, U_NODE, V_DIRECTION,
)
from phyndr.data.graph_builder import validate_graph
from phyndr.models.resource_action import ResourceActionInteraction
from phyndr.models.encoders import ActionEncoder, mlp
from phyndr.models.heads import ChipHeads, PartitionReadout, RegressionHead
from phyndr.models.hetero_block import HeteroBlock
from phyndr.models.physics_bridge import PhysicsBridge
from phyndr.models.pooling import GlobalPartitionPooling, SegmentAttentionPooling


def _batch_ids(graph, ntype: str, device) -> tuple[torch.Tensor, int]:
    try:
        counts = graph.batch_num_nodes(ntype).to(device)
    except Exception:
        counts = torch.tensor([graph.num_nodes(ntype)], device=device)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts), int(counts.numel())


class PhyNDR(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg, feat, dim = self.config, self.config.features, self.config.hidden_dim
        self.supply_encoder = mlp(feat.supply_dim, dim, dim, cfg.dropout)
        self.action_encoder = ActionEncoder(cfg.num_actions, dim, cfg.dropout)
        self.layer_state_encoder = mlp(feat.layer_state_dim, dim, dim, cfg.dropout)
        self.partition_state_encoder = mlp(feat.partition_state_dim, dim, dim, cfg.dropout)
        self.demand_encoder = mlp(feat.partition_demand_dim, dim, dim, cfg.dropout)
        self.net_encoder = mlp(feat.critical_net_dim, dim, dim, cfg.dropout)
        self.resource_action = ResourceActionInteraction(dim, cfg.physics.baseline_action_id)
        self.physics = PhysicsBridge(dim, cfg.physics.epsilon)
        self.r_project = mlp(5 * dim, dim, dim, cfg.dropout)
        self.u_project = mlp(2 * dim, dim, dim, cfg.dropout)
        edge_dims = {
            PHYSICAL_H: feat.physical_edge_dim, PHYSICAL_V: feat.physical_edge_dim,
            CROSS_LAYER: feat.cross_layer_edge_dim, BOUNDARY_H: feat.boundary_edge_dim,
            BOUNDARY_V: feat.boundary_edge_dim, INCIDENT_TO: feat.incidence_edge_dim,
            INCIDENT_FROM: feat.incidence_edge_dim, BELONGS_TO: 0, CONTAINS: 0,
        }
        self.blocks = nn.ModuleList([HeteroBlock(dim, edge_dims, cfg.dropout) for _ in range(3)])
        self.directional_pool = SegmentAttentionPooling(dim)
        self.partition_readout = PartitionReadout(dim, cfg.dropout)
        self.partition_head = RegressionHead(dim, len(PARTITION_TARGETS), cfg.dropout)
        self.partition_global_pool = GlobalPartitionPooling(dim)
        self.critical_net_pooling = GlobalPartitionPooling(dim)
        self.chip_heads = ChipHeads(dim, cfg.dropout)

    def forward(self, graph):
        if self.config.runtime_validate_graph:
            validate_graph(graph, self.config)
        r, u, n = graph.nodes[R_NODE].data, graph.nodes[U_NODE].data, graph.nodes[N_NODE].data
        f = self.config.features
        action_id = r["action_id"].long()
        width_ratio, spacing_ratio = r["width_ratio"].float(), r["spacing_ratio"].float()
        h_supply = self.supply_encoder(r["x_supply"].float())
        h_action = self.action_encoder(action_id, width_ratio, spacing_ratio)
        h_state_r = self.layer_state_encoder(r["x_state"].float())
        track = r["track_number"].float().unsqueeze(-1)
        blockage = r["blockage_rate"].float().clamp(0, 1).unsqueeze(-1)
        c0 = track * (1.0 - blockage)
        resource = self.resource_action(c0, h_supply, h_action, action_id, width_ratio, spacing_ratio)
        r_src, r_to_u = graph.edges(etype=BELONGS_TO)
        owner = torch.empty(graph.num_nodes(R_NODE), dtype=torch.long, device=r_src.device)
        owner[r_src] = r_to_u
        direction = r["direction"].long()
        physics = self.physics(resource.effective_supply, owner, direction, u["demand_h"].float(), u["demand_v"].float(), graph.num_nodes(U_NODE))
        h_r = self.r_project(torch.cat((h_supply, h_action, h_state_r, resource.embedding, physics.r_context), dim=-1))
        h_u = self.u_project(torch.cat((self.partition_state_encoder(u["x_state"].float()), self.demand_encoder(u["x_demand"].float())), dim=-1))
        h_n = self.net_encoder(n["x"].float()) if graph.num_nodes(N_NODE) else h_r.new_zeros((0, self.config.hidden_dim))
        states = {R_NODE: h_r, U_NODE: h_u, N_NODE: h_n}
        block_states = []
        for block in self.blocks:
            states = block(graph, states)
            block_states.append(states)
        h_r, h_u, h_n = states[R_NODE], states[U_NODE], states[N_NODE]
        z_h, att_h = self.directional_pool(h_r, owner, graph.num_nodes(U_NODE), direction == H_DIRECTION)
        z_v, att_v = self.directional_pool(h_r, owner, graph.num_nodes(U_NODE), direction == V_DIRECTION)
        z_p = self.partition_readout(h_u, z_h, z_v)
        utilization_residual = self.partition_head(z_p)
        baseline_utilization = u["baseline_utilization"].float()
        partition_utilization = baseline_utilization + utilization_residual
        u_batch, batch_size = _batch_ids(graph, U_NODE, z_p.device)
        z_chip, chip_attention = self.partition_global_pool(z_p, u_batch, batch_size)
        n_batch, _ = _batch_ids(graph, N_NODE, z_p.device)
        if h_n.shape[0]:
            z_net, net_attention = self.critical_net_pooling(h_n, n_batch, batch_size)
        else:
            z_net, net_attention = z_p.new_zeros((batch_size, z_p.shape[-1])), h_n.new_zeros((0, 1))
        delta_chip = self.chip_heads(z_chip, z_net)
        assert delta_chip.shape[-1] == len(CHIP_TARGETS)
        return {
            "partition_utilization": partition_utilization, "utilization_residual": utilization_residual,
            "delta_chip": delta_chip,
            "z_partition": z_p, "z_critical_net": z_net, "block_states": block_states,
            "attention_h": att_h, "attention_v": att_v, "chip_attention": chip_attention,
            "critical_net_attention": net_attention, "resource_action": resource, "physics": physics,
        }
