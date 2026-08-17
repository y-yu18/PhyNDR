"""Small deterministic graph for smoke tests; not a training-data substitute."""
from __future__ import annotations

import torch

from phyndr.config import ModelConfig
from phyndr.constants import *  # noqa: F403
from phyndr.data.graph_builder import build_graph, validate_graph


def _bi(pairs):
    pairs = list(pairs)
    return (torch.tensor([a for a, b in pairs] + [b for a, b in pairs]), torch.tensor([b for a, b in pairs] + [a for a, b in pairs]))


def make_synthetic_sample(seed: int = 7, include_critical_net: bool = True, config: ModelConfig | None = None):
    cfg = config or ModelConfig()
    g = torch.Generator().manual_seed(seed)
    num_u, num_r, num_n = 4, 8, 2 if include_critical_net else 0
    owner = torch.arange(num_u).repeat_interleave(2)
    direction = torch.tensor([0, 1] * num_u)
    physical_h = _bi([(0, 2), (4, 6)])
    physical_v = _bi([(1, 5), (3, 7)])
    cross = _bi([(0, 1), (2, 3), (4, 5), (6, 7)])
    bh = _bi([(0, 1), (2, 3)])
    bv = _bi([(0, 2), (1, 3)])
    if num_n:
        ui = (torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0, 1, 1]))
        ni = (ui[1], ui[0])
    else:
        ui = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        ni = (ui[1], ui[0])
    edges = {
        PHYSICAL_H: physical_h, PHYSICAL_V: physical_v, CROSS_LAYER: cross,
        BELONGS_TO: (torch.arange(num_r), owner), CONTAINS: (owner, torch.arange(num_r)),
        BOUNDARY_H: bh, BOUNDARY_V: bv, INCIDENT_TO: ui, INCIDENT_FROM: ni,
    }
    action = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3])
    width = torch.tensor(ACTION_WIDTH_RATIO)[action]
    spacing = torch.tensor(ACTION_SPACING_RATIO)[action]
    supply = torch.rand((num_r, cfg.features.supply_dim), generator=g)
    supply[:, 0] = 20.0 + 10.0 * supply[:, 0]
    supply[:, 1] = 0.3 * supply[:, 1]
    supply[:, 2] = 0.04 + 0.02 * supply[:, 2]
    supply[:, 3] = 0.04 + 0.02 * supply[:, 3]
    node_features = {
        R_NODE: {"x_supply": supply, "x_state": torch.randn((num_r, cfg.features.layer_state_dim), generator=g), "action_id": action, "width_ratio": width, "spacing_ratio": spacing, "direction": direction, "track_number": supply[:, 0].clone(), "blockage_rate": supply[:, 1].clone()},
        U_NODE: {"x_state": torch.randn((num_u, cfg.features.partition_state_dim), generator=g), "x_demand": torch.randn((num_u, cfg.features.partition_demand_dim), generator=g), "demand_h": 5.0 + torch.rand(num_u, generator=g), "demand_v": 5.0 + torch.rand(num_u, generator=g), "baseline_utilization": 0.1 + 0.2 * torch.rand((num_u, 2), generator=g)},
        N_NODE: {"x": torch.randn((num_n, cfg.features.critical_net_dim), generator=g)},
    }
    edge_dims = {PHYSICAL_H: cfg.features.physical_edge_dim, PHYSICAL_V: cfg.features.physical_edge_dim, CROSS_LAYER: cfg.features.cross_layer_edge_dim, BOUNDARY_H: cfg.features.boundary_edge_dim, BOUNDARY_V: cfg.features.boundary_edge_dim, INCIDENT_TO: cfg.features.incidence_edge_dim, INCIDENT_FROM: cfg.features.incidence_edge_dim}
    edge_features = {e: {"x": torch.randn((len(edges[e][0]), dim), generator=g)} for e, dim in edge_dims.items()}
    graph = build_graph({R_NODE: num_r, U_NODE: num_u, N_NODE: num_n}, edges, node_features, edge_features)
    validate_graph(graph, cfg)
    labels = {"y_partition_utilization": node_features[U_NODE]["baseline_utilization"] + 0.02 * torch.randn((num_u, 2), generator=g), "y_chip": 0.05 * torch.randn((1, 3), generator=g)}
    return graph, labels

