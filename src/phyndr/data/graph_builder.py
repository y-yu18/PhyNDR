"""DGL graph construction and strict PhyNDR input-contract validation."""
from __future__ import annotations

import torch

from phyndr._dgl import dgl
from phyndr.config import ModelConfig
from phyndr.constants import (
    ACTION_SPACING_RATIO, ACTION_WIDTH_RATIO, BELONGS_TO, CANONICAL_ETYPES,
    N_NODE, R_NODE, U_NODE,
)


def build_graph(num_nodes: dict[str, int], edges: dict, node_features: dict, edge_features: dict | None = None):
    missing = set(CANONICAL_ETYPES) - set(edges)
    if missing:
        raise ValueError(f"missing canonical relations: {sorted(missing)}")
    graph = dgl.heterograph({e: edges[e] for e in CANONICAL_ETYPES}, num_nodes_dict=num_nodes)
    for ntype, values in node_features.items():
        for name, value in values.items():
            graph.nodes[ntype].data[name] = value
    for etype, values in (edge_features or {}).items():
        for name, value in values.items():
            graph.edges[etype].data[name] = value
    return graph


def _require(data, names, where):
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{where} missing fields: {missing}")


def validate_graph(graph, config: ModelConfig | None = None) -> None:
    cfg = config or ModelConfig()
    if set(graph.canonical_etypes) != set(CANONICAL_ETYPES):
        raise ValueError("graph canonical relations do not match the PhyNDR schema")
    r, u, n = graph.nodes[R_NODE].data, graph.nodes[U_NODE].data, graph.nodes[N_NODE].data
    _require(r, ("x_supply", "x_state", "action_id", "width_ratio", "spacing_ratio", "direction", "track_number", "blockage_rate"), "r")
    _require(u, ("x_state", "x_demand", "demand_h", "demand_v", "baseline_utilization"), "u")
    _require(n, ("x",), "n_critical")
    if "ndr_enable" in r:
        raise ValueError("ndr_enable is forbidden: every valid R node selects exactly one action")
    action = r["action_id"].long()
    if action.ndim != 1 or torch.any((action < 0) | (action >= 4)):
        raise ValueError("action_id must be a length-N_R tensor with values in [0,3]")
    expected_w = torch.tensor(ACTION_WIDTH_RATIO, device=action.device)[action]
    expected_s = torch.tensor(ACTION_SPACING_RATIO, device=action.device)[action]
    if not torch.allclose(r["width_ratio"].float(), expected_w) or not torch.allclose(r["spacing_ratio"].float(), expected_s):
        raise ValueError("width_ratio/spacing_ratio do not match action_id")
    if torch.any((r["direction"] != 0) & (r["direction"] != 1)):
        raise ValueError("direction must be 0 (H) or 1 (V)")
    src, _ = graph.edges(etype=BELONGS_TO)
    counts = torch.bincount(src, minlength=graph.num_nodes(R_NODE))
    if not torch.all(counts == 1):
        raise ValueError("every R node must have exactly one belongs_to edge")
    f = cfg.features
    dims = {R_NODE: {"x_supply": f.supply_dim, "x_state": f.layer_state_dim}, U_NODE: {"x_state": f.partition_state_dim, "x_demand": f.partition_demand_dim}, N_NODE: {"x": f.critical_net_dim}}
    for ntype, specs in dims.items():
        for name, dim in specs.items():
            if graph.nodes[ntype].data[name].shape != (graph.num_nodes(ntype), dim):
                raise ValueError(f"{ntype}.{name} must have shape [{graph.num_nodes(ntype)}, {dim}]")
    if u["baseline_utilization"].shape != (graph.num_nodes(U_NODE), 2):
        raise ValueError("u.baseline_utilization must have shape [N_P, 2]")

