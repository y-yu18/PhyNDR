"""Adapter from ``phyndr_ndr_720_v1`` archives to PhyNDR DGL graphs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phyndr._dgl import dgl
from phyndr.constants import (
    ACTION_SPACING_RATIO,
    ACTION_WIDTH_RATIO,
    BELONGS_TO,
    BOUNDARY_H,
    BOUNDARY_V,
    CONTAINS,
    CROSS_LAYER,
    INCIDENT_FROM,
    INCIDENT_TO,
    N_NODE,
    PHYSICAL_H,
    PHYSICAL_V,
    R_NODE,
    U_NODE,
)
from phyndr.data.graph_builder import build_graph


def _signed_log_standardize(values: np.ndarray) -> np.ndarray:
    """Stable node-wise scaling for the single-baseline training setting."""
    values = np.asarray(values, dtype=np.float32)
    transformed = np.sign(values) * np.log1p(np.abs(values))
    mean = transformed.mean(axis=0, keepdims=True)
    std = transformed.std(axis=0, keepdims=True)
    return ((transformed - mean) / np.where(std > 1.0e-6, std, 1.0)).astype(np.float32)


@dataclass(frozen=True)
class NDR720Paths:
    dataset_root: Path
    baseline_root: Path


class NDR720Dataset(Dataset):
    """One fixed physical design paired with many candidate NDR assignments."""

    def __init__(self, dataset_root: str | Path, baseline_root: str | Path, split: str):
        self.paths = NDR720Paths(Path(dataset_root), Path(baseline_root))
        with np.load(self.paths.dataset_root / "samples.npz", allow_pickle=False) as archive:
            stored_split = archive["split"].astype(str)
            self.sample_id = archive["sample_id"].astype(str)
            self.action_id = archive["action_id"].astype(np.int64)
            self.targets = archive["y_partition_utilization"].astype(np.float32)
        if split not in {"train", "validation", "test", "all"}:
            raise ValueError("split must be train, validation, test, or all")
        self.indices = np.arange(len(stored_split)) if split == "all" else np.flatnonzero(stored_split == split)
        with np.load(self.paths.dataset_root / "static_inputs.npz", allow_pickle=False) as archive:
            static = {key: archive[key] for key in archive.files}
        with np.load(self.paths.dataset_root / "synthetic_timing_inputs.npz", allow_pickle=False) as archive:
            timing = {key: archive[key] for key in archive.files}

        # Partition state is [real baseline physical state, synthetic power proxy].
        self.x_supply = _signed_log_standardize(static["x_supply"])
        self.track_number = (static["x_supply"][:, 0] / 1000.0).astype(np.float32)
        self.blockage_rate = static["x_supply"][:, 1].astype(np.float32)
        self.x_state_r = _signed_log_standardize(static["x_state_r"])
        self.x_state_u = _signed_log_standardize(
            np.concatenate((static["x_state_u"], static["power_proxy_u"]), axis=1)
        )
        self.x_demand_u = _signed_log_standardize(static["x_demand_u"])
        self.demand_h = (np.maximum(static["demand_h"], 0) / 1000.0).astype(np.float32)
        self.demand_v = (np.maximum(static["demand_v"], 0) / 1000.0).astype(np.float32)
        self.direction = static["direction_r"].astype(np.int64)
        self.partition_index_r = static["partition_index_r"].astype(np.int64)
        self.baseline_partition = static["x_state_u"][:, 6:8].astype(np.float32)
        self.x_critical = _signed_log_standardize(timing["x_critical"])
        self._topology, self._edge_features = self._build_topology(timing)

    def _build_topology(self, timing: dict[str, np.ndarray]):
        adjacency = pd.read_csv(self.paths.baseline_root / "partition_adjacency.csv")
        edge_lists: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
        features: dict[tuple[str, str, str], list[list[float]]] = {}
        for etype in (PHYSICAL_H, PHYSICAL_V, CROSS_LAYER, BELONGS_TO, CONTAINS, BOUNDARY_H, BOUNDARY_V, INCIDENT_TO, INCIDENT_FROM):
            edge_lists[etype] = ([], [])
            features[etype] = []

        for row in adjacency.itertuples(index=False):
            src, dst = int(row.src_partition_index), int(row.dst_partition_index)
            direction = str(row.adjacency_direction).upper()
            r_etype = PHYSICAL_H if direction == "H" else PHYSICAL_V
            u_etype = BOUNDARY_H if direction == "H" else BOUNDARY_V
            boundary = [
                float(row.shared_boundary_length), float(row.relative_dx), float(row.relative_dy),
                float(direction == "H"), float(direction == "V"),
                float(np.hypot(row.relative_dx, row.relative_dy)),
            ]
            edge_lists[u_etype][0].append(src)
            edge_lists[u_etype][1].append(dst)
            features[u_etype].append(boundary)
            for layer in range(6):
                edge_lists[r_etype][0].append(src * 6 + layer)
                edge_lists[r_etype][1].append(dst * 6 + layer)
                features[r_etype].append(
                    [float(row.shared_boundary_length), float(row.relative_dx), float(row.relative_dy), layer / 5.0]
                )

        for partition in range(10):
            for layer in range(5):
                a, b = partition * 6 + layer, partition * 6 + layer + 1
                for src, dst, sign in ((a, b, 1.0), (b, a, -1.0)):
                    edge_lists[CROSS_LAYER][0].append(src)
                    edge_lists[CROSS_LAYER][1].append(dst)
                    features[CROSS_LAYER].append([sign, 1.0, float(self.direction[src]), float(self.direction[dst])])
        for r_index, owner in enumerate(self.partition_index_r):
            edge_lists[BELONGS_TO][0].append(r_index)
            edge_lists[BELONGS_TO][1].append(int(owner))
            edge_lists[CONTAINS][0].append(int(owner))
            edge_lists[CONTAINS][1].append(r_index)

        incidence_x = timing["incidence_x"].astype(np.float32)
        for i, (partition, net) in enumerate(
            zip(timing["incidence_partition_index"], timing["incidence_critical_index"])
        ):
            partition, net = int(partition), int(net)
            edge_lists[INCIDENT_TO][0].append(partition)
            edge_lists[INCIDENT_TO][1].append(net)
            features[INCIDENT_TO].append(incidence_x[i].tolist())
            edge_lists[INCIDENT_FROM][0].append(net)
            edge_lists[INCIDENT_FROM][1].append(partition)
            features[INCIDENT_FROM].append(incidence_x[i].tolist())

        edges = {
            etype: (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64))
            for etype, (src, dst) in edge_lists.items()
        }
        edge_features = {}
        dims = {PHYSICAL_H: 4, PHYSICAL_V: 4, CROSS_LAYER: 4, BOUNDARY_H: 6, BOUNDARY_V: 6, INCIDENT_TO: 3, INCIDENT_FROM: 3}
        for etype, dim in dims.items():
            array = np.asarray(features[etype], dtype=np.float32).reshape(-1, dim)
            edge_features[etype] = torch.from_numpy(_signed_log_standardize(array))
        return edges, edge_features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample_index = int(self.indices[item])
        action = self.action_id[sample_index]
        width = np.asarray(ACTION_WIDTH_RATIO, dtype=np.float32)[action]
        spacing = np.asarray(ACTION_SPACING_RATIO, dtype=np.float32)[action]
        node_features = {
            R_NODE: {
                "x_supply": torch.from_numpy(self.x_supply),
                "x_state": torch.from_numpy(self.x_state_r),
                "action_id": torch.from_numpy(action),
                "width_ratio": torch.from_numpy(width),
                "spacing_ratio": torch.from_numpy(spacing),
                "track_number": torch.from_numpy(self.track_number),
                "blockage_rate": torch.from_numpy(self.blockage_rate),
                "direction": torch.from_numpy(self.direction),
            },
            U_NODE: {
                "x_state": torch.from_numpy(self.x_state_u),
                "baseline_utilization": torch.from_numpy(self.baseline_partition),
                "x_demand": torch.from_numpy(self.x_demand_u),
                "demand_h": torch.from_numpy(self.demand_h),
                "demand_v": torch.from_numpy(self.demand_v),
            },
            N_NODE: {"x": torch.from_numpy(self.x_critical)},
        }
        edge_features = {etype: {"x": value} for etype, value in self._edge_features.items()}
        graph = build_graph(
            {R_NODE: 60, U_NODE: 10, N_NODE: len(self.x_critical)},
            self._topology,
            node_features,
            edge_features,
        )
        label = torch.from_numpy(self.targets[sample_index])
        baseline = torch.from_numpy(self.baseline_partition)
        return graph, label, baseline, sample_index


def collate_ndr720(batch):
    graphs, labels, baselines, indices = zip(*batch)
    return dgl.batch(graphs), torch.cat(labels, dim=0), torch.cat(baselines, dim=0), torch.tensor(indices)


__all__ = ["NDR720Dataset", "collate_ndr720"]
