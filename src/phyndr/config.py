"""Typed configuration and YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class FeatureConfig:
    supply_dim: int = 8
    layer_state_dim: int = 2
    partition_state_dim: int = 12
    partition_demand_dim: int = 6
    critical_net_dim: int = 8
    num_layer_ids: int = 16
    track_index: int = 0
    blockage_index: int = 1
    default_width_index: int = 2
    default_spacing_index: int = 3
    physical_edge_dim: int = 4
    cross_layer_edge_dim: int = 4
    boundary_edge_dim: int = 6
    incidence_edge_dim: int = 3


@dataclass
class PhysicsConfig:
    epsilon: float = 1.0e-6
    enforce_nonnegative_capacity: bool = True
    baseline_action_id: int = 0


@dataclass
class TrainingConfig:
    loss: str = "huber"
    weighting: str = "uncertainty"
    lambda_congestion: float = 1.0
    lambda_drc: float = 1.0
    lambda_wns: float = 1.0
    lambda_tns: float = 1.0

    def __post_init__(self) -> None:
        if self.loss not in {"huber", "mse"}:
            raise ValueError("loss must be 'huber' or 'mse'")
        if self.weighting not in {"uncertainty", "fixed"}:
            raise ValueError("weighting must be 'uncertainty' or 'fixed'")
        if min(self.lambda_congestion, self.lambda_drc, self.lambda_wns, self.lambda_tns) <= 0:
            raise ValueError("all task priors must be positive")


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    num_hetero_blocks: int = 3
    dropout: float = 0.0
    num_layers: int = 3
    num_actions: int = 4
    runtime_validate_graph: bool = False
    features: FeatureConfig = field(default_factory=FeatureConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.num_hetero_blocks != 3 or self.num_layers != 3:
            raise ValueError("PhyNDR v0.3.3 requires exactly three complete HeteroBlocks")
        if self.num_actions != 4:
            raise ValueError("PhyNDR v0.3.3 requires exactly four NDR actions")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


def _construct(cls: type, values: Mapping[str, Any] | None):
    values = dict(values or {})
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ModelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model = dict(raw.get("model", {}))
    model["features"] = _construct(FeatureConfig, raw.get("features"))
    model["physics"] = _construct(PhysicsConfig, raw.get("physics"))
    model["training"] = _construct(TrainingConfig, raw.get("training"))
    return _construct(ModelConfig, model)
