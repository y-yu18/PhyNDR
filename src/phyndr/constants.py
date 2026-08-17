"""Canonical node, relation, action, target, and direction definitions."""
from __future__ import annotations

R_NODE = "r"
U_NODE = "u"
N_NODE = "n_critical"

PHYSICAL_H = (R_NODE, "physical_h", R_NODE)
PHYSICAL_V = (R_NODE, "physical_v", R_NODE)
CROSS_LAYER = (R_NODE, "cross_layer", R_NODE)
BELONGS_TO = (R_NODE, "belongs_to", U_NODE)
CONTAINS = (U_NODE, "contains", R_NODE)
BOUNDARY_H = (U_NODE, "boundary_h", U_NODE)
BOUNDARY_V = (U_NODE, "boundary_v", U_NODE)
INCIDENT_TO = (U_NODE, "incident_to", N_NODE)
INCIDENT_FROM = (N_NODE, "incident_from", U_NODE)

CANONICAL_ETYPES = (
    PHYSICAL_H, PHYSICAL_V, CROSS_LAYER, BELONGS_TO, CONTAINS,
    BOUNDARY_H, BOUNDARY_V, INCIDENT_TO, INCIDENT_FROM,
)

ACTION_NAMES = ("1W1S", "1W2S", "1W3S", "2W3S")
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}
ACTION_WIDTH_RATIO = (1.0, 1.0, 1.0, 2.0)
ACTION_SPACING_RATIO = (1.0, 2.0, 3.0, 3.0)

H_DIRECTION = 0
V_DIRECTION = 1
PARTITION_TARGETS = ("utilization_h_mean", "utilization_v_mean")
CHIP_TARGETS = ("delta_drc", "delta_wns", "delta_tns")

