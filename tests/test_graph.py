import pytest
import torch

from phyndr.constants import BELONGS_TO
from phyndr.data.graph_builder import validate_graph
from phyndr.data.synthetic import make_synthetic_sample


def test_synthetic_schema_and_labels():
    graph, labels = make_synthetic_sample()
    validate_graph(graph)
    assert labels["y_partition_utilization"].shape == (graph.num_nodes("u"), 2)
    assert labels["y_chip"].shape == (1, 3)


def test_every_r_has_one_owner():
    graph, _ = make_synthetic_sample()
    src, _ = graph.edges(etype=BELONGS_TO)
    assert torch.all(torch.bincount(src, minlength=graph.num_nodes("r")) == 1)


def test_action_ratio_mismatch_rejected():
    graph, _ = make_synthetic_sample()
    graph.nodes["r"].data["width_ratio"][0] = 9.0
    with pytest.raises(ValueError, match="action_id"):
        validate_graph(graph)

