import torch

from phyndr._dgl import dgl
from phyndr.config import ModelConfig
from phyndr.data.synthetic import make_synthetic_sample
from phyndr.losses import PhyNDRLoss
from phyndr.models import PhyNDR


def test_forward_and_backward():
    graph, labels = make_synthetic_sample()
    model = PhyNDR(ModelConfig(hidden_dim=16))
    output = model(graph)
    assert output["partition_utilization"].shape == (graph.num_nodes("u"), 2)
    assert output["delta_chip"].shape == (1, 3)
    PhyNDRLoss()(output, labels)["total"].backward()
    grad = model.partition_head.net[-1].weight.grad
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[0]) and torch.count_nonzero(grad[1])


def test_batch_is_chip_safe():
    ga, _ = make_synthetic_sample(seed=11)
    gb, _ = make_synthetic_sample(seed=13)
    output = PhyNDR(ModelConfig(hidden_dim=16))(dgl.batch([ga, gb]))
    assert output["partition_utilization"].shape == (8, 2)
    assert output["delta_chip"].shape == (2, 3)


def test_without_critical_nets():
    graph, _ = make_synthetic_sample(include_critical_net=False)
    output = PhyNDR(ModelConfig(hidden_dim=16))(graph)
    assert output["partition_utilization"].shape == (4, 2)
    assert output["delta_chip"].shape == (1, 3)
