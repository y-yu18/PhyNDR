import pytest
import torch

from phyndr.losses import PhyNDRLoss


def test_uncertainty_loss_backward():
    prediction = {"partition_utilization": torch.randn(2, 2, requires_grad=True), "delta_chip": torch.randn(1, 3, requires_grad=True)}
    target = {"y_partition_utilization": torch.zeros(2, 2), "y_chip": torch.zeros(1, 3)}
    loss_module = PhyNDRLoss()
    result = loss_module(prediction, target)
    result["total"].backward()
    assert loss_module.log_variances.grad.shape == (4,)


def test_fixed_loss():
    p = {"partition_utilization": torch.zeros(2, 2), "delta_chip": torch.zeros(1, 3)}
    y = {"y_partition_utilization": torch.ones(2, 2), "y_chip": torch.ones(1, 3)}
    assert PhyNDRLoss(weighting="fixed")(p, y)["total"] > 0


def test_scalar_partition_labels_rejected():
    p = {"partition_utilization": torch.zeros(2, 2), "delta_chip": torch.zeros(1, 3)}
    y = {"y_partition_utilization": torch.zeros(2, 1), "y_chip": torch.zeros(1, 3)}
    with pytest.raises(ValueError, match=r"\[N_P, 2\]"):
        PhyNDRLoss(weighting="fixed")(p, y)

