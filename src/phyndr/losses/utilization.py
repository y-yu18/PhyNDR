"""Primary loss for absolute partition H/V utilization."""
from __future__ import annotations

from torch import nn
from torch.nn import functional as F


class UtilizationLoss(nn.Module):
    """Fit absolute utilization and the smaller action-induced response."""

    def __init__(self, loss: str = "huber", response_weight: float = 1.0, huber_beta: float = 0.02):
        super().__init__()
        if loss not in {"huber", "mse"}:
            raise ValueError("loss must be huber or mse")
        if response_weight < 0 or huber_beta <= 0:
            raise ValueError("response_weight must be nonnegative and huber_beta positive")
        self.loss_name = loss
        self.response_weight = float(response_weight)
        self.huber_beta = float(huber_beta)

    def _loss(self, prediction, target, reduction="mean"):
        if self.loss_name == "mse":
            return F.mse_loss(prediction, target, reduction=reduction)
        return F.smooth_l1_loss(prediction, target, beta=self.huber_beta, reduction=reduction)

    def forward(self, predictions, target, baseline):
        prediction = predictions["partition_utilization"]
        if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 2:
            raise ValueError("partition utilization prediction and target must match [N_P, 2]")
        if baseline.shape != target.shape:
            raise ValueError("baseline must match target [N_P, 2]")
        absolute = self._loss(prediction, target)
        elementwise = self._loss(prediction, target, reduction="none")
        importance = (target - baseline).abs()
        importance = importance / importance.mean().clamp_min(1.0e-6)
        response = (elementwise * importance).sum() / importance.sum().clamp_min(1.0)
        return {
            "total": absolute + self.response_weight * response,
            "absolute": absolute,
            "response": response,
        }


__all__ = ["UtilizationLoss"]
