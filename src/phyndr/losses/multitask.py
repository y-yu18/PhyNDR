"""Four-task regression loss with fixed or learned uncertainty weighting."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from phyndr.config import TrainingConfig
from phyndr.constants import PARTITION_TARGETS


class PhyNDRLoss(nn.Module):
    def __init__(self, loss: str = "huber", weighting: str = "uncertainty", priors=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        if loss not in {"huber", "mse"} or weighting not in {"fixed", "uncertainty"}:
            raise ValueError("unsupported loss or weighting")
        self.loss_name, self.weighting = loss, weighting
        self.register_buffer("priors", torch.tensor(priors, dtype=torch.float32))
        if self.priors.shape != (4,) or torch.any(self.priors <= 0):
            raise ValueError("priors must contain four positive values")
        self.log_variances = nn.Parameter(torch.zeros(4)) if weighting == "uncertainty" else None

    @classmethod
    def from_config(cls, training: TrainingConfig):
        return cls(training.loss, training.weighting, (training.lambda_congestion, training.lambda_drc, training.lambda_wns, training.lambda_tns))

    def _loss(self, prediction, target):
        return F.smooth_l1_loss(prediction, target) if self.loss_name == "huber" else F.mse_loss(prediction, target)

    def forward(self, predictions, targets):
        p_part, p_chip = predictions["partition_utilization"], predictions["delta_chip"]
        y_part, y_chip = targets["y_partition_utilization"], targets["y_chip"]
        expected = len(PARTITION_TARGETS)
        if p_part.ndim != 2 or p_part.shape[1] != expected:
            raise ValueError("partition_utilization must have shape [N_P, 2] in fixed H/V order")
        if y_part.shape != p_part.shape:
            raise ValueError("y_partition_utilization must exactly match partition_utilization shape [N_P, 2]")
        if p_chip.ndim != 2 or p_chip.shape[1] != 3 or y_chip.shape != p_chip.shape:
            raise ValueError("delta_chip and y_chip must have matching shape [B, 3]")
        losses = torch.stack((self._loss(p_part, y_part), self._loss(p_chip[:, 0], y_chip[:, 0]), self._loss(p_chip[:, 1], y_chip[:, 1]), self._loss(p_chip[:, 2], y_chip[:, 2])))
        priors = self.priors.to(losses)
        if self.weighting == "uncertainty":
            precision = torch.exp(-self.log_variances)
            effective = priors * precision
            total = torch.sum(priors * (precision * losses + self.log_variances))
        else:
            effective, total = priors, torch.sum(priors * losses)
        return {"total": total, "congestion": losses[0], "drc": losses[1], "wns": losses[2], "tns": losses[3], "effective_weights": effective}
