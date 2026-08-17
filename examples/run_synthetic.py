from __future__ import annotations

import torch

from phyndr.config import ModelConfig
from phyndr.data.synthetic import make_synthetic_sample
from phyndr.losses import PhyNDRLoss
from phyndr.models import PhyNDR


def main():
    torch.manual_seed(7)
    graph, labels = make_synthetic_sample()
    model = PhyNDR(ModelConfig(hidden_dim=16))
    output = model(graph)
    loss = PhyNDRLoss()(output, labels)
    loss["total"].backward()
    print("partition_utilization:", list(output["partition_utilization"].shape))
    print("delta_chip:", list(output["delta_chip"].shape))
    print("loss:", float(loss["total"].detach()))


if __name__ == "__main__":
    main()

