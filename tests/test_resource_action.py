import torch

from phyndr.models.resource_action import ResourceActionInteraction


def test_action_does_not_reduce_physical_supply():
    module = ResourceActionInteraction(4)
    supply = torch.tensor([[10.0], [10.0]])
    h = torch.zeros(2, 4)
    result = module(
        supply, h, h, torch.tensor([0, 3]),
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]),
    )
    assert torch.equal(result.effective_supply, supply)
    assert result.action_pressure[0].item() == 0.0
    assert result.action_pressure[1].item() > 0.0


def test_resource_action_backward():
    module = ResourceActionInteraction(4)
    h1 = torch.randn(2, 4, requires_grad=True)
    h2 = torch.randn(2, 4, requires_grad=True)
    result = module(
        torch.ones(2, 1), h1, h2, torch.tensor([1, 2]),
        torch.ones(2), torch.tensor([2.0, 3.0]),
    )
    result.embedding.sum().backward()
    assert torch.isfinite(h1.grad).all()
