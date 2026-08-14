import torch


def test_torch_imports_and_computes():
    x = torch.ones(2, 3)
    assert x.sum().item() == 6.0
