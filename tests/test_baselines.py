import torch

from src.baselines import CNN, MLP


def test_mlp_output_shape():
    assert MLP(num_classes=10)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_cnn_output_shape():
    assert CNN(num_classes=10).eval()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_cnn_has_dropout():
    assert any(isinstance(m, torch.nn.Dropout) for m in CNN().modules())
