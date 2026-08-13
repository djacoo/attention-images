import torch

from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES, get_device, seed_everything


def test_normalisation_constants_match_lab_04():
    assert CIFAR10_MEAN == (0.4914, 0.4822, 0.4465)
    assert CIFAR10_STD == (0.2023, 0.1994, 0.2010)


def test_ten_class_names():
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES[0] == 'airplane'


def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_get_device_returns_a_valid_device():
    assert get_device().type in {'cuda', 'mps', 'cpu'}
