"""CIFAR-10 loaders — dataset and augmentation from Lab 04, Self_supervised_learning.ipynb."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Lab 04, SimCLRTransform
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

SPLIT_SEED = 0  # fixed for every run so all variants see the same split


def seed_everything(seed):
    """Seeding as in Lab 06, Semi_Supervised_Learning.ipynb."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_transforms(augment):
    """Augmentation list from Lab 04's SimCLRTransform; flip also in Lab 02."""
    steps = []
    if augment:
        steps += [
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(steps)


def get_loaders(batch_size=128, augment=True, seed=0, workers=0, root='./data'):
    """Return train, validation and test loaders. 45k / 5k / 10k, notes §26.1.

    Loading single-process is deliberate. Worker processes load batches faster in
    isolation, but on this machine they compete with the GPU submission thread and
    make a training epoch 2.4x slower (24 s single-process against 57 s with 8
    workers). Augmentation costs 17 ms per batch, cheaper than the IPC to avoid it.
    """
    train_full = datasets.CIFAR10(root, train=True, download=True,
                                  transform=build_transforms(augment))
    val_full = datasets.CIFAR10(root, train=True, download=True,
                                transform=build_transforms(False))
    test = datasets.CIFAR10(root, train=False, download=True,
                            transform=build_transforms(False))

    split = torch.Generator().manual_seed(SPLIT_SEED)
    train_idx, val_idx = random_split(range(len(train_full)), [45000, 5000], generator=split)

    train = torch.utils.data.Subset(train_full, list(train_idx))
    val = torch.utils.data.Subset(val_full, list(val_idx))

    shuffle_gen = torch.Generator().manual_seed(seed)
    common = {'num_workers': workers, 'persistent_workers': workers > 0}
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=shuffle_gen, **common),
        DataLoader(val, batch_size=batch_size, **common),
        DataLoader(test, batch_size=batch_size, **common),
    )
