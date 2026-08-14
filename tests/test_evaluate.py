import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import collect_predictions, score


def test_score_on_perfect_predictions():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = score(labels, labels.copy())

    assert out['test_accuracy'] == 100.0
    assert len(out['confusion_matrix']) == 10
    assert out['per_class']['airplane']['recall'] == 1.0


def test_score_counts_a_single_error():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    out = score(labels, predictions)

    assert out['test_accuracy'] == 75.0
    assert out['confusion_matrix'][0][1] == 1


def test_collect_predictions_shapes():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    labels, predictions = collect_predictions(model, loader, torch.device('cpu'))
    assert labels.shape == (8,)
    assert predictions.shape == (8,)
