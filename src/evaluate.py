"""Metrics and figures — confusion matrix as in Lab 02, classification report as in Lab 03."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES


def collect_predictions(model, loader, device):
    """Run the model over a loader and return true and predicted labels — Lab 02."""
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


def score(labels, predictions):
    """Accuracy, confusion matrix and per-class precision/recall — Labs 02 and 03."""
    report = classification_report(labels, predictions, output_dict=True,
                                   labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))

    return {
        'test_accuracy': 100 * float((labels == predictions).mean()),
        'confusion_matrix': cm.tolist(),
        'per_class': {name: report[name] for name in CLASS_NAMES},
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
    }


def plot_confusion_matrix(cm, title, ax=None):
    """Heatmap of a confusion matrix — Lab 02's plot_confusion_matrix."""
    ax = ax or plt.subplots(figsize=(8, 7))[1]
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    return ax


def denormalise(image):
    """Undo the Lab 04 normalisation so an image can be displayed."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
