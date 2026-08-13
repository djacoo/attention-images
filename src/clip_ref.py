"""CLIP zero-shot classification — Lab 10, VLMs.ipynb; notes §86.6."""

import json
from pathlib import Path

import numpy as np
import torch

from src.data import CLASS_NAMES, get_loaders, get_device
from src.evaluate import score

PROMPT_TEMPLATE = 'A photo of a {}'  # Lab 10, zero_shot_classify


def build_prompts():
    return [PROMPT_TEMPLATE.format(name) for name in CLASS_NAMES]


def run(results_dir='results'):
    """Zero-shot classify the CIFAR-10 test set and save the result as R3."""
    import open_clip

    device = get_device()
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    with torch.no_grad():
        text = model.encode_text(tokenizer(build_prompts()).to(device))
        text = text / text.norm(dim=-1, keepdim=True)

    _, _, test_loader = get_loaders(batch_size=128, augment=False)
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            features = model.encode_image(images.to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            all_predictions.extend((features @ text.T).argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())

    result = {
        'variant': 'R3',
        'seed': 0,
        'config': {'model': 'clip', 'backbone': 'ViT-B-32',
                   'pretrained': 'laion2b_s34b_b79k', 'prompt': PROMPT_TEMPLATE},
        **score(np.array(all_labels), np.array(all_predictions)),
    }

    path = Path(results_dir) / 'R3_seed0.json'
    path.write_text(json.dumps(result, indent=2))
    return path


if __name__ == '__main__':
    print(run())
