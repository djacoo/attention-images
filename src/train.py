"""Training loop — structure from Lab 02's train_model, Adam and CrossEntropyLoss
as in Labs 02/03/04. One JSON result per (variant, seed)."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.baselines import CNN, MLP
from src.data import get_loaders, get_device, seed_everything
from src.evaluate import collect_predictions, score
from src.variants import config
from src.vit import ViT


def build_model(cfg, num_classes=10):
    if cfg['model'] == 'mlp':
        return MLP(num_classes=num_classes)
    if cfg['model'] == 'cnn':
        return CNN(num_classes=num_classes)
    return ViT(
        patch_size=cfg['patch_size'], d_model=cfg['d_model'], depth=cfg['depth'],
        heads=cfg['heads'], d_ff=cfg['d_ff'], dropout=cfg['dropout'],
        num_classes=num_classes, pos=cfg['pos'], pool=cfg['pool'],
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Training when an optimizer is given — Lab 02."""
    model.train() if optimizer else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total


def train_variant(name, seed=0, epochs=None, lr=None, results_dir='results'):
    cfg = config(name)
    epochs = epochs if epochs is not None else cfg['epochs']
    lr = lr if lr is not None else cfg['lr']

    seed_everything(seed)
    device = get_device()
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=cfg['batch_size'], augment=cfg['augment'], seed=seed)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc})
        print(f'{name} seed{seed} epoch {epoch + 1}/{epochs} '
              f'train {train_acc:.2f}% val {val_acc:.2f}%')

    labels, predictions = collect_predictions(model, test_loader, device)
    result = {
        'variant': name,
        'seed': seed,
        'config': cfg,
        'epochs': epochs,
        'lr': lr,
        'parameters': sum(p.numel() for p in model.parameters()),
        'history': history,
        **score(labels, predictions),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f'{name}_seed{seed}.json'
    path.write_text(json.dumps(result, indent=2))

    torch.save(model.state_dict(), results_dir / f'{name}_seed{seed}.pt')
    return path


if __name__ == '__main__':
    import sys
    from src.variants import VARIANTS

    names = sys.argv[1:] or [n for n in VARIANTS if n != 'R3']
    for name in names:
        for seed in (0, 1, 2):
            # skip finished runs so an interrupted grid can be resumed
            if (Path('results') / f'{name}_seed{seed}.json').exists():
                print(f'{name} seed{seed} already done, skipping')
                continue
            train_variant(name, seed=seed)
