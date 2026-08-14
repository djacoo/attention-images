import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import build_model, run_epoch, train_variant


def _tiny_loader():
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_build_model_dispatches_on_the_model_field():
    from src.baselines import CNN, MLP
    from src.vit import ViT

    assert isinstance(build_model({'model': 'mlp'}), MLP)
    assert isinstance(build_model({'model': 'cnn'}), CNN)
    assert isinstance(build_model({'model': 'vit', 'patch_size': 4, 'd_model': 192,
                                   'depth': 2, 'heads': 4, 'd_ff': 384, 'dropout': 0.1,
                                   'pos': 'sinusoidal', 'pool': 'cls'}), ViT)


def test_run_epoch_returns_loss_and_accuracy():
    model = build_model({'model': 'mlp'})
    loader = _tiny_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, acc = run_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                          torch.device('cpu'), optimizer)

    assert loss > 0
    assert 0.0 <= acc <= 100.0


def test_train_variant_writes_a_json_result(tmp_path, monkeypatch):
    monkeypatch.setattr('src.train.get_loaders',
                        lambda **kw: (_tiny_loader(), _tiny_loader(), _tiny_loader()))
    path = train_variant('R1', seed=0, epochs=1, results_dir=tmp_path)

    saved = json.loads(path.read_text())
    assert saved['variant'] == 'R1'
    assert saved['seed'] == 0
    assert 'test_accuracy' in saved
    assert 'per_class' in saved
    assert len(saved['confusion_matrix']) == 10
