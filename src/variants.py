"""The experiment grid. One entry per variant; each entry overrides BASE."""

BASE = {
    'model': 'vit',
    'patch_size': 4,
    'd_model': 192,
    'depth': 6,
    'heads': 4,
    'd_ff': 384,
    'dropout': 0.1,
    'pos': 'sinusoidal',
    'pool': 'cls',
    'augment': True,
    'epochs': 50,
    'batch_size': 128,
    'lr': 3e-4,  # chosen on validation over {1e-3, 3e-4}, then frozen for every variant
}

VARIANTS = {
    # references
    'R1': {'model': 'mlp'},
    'R2': {'model': 'cnn'},
    'R3': {'model': 'clip'},
    # reference ViT configuration
    'V0': {},
    # anatomy ablations
    'V1': {'pos': 'none'},
    'V2': {'pos': 'learned'},
    'V3': {'patch_size': 8},
    'V4': {'heads': 1},
    'V5': {'heads': 2},
    'V6': {'heads': 8},
    'V7a': {'depth': 2},
    'V7b': {'depth': 4},
    'V8': {'pool': 'mean'},
    'V9': {'augment': False},
}

DESCRIPTIONS = {
    'R1': 'MLP baseline (Lab 02)',
    'R2': 'CNN with dropout (Lab 02)',
    'R3': 'CLIP ViT-B/32 zero-shot (Lab 10)',
    'V0': 'reference ViT: patch 4, depth 6, 4 heads, CLS',
    'V1': 'no positional encoding',
    'V2': 'learned positional embeddings',
    'V3': 'patch size 8',
    'V4': '1 attention head',
    'V5': '2 attention heads',
    'V6': '8 attention heads',
    'V7a': 'depth 2',
    'V7b': 'depth 4',
    'V8': 'mean pooling instead of CLS',
    'V9': 'augmentation disabled',
}


def config(name):
    return {**BASE, **VARIANTS[name]}
