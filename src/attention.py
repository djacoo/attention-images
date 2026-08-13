"""Attention-map extraction and head ablation.

Lab 05's ScaledDotProductAttention already returns the attention weights as its
second value, so the maps need no extra machinery — only a forward hook.
"""

from contextlib import contextmanager

import torch


def attention_maps(model, images):
    """Attention weights per encoder layer, each (batch, heads, tokens, tokens)."""
    captured = []
    handles = [
        layer.self_attn.register_forward_hook(lambda m, i, o: captured.append(o[1].detach()))
        for layer in model.encoder.layers
    ]

    try:
        with torch.no_grad():
            model(images)
    finally:
        for handle in handles:
            handle.remove()

    return captured


@contextmanager
def disable_head(model, layer, head):
    """Temporarily remove one attention head's contribution.

    The heads are concatenated before MultiHeadAttention.output, so head h owns
    input columns [h*d_k : (h+1)*d_k] of that linear layer's weight.
    """
    mha = model.encoder.layers[layer].self_attn
    start, end = head * mha.d_k, (head + 1) * mha.d_k
    saved = mha.output.weight.data[:, start:end].clone()

    mha.output.weight.data[:, start:end] = 0
    try:
        yield
    finally:
        mha.output.weight.data[:, start:end] = saved


def head_ablation_matrix(model, loader, device, num_classes=10):
    """Per-class accuracy drop for every (layer, head). Rows are heads, columns classes."""
    from src.evaluate import collect_predictions

    labels, predictions = collect_predictions(model, loader, device)
    baseline = [
        100 * float((predictions[labels == c] == c).mean()) if (labels == c).any() else 0.0
        for c in range(num_classes)
    ]

    rows = {}
    for layer_idx, layer in enumerate(model.encoder.layers):
        for head in range(layer.self_attn.h):
            with disable_head(model, layer_idx, head):
                _, ablated = collect_predictions(model, loader, device)
            rows[f'L{layer_idx}H{head}'] = [
                baseline[c] - (100 * float((ablated[labels == c] == c).mean())
                               if (labels == c).any() else 0.0)
                for c in range(num_classes)
            ]

    return baseline, rows
