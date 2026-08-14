"""Transformer encoder blocks — Lab 05, Vanilla_transformer.ipynb.

The lab ships as a TODO skeleton; the TODOs are completed here as the lab
instructs. Two deviations are marked LAYOUT FIX below, both required to run the
encoder on batch-first image-patch tensors.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — Lab 05; notes §33.3."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # LAYOUT FIX: the lab stores (max_len, 1, d_model) and slices dim 0, which is
        # sequence-first. MultiHeadAttention reads nbatches = query.size(0), i.e.
        # batch-first, and so does every DataLoader in the course. Same sinusoid,
        # stored batch-first so it is indexed on the token axis.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention — Lab 05; notes §34.2."""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention — Lab 05; notes §35."""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)

        query = self.query(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.key(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.value(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, attn = self.attn(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.output(x)
        return x, attn


class FeedForward(nn.Module):
    """Pointwise feed-forward sub-layer — Lab 05; notes §36.1."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """Layer normalization — Lab 05, verbatim; notes §36.2."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection around a sub-layer — Lab 05, verbatim.

    The [0] index expects the sub-layer to return a tuple, as attention does.
    EncoderLayer wraps the feed-forward call to satisfy that.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))[0])


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention then feed-forward — Lab 05; notes §33.4."""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # LAYOUT FIX: wrapped in a 1-tuple because SublayerConnection indexes [0].
        # Passing self.feed_forward directly would slice the tensor and broadcast
        # batch element 0 across the batch.
        return self.sublayer[1](x, lambda x: (self.feed_forward(x),))


class Encoder(nn.Module):
    """Stack of N encoder layers — Lab 05; notes §33.4."""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
