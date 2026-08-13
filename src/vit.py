"""Vision Transformer assembled from Lab 05's encoder blocks.

Recipe — deck 4, "Vision Transformer (VIT)": split the image into patches,
flatten them, project to lower-dimensional embeddings, add positional
embeddings, feed the sequence to a standard transformer encoder.
"""

import torch
import torch.nn as nn

from src import lab05


class PatchEmbedding(nn.Module):
    """Split into patches, flatten, project — deck 4; notes §83.4.1."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=192):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, d_model)

    def forward(self, x):
        b = x.size(0)
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # (b, c, h/p, w/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, self.n_patches, -1)
        return self.proj(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable position matrix added to the tokens — notes §84.7."""

    def __init__(self, n_tokens, d_model, dropout):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe)


class ViT(nn.Module):
    """Patch embedding + Lab 05 encoder + linear classifier head.

    pos:  'sinusoidal' (Lab 05) | 'learned' (notes §84.7) | 'none'
    pool: 'cls' (notes §83.4.1) | 'mean' (global average pooling, notes §17.5)
    """

    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=192,
                 depth=6, heads=4, d_ff=384, dropout=0.1, num_classes=10,
                 pos='sinusoidal', pool='cls'):
        super().__init__()
        self.pool = pool
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)

        n_tokens = self.patch_embed.n_patches
        if pool == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            n_tokens += 1
        else:
            self.cls_token = None

        if pos == 'sinusoidal':
            self.pos = lab05.PositionalEncoding(d_model, dropout)
        elif pos == 'learned':
            self.pos = LearnedPositionalEncoding(n_tokens, d_model, dropout)
        else:
            self.pos = nn.Dropout(dropout)

        layer = lab05.EncoderLayer(
            d_model,
            lab05.MultiHeadAttention(heads, d_model, dropout),
            lab05.FeedForward(d_model, d_ff, dropout),
            dropout,
        )
        self.encoder = lab05.Encoder(layer, depth)
        self.head = nn.Linear(d_model, num_classes)

        # Glorot init, as Lab 05's make_model does.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pos(x)
        x = self.encoder(x, None)
        x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)
        return self.head(x)
