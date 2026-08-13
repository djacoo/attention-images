import torch

from src import lab05


def test_positional_encoding_varies_by_token_not_by_batch():
    """Lab 05 stores PE sequence-first; batch-first input would index the wrong axis.

    Same token position must get the same encoding in every batch element,
    and different positions must get different encodings.
    """
    pe = lab05.PositionalEncoding(d_model=8, dropout=0.0)
    x = torch.zeros(3, 5, 8)  # (batch, tokens, d_model)
    out = pe(x)

    assert out.shape == (3, 5, 8)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0], out[2])
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_attention_weights_are_a_distribution():
    attn = lab05.ScaledDotProductAttention()
    q = torch.randn(2, 4, 6, 8)  # (batch, heads, tokens, d_k)
    out, weights = attn(q, q, q)

    assert out.shape == (2, 4, 6, 8)
    assert weights.shape == (2, 4, 6, 6)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-5)


def test_multihead_shapes():
    mha = lab05.MultiHeadAttention(h=4, d_model=16, dropout=0.0)
    x = torch.randn(2, 7, 16)
    out, weights = mha(x, x, x)

    assert out.shape == (2, 7, 16)
    assert weights.shape == (2, 4, 7, 7)


def test_encoder_does_not_leak_across_batch_elements():
    """SublayerConnection indexes its sublayer output with [0].

    If the feed-forward sublayer is passed unwrapped, that slices the tensor
    and broadcasts batch element 0 across the batch. Encoding two samples
    together must equal encoding them one at a time.
    """
    torch.manual_seed(0)
    layer = lab05.EncoderLayer(
        16,
        lab05.MultiHeadAttention(h=4, d_model=16, dropout=0.0),
        lab05.FeedForward(d_model=16, d_ff=32, dropout=0.0),
        dropout=0.0,
    )
    encoder = lab05.Encoder(layer, N=2).eval()

    a = torch.randn(1, 5, 16)
    b = torch.randn(1, 5, 16)

    together = encoder(torch.cat([a, b], dim=0), None)
    apart = torch.cat([encoder(a, None), encoder(b, None)], dim=0)

    assert torch.allclose(together, apart, atol=1e-5)
