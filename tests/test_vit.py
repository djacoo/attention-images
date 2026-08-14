import torch

from src.vit import PatchEmbedding, ViT


def test_patch_embedding_shape():
    embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, d_model=192)
    out = embed(torch.randn(2, 3, 32, 32))

    assert embed.n_patches == 64
    assert out.shape == (2, 64, 192)


def test_patch_embedding_reads_each_patch_independently():
    """Changing one patch must change only that patch's token."""
    embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, d_model=192)
    x = torch.zeros(1, 3, 32, 32)
    base = embed(x)

    x[:, :, 0:4, 0:4] = 1.0
    changed = embed(x)

    assert not torch.allclose(base[0, 0], changed[0, 0])
    assert torch.allclose(base[0, 1:], changed[0, 1:])


def test_forward_shape_with_cls():
    model = ViT(pos='sinusoidal', pool='cls').eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_forward_shape_with_mean_pool_and_no_cls_token():
    model = ViT(pos='sinusoidal', pool='mean').eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert model.cls_token is None


def test_pos_none_has_no_positional_parameters():
    model = ViT(pos='none')
    names = [n for n, _ in model.named_parameters()]
    assert not any('pos' in n for n in names)


def test_pos_learned_has_one_positional_parameter_per_token():
    model = ViT(pos='learned', pool='cls')
    assert model.pos.pe.shape == (1, 65, 192)


def test_patch_size_eight_gives_sixteen_tokens():
    model = ViT(patch_size=8, pool='mean').eval()
    assert model.patch_embed.n_patches == 16
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
