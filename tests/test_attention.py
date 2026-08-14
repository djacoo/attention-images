import torch

from src.attention import attention_maps, disable_head
from src.vit import ViT


def test_attention_maps_shape_per_layer():
    model = ViT(depth=3, heads=4, pool='cls').eval()
    maps = attention_maps(model, torch.randn(2, 3, 32, 32))

    assert len(maps) == 3
    assert maps[0].shape == (2, 4, 65, 65)


def test_attention_rows_sum_to_one():
    model = ViT(depth=2, heads=2, pool='cls').eval()
    maps = attention_maps(model, torch.randn(1, 3, 32, 32))
    row_sums = maps[0].sum(-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_disable_head_changes_output_then_restores_it():
    torch.manual_seed(0)
    model = ViT(depth=2, heads=4, pool='cls').eval()
    x = torch.randn(2, 3, 32, 32)

    before = model(x)
    with disable_head(model, layer=0, head=1):
        during = model(x)
    after = model(x)

    assert not torch.allclose(before, during)
    assert torch.allclose(before, after)


def test_disabling_every_head_in_a_layer_zeroes_that_attention_output():
    torch.manual_seed(0)
    model = ViT(depth=1, heads=2, pool='cls').eval()
    mha = model.encoder.layers[0].self_attn

    with disable_head(model, layer=0, head=0), disable_head(model, layer=0, head=1):
        assert torch.count_nonzero(mha.output.weight) == 0
