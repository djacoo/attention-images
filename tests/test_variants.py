import pytest

from src.variants import VARIANTS, config


def test_reference_configuration_matches_the_spec():
    v0 = config('V0')
    assert v0['model'] == 'vit'
    assert v0['patch_size'] == 4
    assert v0['d_model'] == 192
    assert v0['depth'] == 6
    assert v0['heads'] == 4
    assert v0['pos'] == 'sinusoidal'
    assert v0['pool'] == 'cls'
    assert v0['augment'] is True


@pytest.mark.parametrize('name', sorted(VARIANTS))
def test_every_variant_is_complete_and_consistent(name):
    cfg = config(name)
    assert cfg['model'] in {'vit', 'mlp', 'cnn', 'clip'}
    if cfg['model'] == 'vit':
        assert cfg['d_model'] % cfg['heads'] == 0
        assert 32 % cfg['patch_size'] == 0
        assert cfg['pos'] in {'sinusoidal', 'learned', 'none'}
        assert cfg['pool'] in {'cls', 'mean'}


def test_each_variant_changes_exactly_what_it_claims():
    assert config('V1')['pos'] == 'none'
    assert config('V2')['pos'] == 'learned'
    assert config('V3')['patch_size'] == 8
    assert config('V4')['heads'] == 1
    assert config('V5')['heads'] == 2
    assert config('V6')['heads'] == 8
    assert config('V7a')['depth'] == 2
    assert config('V7b')['depth'] == 4
    assert config('V8')['pool'] == 'mean'
    assert config('V9')['augment'] is False


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        config('nope')
