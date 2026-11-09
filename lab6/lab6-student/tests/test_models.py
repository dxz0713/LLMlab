import pytest
import torch
from torch import nn

from src.models import Discriminator, Generator, weights_init


def test_generator_output_shape_and_range():
    model = Generator(z_dim=16, base_channels=4)
    out = model(torch.randn(3, 16))
    assert out.shape == (3, 1, 28, 28)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_generator_rejects_wrong_shape():
    model = Generator(z_dim=8, base_channels=4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 1))


def test_generator_contains_transposed_convolution_and_batch_norm():
    model = Generator(z_dim=8, base_channels=4)
    modules = list(model.modules())
    assert sum(isinstance(m, nn.ConvTranspose2d) for m in modules) == 3
    assert sum(isinstance(m, nn.BatchNorm2d) for m in modules) == 2
    assert isinstance(model.net[-1], nn.Tanh)


def test_discriminator_output_is_one_logit_per_image():
    model = Discriminator(base_channels=4)
    out = model(torch.randn(5, 1, 28, 28))
    assert out.shape == (5,)


def test_discriminator_has_no_sigmoid():
    model = Discriminator(base_channels=4)
    assert not any(isinstance(m, nn.Sigmoid) for m in model.modules())


def test_discriminator_rejects_wrong_shape():
    model = Discriminator(base_channels=4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 28, 28))


def test_models_reject_non_positive_dimensions():
    with pytest.raises(ValueError):
        Generator(z_dim=0)
    with pytest.raises(ValueError):
        Discriminator(base_channels=0)


def test_weight_initialization_statistics_and_bias():
    torch.manual_seed(7)
    conv = nn.Conv2d(1, 64, 3, bias=True)
    weights_init(conv)
    assert abs(float(conv.weight.detach().mean())) < 0.01
    assert 0.015 < float(conv.weight.detach().std()) < 0.025
    assert torch.count_nonzero(conv.bias) == 0


def test_batch_norm_initialization():
    torch.manual_seed(7)
    bn = nn.BatchNorm2d(128)
    weights_init(bn)
    assert abs(float(bn.weight.detach().mean()) - 1.0) < 0.01
    assert 0.015 < float(bn.weight.detach().std()) < 0.025
    assert torch.count_nonzero(bn.bias) == 0
