import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    discriminator_step,
    generator_step,
    sample_noise,
    set_requires_grad,
    train_one_epoch,
)


class TinyGenerator(nn.Module):
    def __init__(self, z_dim=5):
        super().__init__()
        self.linear = nn.Linear(z_dim, 16)

    def forward(self, z):
        return self.linear(z).reshape(-1, 1, 4, 4)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        return self.linear(x.flatten(1)).squeeze(1)


def make_pair():
    torch.manual_seed(0)
    g, d = TinyGenerator(), TinyDiscriminator()
    return g, d, torch.optim.SGD(g.parameters(), lr=0.05), torch.optim.SGD(d.parameters(), lr=0.05)


def params(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(not torch.allclose(a, b) for a, b in zip(before, module.parameters()))


def test_sample_noise_shape_device_and_distribution():
    z = sample_noise(2000, 7, "cpu")
    assert z.shape == (2000, 7)
    assert z.device.type == "cpu"
    assert abs(float(z.mean())) < 0.06
    assert 0.94 < float(z.std()) < 1.06


def test_sample_noise_is_reproducible_with_generator():
    a = sample_noise(3, 4, "cpu", torch.Generator().manual_seed(9))
    b = sample_noise(3, 4, "cpu", torch.Generator().manual_seed(9))
    assert torch.equal(a, b)


@pytest.mark.parametrize("batch,z_dim", [(0, 3), (2, 0), (-1, 2)])
def test_sample_noise_rejects_invalid_size(batch, z_dim):
    with pytest.raises(ValueError):
        sample_noise(batch, z_dim, "cpu")


def test_set_requires_grad_changes_all_parameters():
    model = TinyDiscriminator()
    set_requires_grad(model, False)
    assert all(not p.requires_grad for p in model.parameters())
    set_requires_grad(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_discriminator_step_updates_only_discriminator():
    g, d, opt_g, opt_d = make_pair()
    g_before, d_before = params(g), params(d)
    z = torch.randn(6, 5)
    stats = discriminator_step(g, d, torch.randn(6, 1, 4, 4), opt_d, 5, z=z)
    assert not changed(g_before, g)
    assert changed(d_before, d)
    assert set(stats) == {"d_loss", "d_real", "d_fake"}
    assert all(p.grad is None for p in g.parameters())


def test_generator_step_updates_only_generator_and_restores_d():
    g, d, opt_g, opt_d = make_pair()
    g_before, d_before = params(g), params(d)
    stats, fake = generator_step(g, d, 6, opt_g, 5, "cpu", z=torch.randn(6, 5))
    assert changed(g_before, g)
    assert not changed(d_before, d)
    assert fake.shape == (6, 1, 4, 4) and not fake.requires_grad
    assert set(stats) == {"g_loss", "g_score"}
    assert all(p.requires_grad for p in d.parameters())
    assert all(p.grad is None for p in d.parameters())


def test_train_one_epoch_returns_averages_and_updates_both_networks():
    g, d, opt_g, opt_d = make_pair()
    g_before, d_before = params(g), params(d)
    loader = DataLoader(TensorDataset(torch.randn(12, 1, 4, 4)), batch_size=4)
    stats = train_one_epoch(g, d, loader, opt_g, opt_d, 5, "cpu", max_batches=2)
    assert set(stats) == {"d_loss", "g_loss", "d_real", "d_fake", "g_score"}
    assert changed(g_before, g) and changed(d_before, d)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_train_one_epoch_rejects_empty_loader():
    g, d, opt_g, opt_d = make_pair()
    with pytest.raises(ValueError):
        train_one_epoch(g, d, [], opt_g, opt_d, 5, "cpu")

