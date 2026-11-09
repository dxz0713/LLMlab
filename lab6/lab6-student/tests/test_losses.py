import pytest
import torch
import torch.nn.functional as F

from src.losses import discriminator_loss, generator_loss


def test_discriminator_loss_matches_pytorch_definition():
    real = torch.tensor([1.0, -0.5])
    fake = torch.tensor([-1.0, 0.25])
    expected = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
    expected += F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    assert torch.allclose(discriminator_loss(real, fake), expected)


def test_generator_loss_matches_non_saturating_objective():
    fake = torch.tensor([-2.0, 0.0, 2.0])
    expected = F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
    assert torch.allclose(generator_loss(fake), expected)


def test_losses_are_finite_for_extreme_logits():
    real = torch.tensor([1000.0, -1000.0])
    fake = torch.tensor([-1000.0, 1000.0])
    assert torch.isfinite(discriminator_loss(real, fake))
    assert torch.isfinite(generator_loss(fake))


def test_label_smoothing_changes_only_real_target():
    real = torch.zeros(4)
    fake = torch.zeros(4)
    smoothed = discriminator_loss(real, fake, real_label=0.9)
    manual = F.binary_cross_entropy_with_logits(real, torch.full_like(real, 0.9))
    manual += F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    assert torch.allclose(smoothed, manual)


@pytest.mark.parametrize("bad_label", [0.0, -0.1, 1.1])
def test_bad_real_label_is_rejected(bad_label):
    with pytest.raises(ValueError):
        discriminator_loss(torch.zeros(1), torch.zeros(1), bad_label)


def test_losses_backpropagate():
    real = torch.randn(3, requires_grad=True)
    fake = torch.randn(3, requires_grad=True)
    (discriminator_loss(real, fake) + generator_loss(fake)).backward()
    assert real.grad is not None and torch.isfinite(real.grad).all()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()

