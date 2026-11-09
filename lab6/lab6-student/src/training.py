"""Alternating optimization utilities for GAN training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .losses import discriminator_loss, generator_loss


def sample_noise(
    batch_size: int,
    z_dim: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample standard-normal latent vectors with shape ``(N, z_dim)``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if z_dim <= 0:
        raise ValueError("z_dim must be a positive integer")
    return torch.randn(batch_size, z_dim, device=device, generator=generator)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    """Enable or disable gradient accumulation for every module parameter."""

    for param in module.parameters():
        param.requires_grad = enabled


def discriminator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real_images: torch.Tensor,
    optimizer_d: torch.optim.Optimizer,
    z_dim: int,
    real_label: float = 1.0,
    z: torch.Tensor | None = None,
) -> dict[str, float]:
    """Update D once while treating generated images as constants."""

    generator.train()
    discriminator.train()
    if z is None:
        z = sample_noise(real_images.shape[0], z_dim, real_images.device)

    optimizer_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake_images = generator(z)

    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)
    loss = discriminator_loss(real_logits, fake_logits, real_label)
    loss.backward()
    optimizer_d.step()

    with torch.no_grad():
        d_real = torch.sigmoid(real_logits).mean().item()
        d_fake = torch.sigmoid(fake_logits).mean().item()

    return {"d_loss": loss.item(), "d_real": d_real, "d_fake": d_fake}


def generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    batch_size: int,
    optimizer_g: torch.optim.Optimizer,
    z_dim: int,
    device: torch.device | str,
    z: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Update G once while using D only as a differentiable scoring function."""

    generator.train()
    discriminator.train()
    set_requires_grad(discriminator, False)
    optimizer_g.zero_grad(set_to_none=True)

    if z is None:
        z = sample_noise(batch_size, z_dim, device)

    fake_images = generator(z)
    fake_logits = discriminator(fake_images)
    loss = generator_loss(fake_logits)
    loss.backward()
    optimizer_g.step()

    set_requires_grad(discriminator, True)

    with torch.no_grad():
        g_score = torch.sigmoid(fake_logits).mean().item()

    return {"g_loss": loss.item(), "g_score": g_score}, fake_images.detach()


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: Iterable,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    z_dim: int,
    device: torch.device | str,
    real_label: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run alternating D/G updates and return batch-averaged statistics."""

    sum_stats = {"d_loss": 0.0, "d_real": 0.0, "d_fake": 0.0, "g_loss": 0.0, "g_score": 0.0}
    n = 0

    for batch in loader:
        if max_batches is not None and n >= max_batches:
            break

        real_images = batch[0].to(device)
        d_stats = discriminator_step(generator, discriminator, real_images, optimizer_d, z_dim, real_label)
        g_stats, _ = generator_step(generator, discriminator, real_images.shape[0], optimizer_g, z_dim, device)

        for key in sum_stats:
            sum_stats[key] += d_stats.get(key, 0.0) + g_stats.get(key, 0.0)

        n += 1

    if n == 0:
        raise ValueError("loader produced no batches")

    for key in sum_stats:
        sum_stats[key] /= n

    return sum_stats
