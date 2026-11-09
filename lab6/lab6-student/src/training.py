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

    # TODO: validate sizes and sample standard-normal latent vectors.
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if z_dim <= 0:
        raise ValueError("z_dim must be a positive integer")
    return torch.randn(batch_size, z_dim, device=device, generator=generator)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    """Enable or disable gradient accumulation for every module parameter."""

    # TODO: set requires_grad for every parameter.
    raise NotImplementedError


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

    # TODO: update D only. Generated images must not create gradients for G.
    raise NotImplementedError


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

    # TODO: freeze D parameters, update G through D, restore D, detach samples.
    raise NotImplementedError


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

    # TODO: alternate D/G updates and return batch-averaged statistics.
    raise NotImplementedError
