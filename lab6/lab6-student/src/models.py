"""DCGAN generator and discriminator used in Lab 6."""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """Map a latent vector ``(N, z_dim)`` to images ``(N, 1, 28, 28)``."""

    def __init__(self, z_dim: int = 100, base_channels: int = 32):
        super().__init__()
        if z_dim <= 0 or base_channels <= 0:
            raise ValueError("z_dim and base_channels must be positive")
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # (N, z_dim, 1, 1) -> (N, 4B, 7, 7)
            nn.ConvTranspose2d(z_dim, base_channels * 4, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            # (N, 4B, 7, 7) -> (N, 2B, 14, 14)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            # (N, 2B, 14, 14) -> (N, 1, 28, 28)
            nn.ConvTranspose2d(base_channels * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape (N, {self.z_dim}), got {tuple(z.shape)}")
        return self.net(z.view(z.shape[0], self.z_dim, 1, 1))


class Discriminator(nn.Module):
    """Return one real/fake logit per input image (no sigmoid layer)."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.net = nn.Sequential(
            # (N, 1, 28, 28) -> (N, B, 14, 14)
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # (N, B, 14, 14) -> (N, 2B, 7, 7)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            # (N, 2B, 7, 7) -> (N, 1, 1, 1)
            nn.Conv2d(base_channels * 2, 1, kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError(f"x must have shape (N, 1, 28, 28), got {tuple(x.shape)}")
        return self.net(x).reshape(x.shape[0])


def weights_init(module: nn.Module) -> None:
    """DCGAN initialization: conv N(0, .02), batch-norm N(1, .02)."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
