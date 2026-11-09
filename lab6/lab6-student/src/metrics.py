"""Small diagnostics that do not require a pretrained evaluation network."""

from __future__ import annotations

import torch


def batch_diversity(images: torch.Tensor) -> torch.Tensor:
    """Average per-pixel standard deviation across a generated batch."""

    if images.ndim != 4:
        raise ValueError("images must have shape (N, C, H, W)")
    if images.shape[0] < 2:
        raise ValueError("batch_diversity needs at least two images")
    return images.std(dim=0, correction=0).mean()


def interpolate_latents(z_start: torch.Tensor, z_end: torch.Tensor, steps: int) -> torch.Tensor:
    """Linearly interpolate between equally shaped latent vectors, endpoints included."""

    if z_start.shape != z_end.shape:
        raise ValueError("z_start and z_end must have the same shape")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    # `steps` becomes a new leading axis, so append one singleton dimension
    # per input axis (not ndim - 1) for broadcasting.
    t = torch.linspace(0, 1, steps, device=z_start.device, dtype=z_start.dtype).view(
        steps, *([1] * z_start.ndim)
    )
    return (1 - t) * z_start + t * z_end
