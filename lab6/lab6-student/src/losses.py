"""Numerically stable non-saturating GAN losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    real_label: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy loss for real and generated samples.

    ``real_label`` may be set below 1.0 for one-sided label smoothing.
    Inputs are logits; do not apply sigmoid before calling this function.
    """

    if not (0.0 < real_label <= 1.0):
        raise ValueError("real_label must be in the range (0, 1]")
    real_t = torch.full_like(real_logits, real_label)
    fake_t = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_t)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_t)
    return real_loss + fake_loss


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: make generated samples look real."""

    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
