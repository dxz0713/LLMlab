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

    # TODO: validate real_label and sum the real/fake BCE-with-logits terms.
    raise NotImplementedError


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: make generated samples look real."""

    # TODO: implement the non-saturating generator objective.
    raise NotImplementedError
