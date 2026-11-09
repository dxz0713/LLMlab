import pytest
import torch

from src.metrics import batch_diversity, interpolate_latents


def test_batch_diversity_zero_for_identical_images():
    images = torch.ones(4, 1, 3, 3)
    assert batch_diversity(images) == 0


def test_batch_diversity_matches_manual_value():
    images = torch.tensor([[[[0.0]]], [[[2.0]]]])
    assert torch.allclose(batch_diversity(images), torch.tensor(1.0))


def test_batch_diversity_requires_a_batch():
    with pytest.raises(ValueError):
        batch_diversity(torch.ones(1, 1, 3, 3))


def test_interpolation_includes_endpoints_and_midpoint():
    start = torch.tensor([0.0, 2.0])
    end = torch.tensor([2.0, 4.0])
    line = interpolate_latents(start, end, 3)
    assert line.shape == (3, 2)
    assert torch.equal(line[0], start)
    assert torch.equal(line[-1], end)
    assert torch.equal(line[1], torch.tensor([1.0, 3.0]))


def test_interpolation_supports_batched_inputs():
    start = torch.zeros(2, 4)
    end = torch.ones(2, 4)
    assert interpolate_latents(start, end, 5).shape == (5, 2, 4)


def test_interpolation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        interpolate_latents(torch.zeros(2), torch.zeros(3), 4)
    with pytest.raises(ValueError):
        interpolate_latents(torch.zeros(2), torch.ones(2), 1)

