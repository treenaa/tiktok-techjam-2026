from __future__ import annotations

import pytest

from src.models import CLIP_STATS, IMAGENET_STATS, ForensicBranch, LogMagnitudeFFT, ModelError

torch = pytest.importorskip("torch")


def normalize(raw, stats):
    mean, std = stats
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (raw - mean) / std


def test_frequency_representation_is_invariant_to_declared_normalization():
    torch.manual_seed(0)
    raw = torch.rand(2, 3, 24, 24)
    imagenet_map = LogMagnitudeFFT(IMAGENET_STATS)(normalize(raw, IMAGENET_STATS))
    clip_map = LogMagnitudeFFT(CLIP_STATS)(normalize(raw, CLIP_STATS))
    assert torch.allclose(imagenet_map, clip_map, atol=2e-5, rtol=2e-5)


def test_constant_image_has_finite_zero_frequency_map():
    raw = torch.full((2, 3, 16, 16), 0.5)
    frequency = LogMagnitudeFFT(None)(raw)
    assert torch.isfinite(frequency).all()
    assert torch.allclose(frequency, torch.zeros_like(frequency))


def test_forensic_branch_is_lightweight_shape_safe_and_differentiable():
    branch = ForensicBranch(output_dim=12, width=8, normalization=None, dropout=0.0)
    images = torch.rand(3, 3, 31, 27, requires_grad=True)
    features = branch(images)
    assert features.shape == (3, 12)
    assert torch.isfinite(features).all()
    features.sum().backward()
    assert branch.cnn[0].weight.grad is not None


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(3, 16, 16), torch.zeros(2, 1, 16, 16), torch.zeros(2, 3, 16, 16, dtype=torch.uint8)],
)
def test_forensic_branch_rejects_wrong_input_contract(bad):
    with pytest.raises(ModelError, match="forensic input"):
        LogMagnitudeFFT(None)(bad)
