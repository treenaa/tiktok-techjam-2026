"""Lightweight frequency/texture branch for complementary forensic evidence."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .encoders import ModelError
from .specs import IMAGENET_STATS, RGBStats


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class InputDenormalizer(nn.Module):
    """Explicitly recover RGB `[0,1]` values from the visual encoder input."""

    def __init__(self, normalization: Optional[RGBStats] = IMAGENET_STATS) -> None:
        super().__init__()
        if normalization is None:
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            mean, std = normalization
        if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
            raise ValueError("normalization must contain three means and positive standard deviations")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images: Tensor) -> Tensor:
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3:
            raise ModelError("forensic input must be a BCHW tensor with three RGB channels")
        if not images.is_floating_point():
            raise ModelError("forensic input must be floating point")
        mean = self.mean.to(dtype=images.dtype)
        std = self.std.to(dtype=images.dtype)
        return (images * std + mean).clamp(0.0, 1.0)


class LogMagnitudeFFT(nn.Module):
    """RGB image -> standardised, centred log FFT magnitude map.

    The spatial mean is removed before the FFT so the DC brightness component
    cannot dominate the branch. Per-image standardisation keeps this branch
    focused on the frequency pattern rather than arbitrary FFT scale.
    """

    def __init__(
        self,
        normalization: Optional[RGBStats] = IMAGENET_STATS,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        self.denormalize = InputDenormalizer(normalization)
        self.eps = float(eps)
        self.register_buffer(
            "luma_weights",
            torch.tensor((0.2989, 0.5870, 0.1140), dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        rgb = self.denormalize(images)
        # FFT in float32 is stable and supported more consistently than fp16.
        rgb32 = rgb.float()
        gray = (rgb32 * self.luma_weights).sum(dim=1, keepdim=True)
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
        # Eliminate floating-point DC residue explicitly. A perfectly constant
        # image should not acquire an artificial centre-frequency spike merely
        # because its RGB-to-luma sum rounded a few ulps away from zero.
        spectrum = spectrum.clone()
        spectrum[..., 0, 0] = 0
        magnitude = torch.log1p(torch.abs(spectrum))
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (magnitude - mean) / std.clamp_min(self.eps)


class ForensicBranch(nn.Module):
    """Small CNN over log FFT magnitude; returns one forensic embedding."""

    def __init__(
        self,
        output_dim: int = 128,
        width: int = 32,
        normalization: Optional[RGBStats] = IMAGENET_STATS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(output_dim) < 1 or int(width) < 4:
            raise ValueError("output_dim must be positive and width must be at least 4")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.output_dim = int(output_dim)
        self.width = int(width)
        self.frequency = LogMagnitudeFFT(normalization=normalization)
        channels = (self.width, self.width * 2, self.width * 4)
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def frequency_map(self, images: Tensor) -> Tensor:
        """Expose the exact representation for audits and tests."""
        return self.frequency(images)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.cnn(self.frequency_map(images)))
