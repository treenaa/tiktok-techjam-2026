"""Render what the forensic branch sees.

`src.models.forensic.LogMagnitudeFFT` is the exact module the fusion detector's
forensic branch consumes: it removes the spatial mean, takes the centred FFT,
and standardises the log magnitude per image. This module runs that same
transform for display, so the picture on screen is the model's actual input --
not a decorative approximation of it.

The map is colourised with `app.theme.RAMP`, which is also the interface's
accent, so the page's identity is sampled from the signal being measured.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from src.models.forensic import LogMagnitudeFFT

from .theme import RAMP

_FFT = LogMagnitudeFFT(normalization=None).eval()

#: Percentile clip applied before colourising. The standardised log magnitude has
#: a very long upper tail (the DC-adjacent bins), which would otherwise compress
#: every structural detail into the bottom of the ramp.
_CLIP: Tuple[float, float] = (2.0, 99.5)


def _ramp_lut() -> np.ndarray:
    """256-entry RGB lookup table interpolated across the spectral ramp."""
    anchors = np.asarray(RAMP, dtype=np.float64)
    positions = np.linspace(0.0, 255.0, len(anchors))
    grid = np.arange(256, dtype=np.float64)
    return np.stack(
        [np.interp(grid, positions, anchors[:, channel]) for channel in range(3)],
        axis=1,
    ).astype(np.uint8)


_LUT = _ramp_lut()


def spectrum_array(image: Image.Image) -> np.ndarray:
    """Standardised log-FFT magnitude for ``image``, averaged over RGB."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        magnitude = _FFT(tensor)
    return magnitude[0].mean(0).numpy()


def spectrum_image(image: Image.Image, size: int = 256) -> Image.Image:
    """Colourised frequency signature of ``image``, ready for ``st.image``.

    Low frequencies sit at the centre: `LogMagnitudeFFT` returns a centred
    spectrum, so the bright core is the image's coarse structure and the outer
    field is fine detail. Blur empties the outside; noise fills it.
    """
    magnitude = spectrum_array(image)
    low, high = np.percentile(magnitude, _CLIP)
    if high <= low:
        high = low + 1e-6
    scaled = np.clip((magnitude - low) / (high - low), 0.0, 1.0)
    indexed = (scaled * 255.0).astype(np.uint8)
    coloured = _LUT[indexed]
    out = Image.fromarray(coloured, mode="RGB")
    if out.size != (size, size):
        out = out.resize((size, size), Image.Resampling.BILINEAR)
    return out


def spectral_distance(reference: Image.Image, other: Image.Image) -> float:
    """Relative L2 distance between two spectra.

    Reported next to each degraded view so the visible change in the frequency
    signature has a number attached to it rather than being left impressionistic.
    """
    a = spectrum_array(reference)
    b = spectrum_array(other)
    denominator = float(np.linalg.norm(a))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(b - a) / denominator)


def pair_strip(image: Image.Image, size: int = 224, gap: int = 2) -> Image.Image:
    """Compose ``image`` beside its own spectrum as one picture.

    Composed here rather than as two `st.image` calls so the seam, the square
    crop and the alignment are exact, and so each ladder card is a single image.
    """
    photo = _square(image.convert("RGB"), size)
    signature = spectrum_image(image, size)
    canvas = Image.new("RGB", (size * 2 + gap, size), (0x25, 0x32, 0x47))
    canvas.paste(photo, (0, 0))
    canvas.paste(signature, (size + gap, 0))
    return canvas


def _square(image: Image.Image, size: int) -> Image.Image:
    """Centre-crop to a square, then resize. Keeps every card the same shape."""
    width, height = image.size
    edge = min(width, height)
    left, top = (width - edge) // 2, (height - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def ambient(seed: int = 7, size: int = 256) -> Image.Image:
    """A generated spectrum for the empty state, so the page is never blank."""
    rng = np.random.default_rng(seed)
    grid = np.mgrid[0:size, 0:size].astype(np.float64)
    yy, xx = grid[0] - size / 2, grid[1] - size / 2
    radius = np.hypot(xx, yy) + 1.0
    field = (
        1.0 / radius
        + 0.06 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
        + 0.02 * rng.normal(size=(size, size))
    )
    field = np.log1p(np.clip(field, 0.0, None) * 40.0)
    low, high = np.percentile(field, _CLIP)
    scaled = np.clip((field - low) / max(high - low, 1e-6), 0.0, 1.0)
    return Image.fromarray(_LUT[(scaled * 255).astype(np.uint8)], mode="RGB")
