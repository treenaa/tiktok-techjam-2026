"""Correctness audit of the transform implementations.

These tests check that each transform does what its *name* claims -- not merely
that it changes pixels.  A blur that silently replaced JPEG would pass a naive
"output differs from input" test; these do not.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    CenterCropResize,
    ColorJitter,
    GaussianBlur,
    GaussianNoise,
    JPEGCompression,
    RandomCompetitionTransform,
    ResizeRoundTrip,
    get_eval_transform,
    list_eval_transforms,
)
from test_data_fixtures import make_image


@pytest.fixture
def img():
    return make_image(seed=11, size=(64, 48))


def arr(image):
    return np.asarray(image, dtype=np.float32)


# -- JPEG: genuinely re-encoded -------------------------------------------
@pytest.mark.parametrize("quality", [90, 70, 50, 30])
def test_jpeg_matches_an_independent_encode(img, quality):
    """Byte-identical to a plain PIL JPEG round-trip -- not an approximation."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=-1)
    buf.seek(0)
    reference = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    assert np.array_equal(arr(JPEGCompression(quality)(img)).astype(np.int16), reference)


def test_jpeg_quality_actually_reaches_the_codec(img):
    """Different qualities must produce different pixels and file sizes."""
    outputs = {}
    for quality in (90, 70, 50, 30):
        out = JPEGCompression(quality)(img)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        outputs[quality] = arr(out)
    for a, b in ((90, 70), (70, 50), (50, 30)):
        assert not np.array_equal(outputs[a], outputs[b]), (a, b)


def test_jpeg_is_lossy_in_the_expected_direction(img):
    errors = [float(np.abs(arr(JPEGCompression(q)(img)) - arr(img)).mean()) for q in (90, 70, 50, 30)]
    assert errors == sorted(errors)
    assert errors[0] > 0, "even q90 must be lossy"


def test_jpeg_preserves_size_and_channels(img):
    out = JPEGCompression(30)(img)
    assert out.size == img.size and out.mode == "RGB"


# -- resize: real downsample then upsample --------------------------------
@pytest.mark.parametrize("scale,expected", [(0.5, (32, 24)), (0.25, (16, 12))])
def test_resize_visits_the_downscaled_resolution(img, scale, expected, monkeypatch):
    """Instrument PIL to prove an intermediate low-res image really exists."""
    seen = []
    original = Image.Image.resize

    def spy(self, size, *args, **kwargs):
        seen.append(tuple(size))
        return original(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", spy)
    out = ResizeRoundTrip(scale)(img)
    assert seen[0] == expected, "must downscale to %s first, saw %s" % (expected, seen)
    assert seen[-1] == img.size, "must upscale back to the original size"
    assert out.size == img.size


def test_resize_destroys_high_frequency_detail(img):
    """The point of the corruption: fine detail cannot survive the round trip."""
    def high_freq_energy(image):
        a = arr(image).mean(axis=2)
        return float(np.abs(np.diff(a, axis=1)).mean())

    assert high_freq_energy(ResizeRoundTrip(0.25)(img)) < high_freq_energy(ResizeRoundTrip(0.5)(img))
    assert high_freq_energy(ResizeRoundTrip(0.5)(img)) < high_freq_energy(img)


def test_resize_round_trip_is_not_a_no_op(img):
    assert float(np.abs(arr(ResizeRoundTrip(0.5)(img)) - arr(img)).mean()) > 1.0


# -- noise: sensible value range ------------------------------------------
@pytest.mark.parametrize("sigma", [0.02, 0.05, 0.10])
def test_noise_sigma_is_on_the_zero_one_scale(sigma, img):
    """sigma is normalised intensity: measured std ~= sigma * 255."""
    delta = arr(GaussianNoise(sigma, seed=0)(img)) - arr(img)
    measured = float(delta.std()) / 255.0
    # Clipping at 0/255 shaves a little off, hence the asymmetric tolerance.
    assert 0.80 * sigma <= measured <= 1.10 * sigma, measured


def test_noise_is_zero_mean(img):
    delta = arr(GaussianNoise(0.05, seed=0)(img)) - arr(img)
    assert abs(float(delta.mean())) < 1.5


def test_noise_output_stays_in_the_uint8_range(img):
    for sigma in (0.02, 0.10, 0.5, 1.0):
        out = arr(GaussianNoise(sigma, seed=0)(img))
        assert out.min() >= 0.0 and out.max() <= 255.0


def test_noise_is_independent_across_pixels(img):
    """Not a constant offset or a single shared draw."""
    delta = (arr(GaussianNoise(0.10, seed=0)(img)) - arr(img))[..., 0]
    assert float(delta.std()) > 10
    assert abs(float(np.corrcoef(delta[:, :-1].ravel(), delta[:, 1:].ravel())[0, 1])) < 0.2


# -- center crop: 80% spatial ---------------------------------------------
def test_center_crop_is_eighty_percent_of_each_side(img):
    w, h = img.size
    cropped = CenterCropResize(0.80, resize_back=False)(img)
    assert cropped.size == (round(w * 0.80), round(h * 0.80))


def test_center_crop_is_centered(img):
    """The crop must come from the middle, not a corner."""
    w, h = img.size
    cropped = CenterCropResize(0.80, resize_back=False)(img)
    cw, ch = cropped.size
    expected = img.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    assert np.array_equal(arr(cropped), arr(expected))


def test_center_crop_resizes_back_by_default(img):
    assert CenterCropResize(0.80)(img).size == img.size
    assert get_eval_transform("crop_0.80")(img).size == img.size


# -- determinism / stochasticity ------------------------------------------
@pytest.mark.parametrize("name", list_eval_transforms())
def test_evaluation_transforms_are_deterministic_across_instances(name, img):
    """Two separately constructed instances must agree -- no shared RNG state."""
    first = get_eval_transform(name)(img)
    second = get_eval_transform(name)(img)
    assert np.array_equal(arr(first), arr(second)), name


@pytest.mark.parametrize("name", list_eval_transforms())
def test_evaluation_transforms_are_stable_under_repeated_calls(name, img):
    transform = get_eval_transform(name)
    outputs = [arr(transform(img)) for _ in range(3)]
    assert all(np.array_equal(outputs[0], o) for o in outputs[1:]), name


def test_training_augmentation_is_genuinely_stochastic(img):
    """Guard against the sampler collapsing to a single fixed transform."""
    sampler = RandomCompetitionTransform(seed=None)
    outputs = [arr(sampler(img)) for _ in range(12)]
    assert any(not np.array_equal(outputs[0], o) for o in outputs[1:])


def test_training_augmentation_does_not_repeat_one_family(img):
    sampler = RandomCompetitionTransform(seed=None)
    families = {sampler.sample().family for _ in range(200)}
    assert len(families) >= 4, families


def test_unseeded_sampler_differs_between_instances(img):
    a = [RandomCompetitionTransform(seed=None).sample().name for _ in range(8)]
    b = [RandomCompetitionTransform(seed=None).sample().name for _ in range(8)]
    assert a != b


def test_seeded_sampler_stays_reproducible(img):
    a = RandomCompetitionTransform(seed=21)
    b = RandomCompetitionTransform(seed=21)
    assert [a.sample().name for _ in range(10)] == [b.sample().name for _ in range(10)]


def test_sampled_noise_is_not_accidentally_frozen(img):
    """Each drawn noise transform gets its own seed, so views differ."""
    sampler = RandomCompetitionTransform(families=("noise",), seed=1)
    outputs = [arr(sampler(img)) for _ in range(6)]
    assert all(not np.array_equal(outputs[0], o) for o in outputs[1:])


def test_a_drawn_transform_is_itself_replayable(img):
    """Determinism *within* a draw: needed to log and reproduce an augmentation."""
    drawn = RandomCompetitionTransform(seed=5).sample()
    assert np.array_equal(arr(drawn(img)), arr(drawn(img)))


# -- colour jitter ---------------------------------------------------------
def test_colour_jitter_is_twenty_percent(img):
    for channel in ("brightness", "contrast", "saturation"):
        up = get_eval_transform("jitter_%s_up" % channel).params[channel]
        down = get_eval_transform("jitter_%s_down" % channel).params[channel]
        assert up == pytest.approx(1.2) and down == pytest.approx(0.8)


def test_colour_jitter_only_touches_its_own_channel():
    transform = get_eval_transform("jitter_contrast_up")
    assert transform.params["brightness"] == 1.0
    assert transform.params["saturation"] == 1.0
