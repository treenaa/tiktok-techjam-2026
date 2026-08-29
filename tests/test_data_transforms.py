"""Competition transform suite: coverage, determinism, dimensions, robustness."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    BLUR_SIGMAS,
    CROP_RATIO,
    EVAL_TRANSFORM_NAMES,
    JITTER_DELTA,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    TRANSFORM_FAMILIES,
    TRANSFORM_REGISTRY,
    CenterCropResize,
    ColorJitter,
    Compose,
    GaussianBlur,
    GaussianNoise,
    Identity,
    JPEGCompression,
    RandomCompetitionTransform,
    ResizeRoundTrip,
    build_eval_suite,
    get_transform,
    list_transforms,
)
from test_data_fixtures import IMAGE_SIZE, make_image


@pytest.fixture
def img():
    return make_image(seed=1)


def arr(image):
    return np.asarray(image, dtype=np.float32)


def diff(a, b):
    return float(np.abs(arr(a) - arr(b)).mean())


# -- the suite covers exactly what the competition specifies ---------------
def test_required_parameters_are_present():
    assert JPEG_QUALITIES == (90, 70, 50, 30)
    assert BLUR_SIGMAS == (0.5, 1.0, 2.0)
    assert RESIZE_SCALES == (0.5, 0.25)
    assert NOISE_SIGMAS == (0.02, 0.05, 0.10)
    assert JITTER_DELTA == 0.20
    assert CROP_RATIO == 0.80


def test_registry_contains_every_named_transform():
    expected = {
        "clean",
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_sigma0.5", "blur_sigma1.0", "blur_sigma2.0",
        "resize_0.5x", "resize_0.25x",
        "noise_sigma0.02", "noise_sigma0.05", "noise_sigma0.10",
        "jitter_brightness_up", "jitter_brightness_down",
        "jitter_contrast_up", "jitter_contrast_down",
        "jitter_saturation_up", "jitter_saturation_down",
        "crop_0.8",
    }
    assert set(TRANSFORM_REGISTRY) == expected
    assert set(EVAL_TRANSFORM_NAMES) == expected
    assert EVAL_TRANSFORM_NAMES[0] == "clean"


def test_families_partition_the_suite():
    assert set(TRANSFORM_FAMILIES) == {
        "identity", "jpeg", "blur", "resize", "noise", "jitter", "crop"
    }
    assert sum(len(v) for v in TRANSFORM_FAMILIES.values()) == len(TRANSFORM_REGISTRY)
    assert len(list_transforms("jpeg")) == 4


def test_unknown_names_raise():
    with pytest.raises(KeyError, match="unknown transform"):
        get_transform("jpeg_q42")
    with pytest.raises(KeyError, match="unknown family"):
        list_transforms("sharpen")


# -- A. deterministic named transforms ------------------------------------
@pytest.mark.parametrize("name", EVAL_TRANSFORM_NAMES)
def test_named_transforms_are_deterministic(name, img):
    """Same transform, same input -> byte-identical output (noise is seeded)."""
    a, b = get_transform(name)(img), get_transform(name)(img)
    assert np.array_equal(arr(a), arr(b)), name


@pytest.mark.parametrize("name", EVAL_TRANSFORM_NAMES)
def test_named_transforms_preserve_size_and_mode(name, img):
    out = get_transform(name)(img)
    assert out.size == img.size == IMAGE_SIZE, name
    assert out.mode == "RGB", name


@pytest.mark.parametrize("name", [n for n in EVAL_TRANSFORM_NAMES if n != "clean"])
def test_named_transforms_actually_change_the_image(name, img):
    assert diff(get_transform(name)(img), img) > 0.0, name


def test_clean_is_an_identity_but_returns_a_copy(img):
    out = get_transform("clean")(img)
    assert np.array_equal(arr(out), arr(img))
    assert out is not img


def test_transform_objects_expose_name_family_and_params():
    t = get_transform("jpeg_q30")
    assert t.name == "jpeg_q30" and t.family == "jpeg" and t.params == {"quality": 30}
    assert "quality=30" in repr(t)


def test_non_image_input_is_rejected():
    with pytest.raises(TypeError, match="PIL.Image"):
        get_transform("jpeg_q50")(np.zeros((4, 4, 3)))


# -- per-family behaviour --------------------------------------------------
def test_lower_jpeg_quality_distorts_more(img):
    errors = [diff(JPEGCompression(q)(img), img) for q in (90, 70, 50, 30)]
    assert errors == sorted(errors), errors


def test_larger_blur_sigma_smooths_more(img):
    variances = [float(arr(GaussianBlur(s)(img)).var()) for s in (0.0, 0.5, 1.0, 2.0)]
    assert variances == sorted(variances, reverse=True), variances


def test_blur_sigma_zero_is_a_no_op(img):
    assert diff(GaussianBlur(0.0)(img), img) == 0.0


def test_resize_round_trip_restores_the_original_size(img):
    for scale in RESIZE_SCALES:
        out = ResizeRoundTrip(scale)(img)
        assert out.size == img.size
    assert diff(ResizeRoundTrip(0.25)(img), img) > diff(ResizeRoundTrip(0.5)(img), img)


def test_larger_noise_sigma_adds_more_noise(img):
    errors = [diff(GaussianNoise(s, seed=0)(img), img) for s in (0.02, 0.05, 0.10)]
    assert errors == sorted(errors), errors
    # sigma is on the [0, 1] scale: sigma=0.05 -> ~12.75/255 mean |error| * 0.8
    assert 5 < errors[1] < 20


def test_unseeded_noise_is_stochastic(img):
    t = GaussianNoise(0.05, seed=None)
    assert not np.array_equal(arr(t(img)), arr(t(img)))


def test_noise_output_stays_in_range(img):
    out = arr(GaussianNoise(0.5, seed=0)(img))
    assert out.min() >= 0 and out.max() <= 255


def test_color_jitter_directions(img):
    assert arr(ColorJitter(brightness=1.2)(img)).mean() > arr(img).mean()
    assert arr(ColorJitter(brightness=0.8)(img)).mean() < arr(img).mean()
    # +-20% saturation moves colourfulness in opposite directions.
    def chroma(image):
        a = arr(image)
        return float((a.max(axis=2) - a.min(axis=2)).mean())

    assert chroma(ColorJitter(saturation=1.2)(img)) > chroma(img)
    assert chroma(ColorJitter(saturation=0.8)(img)) < chroma(img)


def test_color_jitter_identity_factors_are_a_no_op(img):
    assert diff(ColorJitter()(img), img) == 0.0


def test_center_crop_ratio_and_resize_back(img):
    w, h = img.size
    cropped = CenterCropResize(0.8, resize_back=False)(img)
    assert cropped.size == (round(w * 0.8), round(h * 0.8))
    assert CenterCropResize(0.8)(img).size == img.size


def test_compose_applies_in_order_and_names_itself(img):
    chain = Compose([GaussianBlur(1.0), JPEGCompression(30)])
    assert chain.name == "blur_sigma1.0+jpeg_q30"
    assert chain(img).size == img.size
    assert diff(chain(img), img) > diff(GaussianBlur(1.0)(img), img)


@pytest.mark.parametrize(
    "factory,bad",
    [
        (JPEGCompression, 0), (JPEGCompression, 101),
        (GaussianBlur, -1.0), (ResizeRoundTrip, 0.0), (ResizeRoundTrip, 1.5),
        (GaussianNoise, -0.1), (CenterCropResize, 0.0), (CenterCropResize, 1.5),
    ],
)
def test_out_of_range_parameters_are_rejected(factory, bad):
    with pytest.raises(ValueError):
        factory(bad)


# -- extreme / awkward inputs ---------------------------------------------
@pytest.mark.parametrize("size", [(1, 1), (2, 3), (32, 32), (7, 129)])
def test_transforms_survive_tiny_and_odd_sizes(size):
    """0.25x of a small image must not collapse a dimension to zero."""
    small = make_image(seed=2, size=size)
    for name in EVAL_TRANSFORM_NAMES:
        out = get_transform(name)(small)
        assert out.size == size, (name, size)


@pytest.mark.parametrize(
    "transform",
    [
        JPEGCompression(1), JPEGCompression(100),
        GaussianBlur(50.0), ResizeRoundTrip(0.01),
        GaussianNoise(1.0, seed=0), GaussianNoise(0.0, seed=0),
        ColorJitter(brightness=0.0), ColorJitter(contrast=5.0, saturation=0.0),
        CenterCropResize(0.01), CenterCropResize(1.0),
    ],
)
def test_extreme_transform_parameters_run_without_errors(transform):
    image = make_image(seed=3, size=(16, 16))
    out = transform(image)
    assert out.size == image.size
    assert np.isfinite(arr(out)).all()


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_non_rgb_inputs_are_handled(mode):
    image = make_image(seed=4, size=(16, 16)).convert(mode)
    for name in EVAL_TRANSFORM_NAMES:
        out = get_transform(name)(image)
        assert out.size == image.size, (name, mode)


def test_build_eval_suite(img):
    suite = build_eval_suite()
    assert len(suite) == len(EVAL_TRANSFORM_NAMES)
    assert all(t(img).size == img.size for t in suite.values())
    assert set(build_eval_suite(["clean", "jpeg_q30"])) == {"clean", "jpeg_q30"}


# -- B. stochastic sampling ------------------------------------------------
def test_sampler_is_reproducible_under_a_seed():
    a = [RandomCompetitionTransform(seed=13).sample().name for _ in range(20)]
    b = [RandomCompetitionTransform(seed=13).sample().name for _ in range(20)]
    c = [RandomCompetitionTransform(seed=99).sample().name for _ in range(20)]
    assert a == b
    assert a != c


def test_sampler_varies_across_draws_and_covers_all_families():
    sampler = RandomCompetitionTransform(seed=0)
    families = {sampler.sample().family for _ in range(300)}
    assert families == set(sampler.families)


def test_sampler_applies_and_preserves_size(img):
    sampler = RandomCompetitionTransform(seed=5)
    for _ in range(30):
        assert sampler(img).size == img.size


def test_sampler_restriction_to_families():
    sampler = RandomCompetitionTransform(families=("jpeg", "blur"), seed=1)
    assert {sampler.sample().family for _ in range(50)} == {"jpeg", "blur"}


def test_sampler_chains_multiple_ops():
    sampler = RandomCompetitionTransform(seed=2, n_ops=(2, 3))
    for _ in range(20):
        assert isinstance(sampler.sample(), Compose)


def test_sampler_p_identity():
    assert all(
        isinstance(RandomCompetitionTransform(seed=i, p_identity=1.0).sample(), Identity)
        for i in range(5)
    )
    sampler = RandomCompetitionTransform(seed=3, p_identity=0.5)
    names = [sampler.sample().name for _ in range(60)]
    assert 0 < names.count("clean") < 60


def test_sampled_noise_is_seeded_so_the_drawn_transform_is_reproducible(img):
    """A drawn transform is itself deterministic -- required to log/replay it."""
    sampler = RandomCompetitionTransform(families=("noise",), seed=8)
    drawn = sampler.sample()
    assert np.array_equal(arr(drawn(img)), arr(drawn(img)))


def test_sampled_noise_differs_between_draws(img):
    sampler = RandomCompetitionTransform(families=("noise",), seed=8)
    a, b = sampler.sample(), sampler.sample()
    assert not np.array_equal(arr(a(img)), arr(b(img)))


def test_set_seed_restarts_the_stream():
    sampler = RandomCompetitionTransform(seed=4)
    first = [sampler.sample().name for _ in range(5)]
    sampler.set_seed(4)
    assert [sampler.sample().name for _ in range(5)] == first


def test_sampler_weights_bias_the_draw():
    sampler = RandomCompetitionTransform(families=("jpeg", "blur"), weights=(1.0, 0.0), seed=6)
    assert {sampler.sample().family for _ in range(30)} == {"jpeg"}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"families": ("sharpen",)}, "unknown transform families"),
        ({"families": ("jpeg",), "weights": (1.0, 2.0)}, "weights must match"),
        ({"p_identity": 1.5}, "p_identity"),
        ({"n_ops": (3, 1)}, "invalid n_ops"),
    ],
)
def test_sampler_rejects_bad_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RandomCompetitionTransform(**kwargs)
