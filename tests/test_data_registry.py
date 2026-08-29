"""The central transform registry: official names, aliases, enumeration."""

from __future__ import annotations

import pytest

from src.data import (
    OFFICIAL_TRANSFORM_NAMES,
    TRANSFORM_ALIASES,
    build_eval_suite,
    canonical_transform_name,
    describe_eval_transforms,
    get_eval_transform,
    list_eval_transforms,
)

#: The competition specification, spelled out independently of the source.
OFFICIAL_SPEC = {
    "jpeg_90": ("jpeg", {"quality": 90}),
    "jpeg_70": ("jpeg", {"quality": 70}),
    "jpeg_50": ("jpeg", {"quality": 50}),
    "jpeg_30": ("jpeg", {"quality": 30}),
    "blur_0.5": ("blur", {"sigma": 0.5}),
    "blur_1.0": ("blur", {"sigma": 1.0}),
    "blur_2.0": ("blur", {"sigma": 2.0}),
    "resize_0.5": ("resize", {"scale": 0.5}),
    "resize_0.25": ("resize", {"scale": 0.25}),
    "noise_0.02": ("noise", {"sigma": 0.02}),
    "noise_0.05": ("noise", {"sigma": 0.05}),
    "noise_0.10": ("noise", {"sigma": 0.10}),
    "crop_0.80": ("crop", {"ratio": 0.80}),
}


@pytest.mark.parametrize("name", sorted(OFFICIAL_SPEC))
def test_every_official_transform_resolves_with_the_right_parameters(name):
    family, params = OFFICIAL_SPEC[name]
    transform = get_eval_transform(name)
    assert transform.name == name
    assert transform.family == family
    for key, value in params.items():
        assert transform.params[key] == pytest.approx(value)


def test_the_spec_examples_from_the_brief_all_work():
    for name in ("jpeg_30", "blur_2.0", "resize_0.25", "noise_0.10", "crop_0.80"):
        assert get_eval_transform(name).name == name


def test_colour_jitter_covers_all_three_channels_in_both_directions():
    names = list_eval_transforms("jitter")
    assert set(names) == {
        "jitter_brightness_up", "jitter_brightness_down",
        "jitter_contrast_up", "jitter_contrast_down",
        "jitter_saturation_up", "jitter_saturation_down",
    }
    for name in names:
        channel = name.split("_")[1]
        factor = get_eval_transform(name).params[channel]
        expected = 1.2 if name.endswith("_up") else 0.8
        assert factor == pytest.approx(expected), name


def test_enumeration_covers_exactly_the_official_suite():
    names = list_eval_transforms()
    assert names[0] == "clean"
    assert set(names) == set(OFFICIAL_SPEC) | set(list_eval_transforms("jitter")) | {"clean"}
    assert len(names) == 20
    assert names == list(OFFICIAL_TRANSFORM_NAMES)


def test_enumeration_can_exclude_the_clean_reference():
    names = list_eval_transforms(include_clean=False)
    assert "clean" not in names and len(names) == 19


def test_enumeration_order_is_stable():
    assert list_eval_transforms() == list_eval_transforms()


def test_names_are_machine_readable():
    """Names must be safe as filenames, dict keys and CSV column values."""
    for name in list_eval_transforms():
        assert name == name.strip().lower()
        assert " " not in name and "," not in name and "/" not in name
        assert all(c.isalnum() or c in "._" for c in name), name


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("jpeg_q30", "jpeg_30"), ("jpeg30", "jpeg_30"),
        ("blur_sigma2.0", "blur_2.0"),
        ("resize_0.25x", "resize_0.25"),
        ("noise_sigma0.10", "noise_0.10"),
        ("crop_0.8", "crop_0.80"),
        ("identity", "clean"), ("none", "clean"),
    ],
)
def test_deprecated_spellings_still_resolve(alias, canonical):
    assert canonical_transform_name(alias) == canonical
    assert get_eval_transform(alias).name == canonical


def test_alias_lookup_is_case_insensitive():
    assert canonical_transform_name("JPEG_30") == "jpeg_30"
    assert canonical_transform_name("  blur_2.0  ") == "blur_2.0"


def test_aliases_never_shadow_a_canonical_name():
    assert not set(TRANSFORM_ALIASES) & set(OFFICIAL_TRANSFORM_NAMES)


def test_unknown_names_fail_loudly_and_list_the_options():
    with pytest.raises(KeyError) as excinfo:
        get_eval_transform("jpeg_42")
    assert "jpeg_30" in str(excinfo.value)


def test_each_call_returns_an_independent_instance():
    a, b = get_eval_transform("noise_0.10"), get_eval_transform("noise_0.10")
    assert a is not b


def test_describe_eval_transforms_is_serialisable():
    import json

    spec = describe_eval_transforms()
    assert len(spec) == 20
    assert {"name", "family", "params", "severity"} <= set(spec[0])
    json.dumps(spec)  # must not raise


def test_severity_orders_each_family_from_mild_to_severe():
    spec = {entry["name"]: entry["severity"] for entry in describe_eval_transforms()}
    assert spec["jpeg_90"] < spec["jpeg_70"] < spec["jpeg_50"] < spec["jpeg_30"]
    assert spec["blur_0.5"] < spec["blur_1.0"] < spec["blur_2.0"]
    assert spec["noise_0.02"] < spec["noise_0.05"] < spec["noise_0.10"]
    assert spec["resize_0.5"] < spec["resize_0.25"]


def test_build_eval_suite_normalises_alias_keys():
    suite = build_eval_suite(["jpeg_q30", "crop_0.8"])
    assert set(suite) == {"jpeg_30", "crop_0.80"}


def test_build_eval_suite_defaults_to_the_whole_benchmark():
    assert list(build_eval_suite()) == list_eval_transforms()
