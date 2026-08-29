"""``source_id`` stability -- the property the whole leakage story rests on."""

from __future__ import annotations

import pytest

from src.data import canonical_source_id, make_source_id_fn, strip_transform_suffixes
from src.data.source_id import find_source_id_collisions, group_paths_by_source_id
from src.data.transforms import EVAL_TRANSFORM_NAMES


@pytest.mark.parametrize(
    "derivative",
    [
        "cat_017.png",
        "cat_017_jpeg70.png",
        "cat_017_jpeg_30.png",
        "cat_017_blur_1.0.png",
        "cat_017_blur_2.0.png",
        "cat_017_resize_0.25.png",
        "cat_017_noise_0.05.png",
        "cat_017_crop_0.8.png",
        "cat_017_jitter_brightness_up.png",
        "cat_017_clean.png",
        "cat_017_aug3.png",
        "cat_017.jpg",
    ],
)
def test_transform_suffixes_do_not_change_source_id(derivative):
    fn = make_source_id_fn("stem")
    assert fn("/data/real/" + derivative) == fn("/data/real/cat_017.png") == "cat_017"


def test_chained_transform_suffixes_are_stripped():
    fn = make_source_id_fn("stem")
    assert fn("a/cat_017_blur_1.0_jpeg30.png") == "cat_017"
    assert fn("a/cat_017_noise_0.05_resize_0.5_jpeg50.png") == "cat_017"


def test_every_registry_transform_name_is_strippable():
    """A file named after any eval transform still maps to its original id."""
    fn = make_source_id_fn("stem")
    for name in EVAL_TRANSFORM_NAMES:
        assert fn("d/img042_%s.png" % name) == "img042", name


def test_stripping_never_yields_an_empty_id():
    assert strip_transform_suffixes("clean") == "clean"
    assert strip_transform_suffixes("jpeg90") == "jpeg90"
    assert canonical_source_id("/d/clean.png") == "clean"


def test_unrelated_images_keep_distinct_ids():
    fn = make_source_id_fn("stem")
    ids = {fn(p) for p in ["a/cat_017.png", "a/cat_018.png", "a/dog_017.png"]}
    assert len(ids) == 3


def test_relpath_policy_separates_identical_filenames():
    """CIFAKE reuses ``0001.png`` in REAL and FAKE -- they are different images."""
    fn = make_source_id_fn("relpath", root="/data/cifake")
    real = fn("/data/cifake/train/REAL/0001.png")
    fake = fn("/data/cifake/train/FAKE/0001.png")
    assert real != fake
    assert real == "train/REAL/0001"
    # ...while derivatives inside one folder still collapse.
    assert fn("/data/cifake/train/REAL/0001_jpeg50.png") == real


def test_prefix_namespaces_ids_across_datasets():
    a = make_source_id_fn("stem", prefix="cifake")("x/img.png")
    b = make_source_id_fn("stem", prefix="wildfake")("y/img.png")
    assert a == "cifake:img" and b == "wildfake:img" and a != b


def test_parent_policy_groups_per_image_folders():
    fn = make_source_id_fn("parent")
    assert fn("/d/img042/clean.png") == fn("/d/img042/jpeg30.png") == "img042"


def test_regex_policy():
    fn = make_source_id_fn("regex", regex=r"(img\d+)")
    assert fn("/d/x/img0042_jpeg50.png") == "img0042"
    with pytest.raises(ValueError, match="did not match"):
        fn("/d/x/nothing.png")


def test_regex_policy_requires_a_pattern():
    with pytest.raises(ValueError, match="requires a regex"):
        make_source_id_fn("regex")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown source_id policy"):
        make_source_id_fn("magic")


def test_strip_can_be_disabled():
    fn = make_source_id_fn("stem", strip_suffixes=False)
    assert fn("a/cat_jpeg70.png") == "cat_jpeg70"


def test_custom_tokens():
    fn = make_source_id_fn("stem", tokens=("view\\d+",))
    assert fn("a/cat_view2.png") == "cat"
    assert fn("a/cat_jpeg70.png") == "cat_jpeg70"  # default tokens not applied


def test_grouping_helpers_report_cross_directory_collisions():
    paths = ["a/img.png", "b/img.png", "a/img_jpeg70.png"]
    fn = make_source_id_fn("stem")
    groups = group_paths_by_source_id(paths, fn)
    assert len(groups["img"]) == 3
    collisions = find_source_id_collisions(paths, fn)
    assert collisions and collisions[0][0] == "img"
    # The relpath policy resolves the ambiguity.
    assert not find_source_id_collisions(paths, make_source_id_fn("relpath"))
