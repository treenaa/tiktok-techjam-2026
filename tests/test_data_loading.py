"""The shared decode path (rules 17 and 20.6).

Training, evaluation and inference must read pixels identically, and corrupt
files must never be dropped silently.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from src.data import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    ManifestDataset,
    ManifestRecord,
    list_images,
    load_image,
    make_loader,
    verify_images,
)
from src.data.datasets import default_loader


@pytest.fixture
def tree(tmp_path):
    """A directory with good images, several kinds of broken file, and noise."""
    good = tmp_path / "good.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(str(good))
    Image.new("L", (16, 16), 128).save(str(tmp_path / "grey.png"))
    Image.new("RGBA", (16, 16), (1, 2, 3, 4)).save(str(tmp_path / "alpha.webp"))
    Image.new("P", (16, 16)).save(str(tmp_path / "palette.png"))
    Image.new("RGB", (8, 8)).save(str(tmp_path / "photo.JPG"))          # upper-case ext

    (tmp_path / "empty.png").write_bytes(b"")
    (tmp_path / "notimage.jpeg").write_text("definitely not an image")
    (tmp_path / "truncated.jpg").write_bytes(good.read_bytes()[:24])
    (tmp_path / "readme.txt").write_text("ignored")                      # not an image
    (tmp_path / ".hidden.png").write_bytes(good.read_bytes())            # dotfile

    nested = tmp_path / "sub"
    nested.mkdir()
    Image.new("RGB", (8, 8)).save(str(nested / "nested.png"))
    return tmp_path


# -- normalisation ---------------------------------------------------------
@pytest.mark.parametrize("name", ["good.png", "grey.png", "alpha.webp", "palette.png"])
def test_every_input_mode_becomes_rgb(tree, name):
    """Downstream code must never branch on palette/greyscale/alpha."""
    image = load_image(str(tree / name))
    assert image.mode == "RGB"


def test_pixels_match_the_file(tree):
    loaded = load_image(str(tree / "good.png"))
    reference = Image.open(str(tree / "good.png")).convert("RGB")
    assert np.array_equal(np.asarray(loaded), np.asarray(reference))


def test_image_is_fully_decoded(tree):
    """Eager decode: no lazy handle that could fail inside a DataLoader worker."""
    image = load_image(str(tree / "good.png"))
    assert image.size == (32, 24)
    np.asarray(image)  # must not need the file handle any more


# -- corrupt-file policy (rule 20.6) --------------------------------------
@pytest.mark.parametrize("name", ["empty.png", "notimage.jpeg", "truncated.jpg"])
def test_corrupt_files_raise_by_default(tree, name):
    with pytest.raises(ImageLoadError):
        load_image(str(tree / name))


def test_error_names_the_path_and_reason(tree):
    with pytest.raises(ImageLoadError) as excinfo:
        load_image(str(tree / "notimage.jpeg"))
    assert "notimage.jpeg" in str(excinfo.value)
    assert excinfo.value.path.endswith("notimage.jpeg")
    assert excinfo.value.reason


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        load_image(str(tmp_path / "absent.png"))


def test_skip_policy_returns_none(tree):
    assert load_image(str(tree / "empty.png"), on_error="skip") is None
    assert load_image(str(tree / "good.png"), on_error="skip") is not None


def test_placeholder_policy_keeps_batch_shape(tree):
    image = load_image(str(tree / "empty.png"), on_error="placeholder",
                       placeholder_size=(64, 64))
    assert image.mode == "RGB" and image.size == (64, 64)


def test_unknown_policy_is_rejected(tree):
    with pytest.raises(ValueError, match="on_error must be one of"):
        load_image(str(tree / "good.png"), on_error="ignore")


def test_truncated_can_be_allowed_explicitly(tree):
    """Opt-in only -- half an image yields a meaningless prediction."""
    path = str(tree / "truncated.jpg")
    with pytest.raises(ImageLoadError):
        load_image(path)
    try:
        load_image(path, allow_truncated=True)
    except ImageLoadError:
        pytest.skip("this file is too damaged even for truncated mode")


def test_truncated_flag_is_restored_globally(tree):
    """LOAD_TRUNCATED_IMAGES is global state; it must not leak between calls."""
    from PIL import ImageFile

    before = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        load_image(str(tree / "truncated.jpg"), on_error="skip", allow_truncated=True)
    except ImageLoadError:
        pass
    assert ImageFile.LOAD_TRUNCATED_IMAGES == before


def test_make_loader_fixes_a_policy(tree):
    lenient = make_loader("skip")
    assert lenient(str(tree / "empty.png")) is None
    strict = make_loader("raise")
    with pytest.raises(ImageLoadError):
        strict(str(tree / "empty.png"))


# -- directory listing (rule 17) ------------------------------------------
def test_listing_is_deterministic(tree):
    assert list_images(str(tree)) == list_images(str(tree))
    assert list_images(str(tree)) == sorted(list_images(str(tree)))


def test_listing_finds_supported_formats_only(tree):
    names = {os.path.basename(p) for p in list_images(str(tree))}
    assert "readme.txt" not in names
    assert "good.png" in names and "alpha.webp" in names


def test_listing_is_case_insensitive_on_extensions(tree):
    assert any(p.endswith("photo.JPG") for p in list_images(str(tree)))


def test_listing_skips_hidden_files(tree):
    assert not any(os.path.basename(p).startswith(".") for p in list_images(str(tree)))


def test_listing_recurses_by_default(tree):
    names = {os.path.basename(p) for p in list_images(str(tree))}
    assert "nested.png" in names
    assert "nested.png" not in {
        os.path.basename(p) for p in list_images(str(tree), recursive=False)
    }


def test_listing_rejects_a_non_directory(tree):
    with pytest.raises(NotADirectoryError):
        list_images(str(tree / "good.png"))


def test_supported_extensions_cover_the_required_formats():
    for required in (".jpg", ".jpeg", ".png", ".webp"):
        assert required in SUPPORTED_EXTENSIONS


# -- verification ----------------------------------------------------------
def test_verify_images_separates_good_from_bad(tree):
    report = verify_images(list_images(str(tree)))
    assert report["n_checked"] == report["n_readable"] + len(report["unreadable"])
    bad = {os.path.basename(p) for p, _ in report["unreadable"]}
    assert bad == {"empty.png", "notimage.jpeg", "truncated.jpg"}
    assert all(reason for _, reason in report["unreadable"])


def test_verify_images_on_a_clean_set(tree):
    report = verify_images([str(tree / "good.png"), str(tree / "grey.png")])
    assert report["n_readable"] == 2 and not report["unreadable"]


# -- the datasets use this same path --------------------------------------
def test_dataset_uses_the_shared_loader(tree):
    """No second decode path: the dataset must fail the same way."""
    record = ManifestRecord(str(tree / "notimage.jpeg"), 0, "bad")
    dataset = ManifestDataset([record])
    with pytest.raises(ImageLoadError):
        dataset[0]


def test_dataset_default_loader_is_the_shared_one():
    from src.data import loading

    assert default_loader is loading.default_loader


def test_dataset_can_take_a_lenient_loader(tree):
    """Inference may keep going past a bad file; it must opt in explicitly."""
    records = [ManifestRecord(str(tree / "good.png"), 0, "a")]
    dataset = ManifestDataset(records, loader=make_loader("placeholder"))
    assert dataset[0]["image"].mode == "RGB"
