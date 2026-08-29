"""Where the data layer stops and model-specific preprocessing begins.

The raw dataset must stay model-agnostic: no I-JEPA, CLIP or DINO statistics
may be baked in.  These tests pin that boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data import (
    NORMALIZATION_PRESETS,
    ImagePreprocessing,
    ManifestDataset,
    PairedViewDataset,
    build_preprocess,
    describe_contract,
)
from src.data.synthetic import make_synthetic_dataset

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("bound"), n_per_class=6)


# -- what leaves the raw dataset ------------------------------------------
def test_raw_dataset_yields_pil_rgb_uint8(bundle):
    """No preprocess -> PIL RGB in [0, 255].  Documented in DATA_CONTRACT.md."""
    sample = ManifestDataset(bundle.train)[0]
    image = sample["image"]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    array = np.asarray(image)
    assert array.dtype == np.uint8
    assert 0 <= array.min() and array.max() <= 255


def test_raw_dataset_applies_no_normalization(bundle):
    """Pixels must match the file on disk exactly -- nothing scaled or shifted."""
    dataset = ManifestDataset(bundle.train)
    sample = dataset[0]
    on_disk = Image.open(sample["image_path"]).convert("RGB")
    assert np.array_equal(np.asarray(sample["image"]), np.asarray(on_disk))


def test_raw_dataset_does_not_resize(bundle):
    """Native resolution is preserved; resizing is the model's decision."""
    sample = ManifestDataset(bundle.train)[0]
    assert sample["image"].size == Image.open(sample["image_path"]).size


def test_paired_dataset_is_equally_raw_by_default(bundle):
    sample = PairedViewDataset(bundle.train)[0]
    assert isinstance(sample["clean"], Image.Image)
    assert isinstance(sample["augmented"], Image.Image)


def test_contract_documents_the_boundary():
    contract = describe_contract()
    assert contract["raw_image"] == {
        "type": "PIL.Image.Image", "mode": "RGB", "dtype": "uint8", "range": [0, 255],
    }
    assert "model" in contract["preprocessed_image"]["owner"]


# -- preprocessing is injected, never assumed -----------------------------
def test_preprocessing_is_supplied_externally(bundle):
    """Any PIL -> Any callable works; the dataset does not care what it is."""
    dataset = ManifestDataset(bundle.train, preprocess=lambda img: "sentinel")
    assert dataset[0]["image"] == "sentinel"


def test_a_torchvision_style_callable_slots_in(bundle):
    def fake_processor(img):
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0

    sample = ManifestDataset(bundle.train, preprocess=fake_processor)[0]
    assert sample["image"].shape[0] == 3 and float(sample["image"].max()) <= 1.0


@pytest.mark.parametrize("preset", ["imagenet", "ijepa", "clip", "half", "none"])
def test_all_presets_are_available_but_none_is_the_default(preset, bundle):
    """Presets exist for convenience; the dataset defaults to no preprocessing."""
    assert preset in NORMALIZATION_PRESETS
    assert ManifestDataset(bundle.train).preprocess is None


def test_ijepa_and_clip_statistics_differ():
    """Proof that the choice matters -- and so must not be hard-coded upstream."""
    assert NORMALIZATION_PRESETS["ijepa"] != NORMALIZATION_PRESETS["clip"]
    assert NORMALIZATION_PRESETS["ijepa"] == NORMALIZATION_PRESETS["imagenet"]


def test_none_preset_skips_normalization():
    pre = ImagePreprocessing(image_size=8, normalization="none")
    tensor = pre(Image.new("RGB", (16, 16), (255, 255, 255)))
    assert float(tensor.max()) == pytest.approx(1.0)
    assert float(tensor.min()) == pytest.approx(1.0)


def test_normalization_actually_applies_the_statistics():
    mean, std = NORMALIZATION_PRESETS["imagenet"]
    pre = ImagePreprocessing(image_size=8, normalization="imagenet")
    tensor = pre(Image.new("RGB", (16, 16), (0, 0, 0)))
    for channel in range(3):
        assert float(tensor[channel].mean()) == pytest.approx(-mean[channel] / std[channel], abs=1e-5)


def test_preprocessing_output_is_float32_chw():
    pre = build_preprocess("ijepa", image_size=16)
    tensor = pre(Image.new("RGB", (40, 30)))
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 16, 16)


def test_build_preprocess_selects_clip_statistics_for_clip_models():
    assert build_preprocess("clip_vit_b16").normalization == NORMALIZATION_PRESETS["clip"]
    assert build_preprocess("ijepa_vith14").normalization == NORMALIZATION_PRESETS["imagenet"]
    assert build_preprocess("some_unknown_backbone").normalization == NORMALIZATION_PRESETS["imagenet"]


def test_to_tensor_false_keeps_pil_for_downstream_processors():
    """Lets a HuggingFace image_processor take over after resizing."""
    pre = ImagePreprocessing(image_size=16, normalization=None, to_tensor=False)
    assert isinstance(pre(Image.new("RGB", (40, 30))), Image.Image)


def test_normalization_without_tensors_is_rejected():
    with pytest.raises(ValueError, match="requires to_tensor"):
        ImagePreprocessing(normalization="imagenet", to_tensor=False)


def test_unknown_preset_is_rejected():
    with pytest.raises(KeyError, match="unknown normalization preset"):
        ImagePreprocessing(normalization="not_a_backbone")


def test_zero_std_is_rejected():
    with pytest.raises(ValueError, match="std must be"):
        ImagePreprocessing(normalization=((0.5, 0.5, 0.5), (0.0, 1.0, 1.0)))(
            Image.new("RGB", (8, 8))
        )


# -- ordering: corruption happens before model preprocessing --------------
def test_transforms_run_at_native_resolution_before_preprocessing(bundle):
    """A corruption must not be applied to an already-resized image.

    Blurring at 224px then downscaling is a different operation from blurring
    at native resolution, so the pipeline order is part of the contract.
    """
    seen = []

    def spy_preprocess(img):
        seen.append(img.size)
        return img

    native = Image.open(bundle.train[0].image_path).size
    ManifestDataset(bundle.train, transform="blur_2.0", preprocess=spy_preprocess)[0]
    assert seen == [native]


def test_preprocessing_can_differ_between_paired_views(bundle):
    dataset = PairedViewDataset(
        bundle.train,
        preprocess=lambda img: "clean_pre",
        augmented_preprocess=lambda img: "aug_pre",
    )
    sample = dataset[0]
    assert sample["clean"] == "clean_pre" and sample["augmented"] == "aug_pre"
