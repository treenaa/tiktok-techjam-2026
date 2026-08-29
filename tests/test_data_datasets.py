"""Dataset objects: sample contract, paired views, model-aware preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data import (
    ImagePreprocessing,
    ManifestDataset,
    ManifestRecord,
    PairedViewDataset,
    TransformedEvalDataset,
    build_eval_datasets,
    build_preprocess,
    default_loader,
    from_folder,
    get_transform,
)
from src.data.transforms import EVAL_TRANSFORM_NAMES, GaussianBlur, RandomCompetitionTransform
from test_data_fixtures import build_derivative_tree, write_image

torch = pytest.importorskip("torch")


@pytest.fixture
def records(tmp_path):
    return from_folder(build_derivative_tree(tmp_path / "ds", n_originals=6), dataset="demo")


# -- the sample contract ---------------------------------------------------
def test_sample_exposes_the_required_fields(records):
    sample = ManifestDataset(records)[0]
    for key in ("image", "label", "source_id", "image_path"):
        assert key in sample, key
    assert isinstance(sample["image"], Image.Image), "raw dataset yields PIL by default"
    assert sample["label"] in (0, 1)
    assert isinstance(sample["source_id"], str) and sample["source_id"]


def test_sample_carries_optional_metadata(records):
    sample = ManifestDataset(records)[0]
    assert sample["dataset"] == "demo"
    assert "generator" in sample and "index" in sample and "transform" in sample


def test_dataset_length_and_introspection(records):
    dataset = ManifestDataset(records)
    assert len(dataset) == len(records) == 24
    assert dataset.label_counts() == {0: 12, 1: 12}
    assert len(set(dataset.source_ids)) == 6
    assert "ManifestDataset(n=24" in repr(dataset)


def test_labels_and_source_ids_match_the_records(records):
    dataset = ManifestDataset(records)
    for i in range(len(dataset)):
        assert dataset[i]["label"] == records[i].label
        assert dataset[i]["source_id"] == records[i].source_id


def test_relative_paths_resolve_against_root(tmp_path):
    root = build_derivative_tree(tmp_path / "ds", n_originals=2)
    absolute = from_folder(root, dataset="demo")
    import os

    relative = [r.with_fields(image_path=os.path.relpath(r.image_path, root)) for r in absolute]
    sample = ManifestDataset(relative, root=root, check_paths_exist=True)[0]
    assert isinstance(sample["image"], Image.Image)


def test_subset_preserves_settings(records):
    dataset = ManifestDataset(records, transform="jpeg_q30")
    subset = dataset.subset([0, 1, 2])
    assert len(subset) == 3 and subset.transform.name == "jpeg_q30"


# -- transforms inside the dataset ----------------------------------------
def test_transform_can_be_a_name_or_a_callable(records):
    by_name = ManifestDataset(records, transform="blur_sigma2.0")[0]["image"]
    by_object = ManifestDataset(records, transform=GaussianBlur(2.0, decimals=1))[0]["image"]
    assert np.array_equal(np.asarray(by_name), np.asarray(by_object))
    assert ManifestDataset(records, transform="blur_sigma2.0")[0]["transform"] == "blur_sigma2.0"


def test_transform_changes_pixels_but_not_the_label(records):
    clean = ManifestDataset(records)[0]
    blurred = ManifestDataset(records, transform="blur_sigma2.0")[0]
    assert not np.array_equal(np.asarray(clean["image"]), np.asarray(blurred["image"]))
    assert clean["label"] == blurred["label"]
    assert clean["source_id"] == blurred["source_id"]


def test_unknown_transform_name_raises(records):
    with pytest.raises(KeyError, match="unknown transform"):
        ManifestDataset(records, transform="jpeg_q42")


def test_non_callable_transform_is_rejected(records):
    with pytest.raises(TypeError, match="callable"):
        ManifestDataset(records, transform=123)


# -- model-aware preprocessing --------------------------------------------
def test_dataset_is_model_agnostic_by_default(records):
    """No normalisation is baked in -- the raw dataset hands back PIL images."""
    assert isinstance(ManifestDataset(records)[0]["image"], Image.Image)


def test_external_preprocessing_produces_normalised_tensors(records):
    preprocess = build_preprocess("ijepa", image_size=224)
    image = ManifestDataset(records, preprocess=preprocess)[0]["image"]
    assert torch.is_tensor(image)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert image.min() < 0, "ImageNet normalisation shifts values below zero"


def test_preprocessing_presets_differ(records):
    dataset = ManifestDataset(records)
    raw = dataset[0]["image"]
    imagenet = ImagePreprocessing(224, normalization="imagenet")(raw)
    clip = ImagePreprocessing(224, normalization="clip")(raw)
    none = ImagePreprocessing(224, normalization="none")(raw)
    assert not torch.allclose(imagenet, clip)
    assert float(none.min()) >= 0.0 and float(none.max()) <= 1.0


def test_preprocessing_can_be_any_callable(records):
    sample = ManifestDataset(records, preprocess=lambda img: img.size)[0]
    assert sample["image"] == (64, 48)


def test_preprocessing_resize_modes():
    image = Image.new("RGB", (100, 50))
    assert ImagePreprocessing(32, resize_mode="squash", to_tensor=False)(image).size == (32, 32)
    assert ImagePreprocessing(32, resize_mode="shortest", to_tensor=False)(image).size == (32, 32)
    assert ImagePreprocessing((16, 48), to_tensor=False)(image).size == (48, 16)
    assert ImagePreprocessing(None, to_tensor=False)(image).size == (100, 50)


def test_preprocessing_rejects_bad_configuration():
    with pytest.raises(KeyError, match="unknown normalization preset"):
        ImagePreprocessing(normalization="mystery")
    with pytest.raises(ValueError, match="resize_mode"):
        ImagePreprocessing(resize_mode="stretchy")
    with pytest.raises(ValueError, match="to_tensor"):
        ImagePreprocessing(normalization="imagenet", to_tensor=False)


def test_batching_with_a_dataloader(records):
    from torch.utils.data import DataLoader

    dataset = ManifestDataset(records, preprocess=build_preprocess("ijepa", image_size=64))
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["label"].shape == (4,)
    assert len(batch["source_id"]) == 4


# -- paired views ----------------------------------------------------------
def test_paired_sample_shape_of_the_api(records):
    sample = PairedViewDataset(records, augment="jpeg_q30")[0]
    assert {"clean", "augmented", "label", "source_id"} <= set(sample)
    assert isinstance(sample["clean"], Image.Image)
    assert isinstance(sample["augmented"], Image.Image)


def test_paired_views_share_label_and_source_id(records):
    dataset = PairedViewDataset(records, augment=RandomCompetitionTransform(seed=0))
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["label"] == records[i].label
        assert sample["source_id"] == records[i].source_id
        assert sample["image_path"] == records[i].image_path


def test_paired_views_come_from_the_same_image_and_differ(records):
    sample = PairedViewDataset(records, augment="blur_sigma2.0")[0]
    clean, augmented = np.asarray(sample["clean"]), np.asarray(sample["augmented"])
    assert clean.shape == augmented.shape
    assert not np.array_equal(clean, augmented)


def test_paired_views_keep_the_clean_branch_clean(records):
    dataset = PairedViewDataset(records, augment="jpeg_q30")
    reference = ManifestDataset(records)[0]["image"]
    assert np.array_equal(np.asarray(dataset[0]["clean"]), np.asarray(reference))


def test_paired_augmentation_is_stochastic_across_accesses(records):
    """Training wants a fresh corruption each epoch, not a fixed one."""
    dataset = PairedViewDataset(records, augment=RandomCompetitionTransform(seed=0))
    names = {dataset[0]["transform"] for _ in range(25)}
    assert len(names) > 1


def test_paired_augmentation_can_be_fixed_for_debugging(records):
    dataset = PairedViewDataset(records, augment="jpeg_q50")
    assert {dataset[0]["transform"] for _ in range(5)} == {"jpeg_q50"}
    first = np.asarray(dataset[0]["augmented"])
    assert np.array_equal(first, np.asarray(dataset[0]["augmented"]))


def test_paired_records_the_applied_transform_name(records):
    dataset = PairedViewDataset(records, augment=RandomCompetitionTransform(seed=3))
    assert isinstance(dataset[0]["transform"], str) and dataset[0]["transform"]


def test_paired_views_with_preprocessing_are_tensors_of_equal_shape(records):
    dataset = PairedViewDataset(
        records,
        augment=RandomCompetitionTransform(seed=1),
        preprocess=build_preprocess("ijepa", image_size=96),
    )
    sample = dataset[0]
    assert sample["clean"].shape == sample["augmented"].shape == (3, 96, 96)


def test_paired_views_batch_through_a_dataloader(records):
    from torch.utils.data import DataLoader

    dataset = PairedViewDataset(
        records,
        augment=RandomCompetitionTransform(seed=2),
        preprocess=build_preprocess("ijepa", image_size=32),
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["clean"].shape == batch["augmented"].shape == (4, 3, 32, 32)
    assert batch["label"].tolist() == [r.label for r in records[:4]]
    assert list(batch["source_id"]) == [r.source_id for r in records[:4]]


def test_paired_defaults_to_the_full_random_suite(records):
    dataset = PairedViewDataset(records)
    assert isinstance(dataset.augment, RandomCompetitionTransform)
    assert dataset[0]["clean"] is not None


def test_paired_augmented_preprocess_can_differ(records):
    dataset = PairedViewDataset(
        records,
        augment="jpeg_q50",
        preprocess=ImagePreprocessing(32, to_tensor=True),
        augmented_preprocess=ImagePreprocessing(16, to_tensor=True),
    )
    sample = dataset[0]
    assert sample["clean"].shape == (3, 32, 32)
    assert sample["augmented"].shape == (3, 16, 16)


# -- evaluation grid -------------------------------------------------------
def test_transformed_eval_dataset(records):
    dataset = TransformedEvalDataset(records, transform_name="jpeg_q30")
    assert dataset.transform_name == "jpeg_q30"
    assert dataset[0]["transform"] == "jpeg_q30"
    assert "jpeg_q30" in repr(dataset)


def test_build_eval_datasets_covers_the_suite_and_aligns_rows(records):
    grid = build_eval_datasets(records)
    assert set(grid) == set(EVAL_TRANSFORM_NAMES)
    for name, dataset in grid.items():
        assert len(dataset) == len(records)
        assert dataset[3]["source_id"] == records[3].source_id, name
        assert dataset[3]["label"] == records[3].label, name


def test_eval_datasets_preserve_dimensions(records):
    for name, dataset in build_eval_datasets(records).items():
        assert dataset[0]["image"].size == (64, 48), name


# -- loader ----------------------------------------------------------------
def test_default_loader_returns_rgb(tmp_path):
    path = write_image(str(tmp_path / "a.png"), 1)
    image = default_loader(path)
    assert image.mode == "RGB"


def test_grayscale_images_are_converted(tmp_path):
    from test_data_fixtures import make_image

    path = str(tmp_path / "gray.png")
    make_image(1).convert("L").save(path)
    assert default_loader(path).mode == "RGB"
    dataset = ManifestDataset([ManifestRecord(path, 0, "gray")])
    assert dataset[0]["image"].mode == "RGB"


def test_empty_dataset_is_rejected():
    with pytest.raises(Exception):
        ManifestDataset([])
