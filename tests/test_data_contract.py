"""The canonical batch schemas, including real DataLoader collation."""

from __future__ import annotations

import pytest

from src.data import (
    MODE_PAIRED,
    MODE_STANDARD,
    ManifestDataset,
    PairedViewDataset,
    SchemaError,
    all_keys,
    build_preprocess,
    describe_contract,
    optional_keys,
    required_keys,
    validate_batch,
    validate_sample,
)
from src.data.synthetic import make_synthetic_dataset

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("contract"), n_per_class=8)


@pytest.fixture(scope="module")
def preprocess():
    return build_preprocess("ijepa", image_size=32)


# -- the documented key sets ----------------------------------------------
def test_standard_schema_keys():
    assert required_keys(MODE_STANDARD) == ("image", "label", "source_id", "image_path")
    assert set(optional_keys(MODE_STANDARD)) >= {"dataset", "generator"}


def test_paired_schema_keys():
    assert required_keys(MODE_PAIRED) == (
        "clean", "augmented", "label", "source_id", "image_path",
    )
    assert "transform_name" in optional_keys(MODE_PAIRED)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        required_keys("triplet")


def test_describe_contract_documents_the_image_boundary():
    contract = describe_contract()
    assert contract["raw_image"]["range"] == [0, 255]
    assert contract["raw_image"]["type"] == "PIL.Image.Image"
    assert "model" in contract["preprocessed_image"]["owner"]
    assert contract["label_domain"] == {0: "real", 1: "aigc"}


# -- standard mode ---------------------------------------------------------
def test_standard_sample_matches_the_contract(bundle, preprocess):
    dataset = ManifestDataset(bundle.train, preprocess=preprocess)
    sample = dataset[0]
    validate_sample(sample, mode=MODE_STANDARD, require_tensor_images=True)
    assert set(required_keys(MODE_STANDARD)).issubset(sample)
    assert sample["image"].shape == (3, 32, 32)
    assert sample["label"] in (0, 1)
    assert isinstance(sample["source_id"], str) and sample["source_id"]
    assert isinstance(sample["image_path"], str)
    assert sample["dataset"] == "synthetic"


def test_standard_sample_declares_no_undocumented_keys(bundle, preprocess):
    sample = ManifestDataset(bundle.train, preprocess=preprocess)[0]
    assert not set(sample) - set(all_keys(MODE_STANDARD))


def test_dataset_self_check_helper(bundle, preprocess):
    ManifestDataset(bundle.train, preprocess=preprocess).validate_schema(
        require_tensor_images=True
    )
    PairedViewDataset(bundle.train, preprocess=preprocess).validate_schema(
        require_tensor_images=True
    )


def test_standard_batch_collates_to_the_contract(bundle, preprocess):
    dataset = ManifestDataset(bundle.train, preprocess=preprocess)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    validate_batch(batch, mode=MODE_STANDARD, batch_size=4)
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].shape == (4,)
    assert batch["label"].dtype in (torch.int64, torch.int32)
    # Strings survive collation as lists, not tensors.
    assert isinstance(batch["source_id"], list) and len(batch["source_id"]) == 4
    assert isinstance(batch["image_path"], list)


# -- paired mode -----------------------------------------------------------
def test_paired_sample_matches_the_contract(bundle, preprocess):
    dataset = PairedViewDataset(bundle.train, preprocess=preprocess)
    sample = dataset[0]
    validate_sample(sample, mode=MODE_PAIRED, require_tensor_images=True)
    assert sample["clean"].shape == sample["augmented"].shape == (3, 32, 32)
    assert isinstance(sample["transform_name"], str)


def test_paired_batch_collates_to_the_contract(bundle, preprocess):
    dataset = PairedViewDataset(bundle.train, preprocess=preprocess)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    validate_batch(batch, mode=MODE_PAIRED, batch_size=4)
    assert batch["clean"].shape == batch["augmented"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)
    assert len(batch["transform_name"]) == 4


def test_paired_batch_views_are_shape_aligned(bundle, preprocess):
    """clean/augmented must stack -- the property the training loss relies on."""
    batch = next(iter(DataLoader(PairedViewDataset(bundle.train, preprocess=preprocess), batch_size=4)))
    stacked = torch.cat([batch["clean"], batch["augmented"]], dim=0)
    assert stacked.shape == (8, 3, 32, 32)


def test_dataloader_with_workers_preserves_the_schema(bundle, preprocess):
    """Worker processes must not change keys or drop metadata."""
    loader = DataLoader(
        PairedViewDataset(bundle.train, preprocess=preprocess),
        batch_size=2, num_workers=2,
    )
    for batch in loader:
        validate_batch(batch, mode=MODE_PAIRED, batch_size=2)


# -- validator behaviour ---------------------------------------------------
def test_missing_required_key_is_loud():
    with pytest.raises(SchemaError, match="missing required key"):
        validate_sample({"image": 1, "label": 0, "source_id": "a"}, mode=MODE_STANDARD)


def test_undocumented_key_is_loud():
    sample = {"image": 1, "label": 0, "source_id": "a", "image_path": "p", "oops": 1}
    with pytest.raises(SchemaError, match="undocumented key"):
        validate_sample(sample, mode=MODE_STANDARD)
    validate_sample(sample, mode=MODE_STANDARD, allow_extra_keys=True)


def test_non_binary_label_in_a_batch_is_loud():
    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "label": torch.tensor([0, 5]),
        "source_id": ["a", "b"],
        "image_path": ["p", "q"],
    }
    with pytest.raises(SchemaError, match="outside the binary domain"):
        validate_batch(batch, mode=MODE_STANDARD)


def test_inconsistent_batch_dimension_is_loud():
    batch = {
        "image": torch.zeros(4, 3, 4, 4),
        "label": torch.tensor([0, 1]),
        "source_id": ["a", "b", "c", "d"],
        "image_path": ["p", "q", "r", "s"],
    }
    with pytest.raises(SchemaError, match="inconsistent batch dimension"):
        validate_batch(batch, mode=MODE_STANDARD)


def test_mismatched_paired_shapes_are_loud():
    batch = {
        "clean": torch.zeros(2, 3, 32, 32),
        "augmented": torch.zeros(2, 3, 16, 16),
        "label": torch.tensor([0, 1]),
        "source_id": ["a", "b"],
        "image_path": ["p", "q"],
    }
    with pytest.raises(SchemaError, match="identical shapes"):
        validate_batch(batch, mode=MODE_PAIRED)


def test_require_tensor_images_points_at_preprocess(bundle):
    """Raw (PIL) datasets fail the tensor check with an actionable message."""
    sample = ManifestDataset(bundle.train)[0]
    validate_sample(sample, mode=MODE_STANDARD)  # PIL is legal
    with pytest.raises(SchemaError, match="preprocess"):
        validate_sample(sample, mode=MODE_STANDARD, require_tensor_images=True)


def test_wrong_batch_size_is_loud():
    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "label": torch.tensor([0, 1]),
        "source_id": ["a", "b"],
        "image_path": ["p", "q"],
    }
    with pytest.raises(SchemaError, match="expected batch size"):
        validate_batch(batch, mode=MODE_STANDARD, batch_size=8)
