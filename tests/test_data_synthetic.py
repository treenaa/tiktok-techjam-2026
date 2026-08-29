"""The tiny synthetic dataset: the repo's download-free integration fixture."""

from __future__ import annotations

import os

import pytest

from src.data import (
    MODE_PAIRED,
    MODE_STANDARD,
    ManifestDataset,
    PairedViewDataset,
    build_eval_datasets,
    build_preprocess,
    list_generators,
    read_manifest,
    validate_batch,
    validate_splits,
)
from src.data.synthetic import make_synthetic_dataset, make_synthetic_images

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_per_class=12)


# -- shape of the fixture --------------------------------------------------
def test_bundle_exposes_three_populated_splits(bundle):
    assert len(bundle.records) == 24
    assert len(bundle.train) + len(bundle.val) + len(bundle.test) == 24
    for split in (bundle.train, bundle.val, bundle.test):
        assert split and {r.label for r in split} == {0, 1}


def test_bundle_is_small_enough_for_ci(bundle):
    total = sum(
        os.path.getsize(os.path.join(dirpath, name))
        for dirpath, _, names in os.walk(bundle.root)
        for name in names
    )
    assert total < 2 * 1024 * 1024, "fixture should stay well under 2 MB"


def test_bundle_writes_readable_manifests(bundle):
    for path in bundle.manifest_paths.values():
        assert os.path.exists(path)
    records = read_manifest(bundle.train_manifest, check_paths_exist=True)
    assert records and all(r.dataset == "synthetic" for r in records)


def test_bundle_is_reproducible(tmp_path):
    a = make_synthetic_dataset(tmp_path / "a", seed=5, n_per_class=6)
    b = make_synthetic_dataset(tmp_path / "b", seed=5, n_per_class=6)
    assert [os.path.basename(r.image_path) for r in a.train] == [
        os.path.basename(r.image_path) for r in b.train
    ]


def test_different_seeds_give_different_pixels(tmp_path):
    import numpy as np
    from PIL import Image

    a = make_synthetic_dataset(tmp_path / "a", seed=1, n_per_class=4)
    b = make_synthetic_dataset(tmp_path / "b", seed=2, n_per_class=4)
    pa = np.asarray(Image.open(a.records[0].image_path))
    pb = np.asarray(Image.open(b.records[0].image_path))
    assert not np.array_equal(pa, pb)


def test_bundle_carries_generator_metadata(bundle):
    generators = list_generators(bundle.records)
    assert len(generators) == 2
    assert all(r.generator == "" for r in bundle.records if r.label == 0)


def test_bundle_splits_pass_validation(bundle):
    assert validate_splits(
        bundle.train_manifest, bundle.val_manifest, bundle.test_manifest
    ).ok


def test_multi_view_fixture_keeps_derivatives_together(tmp_path):
    bundle = make_synthetic_dataset(tmp_path / "views", n_per_class=8, n_views=3)
    assert len(bundle.records) == 8 * 2 * 3
    seen = {}
    for name, members in bundle.splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name


def test_too_small_a_request_is_refused(tmp_path):
    with pytest.raises(ValueError, match="n_per_class"):
        make_synthetic_images(tmp_path / "tiny", n_per_class=1)


# -- the end-to-end path other owners depend on ---------------------------
def test_dataset_to_dataloader_to_train_to_evaluate(bundle):
    """dataset -> dataloader -> train -> evaluate, with no download.

    Stands in for the real training loop: if this breaks, the data layer has
    broken the model owner's integration.
    """
    preprocess = build_preprocess("ijepa", image_size=32)
    train_loader = DataLoader(
        ManifestDataset(bundle.train, preprocess=preprocess), batch_size=4, shuffle=True
    )
    test_loader = DataLoader(
        ManifestDataset(bundle.test, preprocess=preprocess), batch_size=4
    )

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(12):
        for batch in train_loader:
            validate_batch(batch, mode=MODE_STANDARD)
            optimizer.zero_grad()
            loss = criterion(model(batch["image"]), batch["label"])
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch["image"]).argmax(dim=1)
            correct += int((predictions == batch["label"]).sum())
            total += int(batch["label"].numel())
    accuracy = correct / total
    assert total == len(bundle.test)
    assert accuracy > 0.6, "the two synthetic classes should be learnable (got %.2f)" % accuracy


def test_paired_training_path_runs(bundle):
    """Paired mode must feed a model the same way, with aligned views."""
    preprocess = build_preprocess("ijepa", image_size=32)
    loader = DataLoader(
        PairedViewDataset(bundle.train, preprocess=preprocess), batch_size=4
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2))
    for batch in loader:
        validate_batch(batch, mode=MODE_PAIRED)
        clean_logits = model(batch["clean"])
        augmented_logits = model(batch["augmented"])
        assert clean_logits.shape == augmented_logits.shape
        # Labels apply to both views -- the property a consistency loss needs.
        assert batch["label"].shape[0] == clean_logits.shape[0]


def test_robustness_evaluation_grid_runs(bundle):
    """The evaluation owner's access pattern: one dataset per named transform."""
    preprocess = build_preprocess("ijepa", image_size=32)
    grid = build_eval_datasets(
        bundle.test, transform_names=["clean", "jpeg_30", "blur_2.0", "noise_0.10"],
        preprocess=preprocess,
    )
    assert set(grid) == {"clean", "jpeg_30", "blur_2.0", "noise_0.10"}
    for name, dataset in grid.items():
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        validate_batch(batch, mode=MODE_STANDARD)
        assert batch["image"].shape == (4, 3, 32, 32), name

    # Same records in the same order across the grid -- predictions line up.
    references = [r.image_path for r in grid["clean"].records]
    for dataset in grid.values():
        assert [r.image_path for r in dataset.records] == references
