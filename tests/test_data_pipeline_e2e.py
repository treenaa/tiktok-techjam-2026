"""End-to-end: folders -> manifest -> leakage-safe split -> paired training views.

Exercises the whole subsystem the way a training script would use it, on a
synthetic dataset that deliberately contains on-disk transformed derivatives.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    ManifestDataset,
    PairedViewDataset,
    assign_splits,
    build_eval_datasets,
    build_preprocess,
    check_split_integrity,
    from_folder,
    read_manifest,
    split_records,
    write_manifest,
)
from src.data.transforms import RandomCompetitionTransform
from test_data_fixtures import build_derivative_tree

torch = pytest.importorskip("torch")


def test_full_pipeline(tmp_path):
    root = build_derivative_tree(tmp_path / "ds", n_originals=20)

    # 1. adapt a folder dataset -> records (4 on-disk views per original)
    records = from_folder(root, dataset="demo")
    assert len(records) == 80
    assert len({r.source_id for r in records}) == 20

    # 2. persist a manifest and read it back
    manifest = str(tmp_path / "demo.csv")
    write_manifest(records, manifest, relative_to=root)
    records = read_manifest(manifest, root=root, check_paths_exist=True)

    # 3. leakage-safe split
    splits = split_records(records, ratios=(0.6, 0.2, 0.2), seed=123)
    report = check_split_integrity(splits, original=records)
    assert [s["n_groups"] for s in report["splits"].values()] == [12, 4, 4]

    # every view of an original stays on one side
    placement = {}
    for name, members in splits.items():
        for rec in members:
            assert placement.setdefault(rec.source_id, name) == name

    # 4. a single manifest carrying its own split column round-trips
    assigned = assign_splits(records, ratios=(0.6, 0.2, 0.2), seed=123)
    split_manifest = str(tmp_path / "demo_split.csv")
    write_manifest(assigned, split_manifest, relative_to=root)
    train_records = read_manifest(split_manifest, root=root, split="train")
    assert {r.image_path for r in train_records} == {r.image_path for r in splits["train"]}

    # 5. paired training views, with model preprocessing supplied externally
    preprocess = build_preprocess("ijepa", image_size=64)
    train = PairedViewDataset(
        splits["train"],
        augment=RandomCompetitionTransform(seed=0),
        preprocess=preprocess,
    )
    sample = train[0]
    assert sample["clean"].shape == sample["augmented"].shape == (3, 64, 64)
    assert sample["label"] == splits["train"][0].label
    assert sample["source_id"] == splits["train"][0].source_id

    # 6. clean single-view validation set
    val = ManifestDataset(splits["val"], preprocess=preprocess)
    assert val[0]["image"].shape == (3, 64, 64)

    # 7. the robustness evaluation grid over the held-out split
    grid = build_eval_datasets(splits["test"], preprocess=preprocess)
    assert len(grid) == 20
    for name, dataset in grid.items():
        assert len(dataset) == len(splits["test"]), name
        assert dataset[0]["label"] == splits["test"][0].label, name
    clean = grid["clean"][0]["image"]
    assert not torch.allclose(clean, grid["jpeg_30"][0]["image"])


def test_training_epoch_smoke(tmp_path):
    """Two passes over a DataLoader: labels stable, augmentations resampled."""
    from torch.utils.data import DataLoader

    root = build_derivative_tree(tmp_path / "ds", n_originals=8)
    records = from_folder(root, dataset="demo")
    train = PairedViewDataset(
        records,
        augment=RandomCompetitionTransform(seed=0),
        preprocess=build_preprocess("ijepa", image_size=32),
    )
    loader = DataLoader(train, batch_size=8, num_workers=0)

    epochs = []
    for _ in range(2):
        batches = [
            (b["label"].tolist(), list(b["source_id"]), b["augmented"].clone()) for b in loader
        ]
        epochs.append(batches)

    labels_a = [b[0] for b in epochs[0]]
    labels_b = [b[0] for b in epochs[1]]
    assert labels_a == labels_b, "labels must be stable across epochs"
    assert [b[1] for b in epochs[0]] == [b[1] for b in epochs[1]]
    assert not torch.allclose(epochs[0][0][2], epochs[1][0][2]), (
        "augmentations must be resampled each epoch"
    )
