from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from src.data import ManifestDataset, ManifestRecord, PairedViewDataset, build_preprocess
from src.training import (
    TrainingConfig,
    build_datasets,
    build_loaders,
    capture_loader_state,
    restore_loader_state,
    seed_training_worker,
)

torch = pytest.importorskip("torch")


def records(tmp_path, n=8):
    output = []
    for index in range(n):
        path = tmp_path / ("image-%d.png" % index)
        Image.new("RGB", (20, 20), (index * 20, index * 20, index * 20)).save(path)
        output.append(
            ManifestRecord(
                str(path),
                index % 2,
                "source-%d" % index,
                dataset="tiny",
                generator="toy" if index % 2 else "",
            )
        )
    return output


def test_phase_one_uses_one_view_while_robust_phase_uses_pairs(tmp_path):
    items = records(tmp_path)
    preprocess = build_preprocess("none", image_size=16)
    clean, validation = build_datasets(
        items[:6], items[6:], preprocess=preprocess, config=TrainingConfig(augment="none"), seed=4
    )
    assert isinstance(clean, ManifestDataset)
    assert "image" in clean[0] and "clean" not in clean[0]
    paired, _ = build_datasets(
        items[:6],
        items[6:],
        preprocess=preprocess,
        config=TrainingConfig(augment="competition"),
        seed=4,
    )
    assert isinstance(paired, PairedViewDataset)
    assert paired[0]["clean"].shape == paired[0]["augmented"].shape


def test_loader_state_round_trips_shuffle_and_augmentation_rng(tmp_path):
    items = records(tmp_path)
    config = TrainingConfig(batch_size=2, augment="competition")
    train, val = build_datasets(
        items[:6], items[6:], preprocess=build_preprocess("none", 8), config=config, seed=9
    )
    loader, _ = build_loaders(train, val, config, seed=9)
    state = capture_loader_state(loader)
    expected_transform = train.augment.sample().name
    restore_loader_state(loader, state)
    assert train.augment.sample().name == expected_transform
    assert torch.equal(loader.generator.get_state(), state["generator"])


def test_worker_initializer_reseeds_private_augmentation(monkeypatch):
    class Augment:
        seed = None

        def set_seed(self, value):
            self.seed = value

    dataset = SimpleNamespace(augment=Augment())
    monkeypatch.setattr("src.training.loaders.seed_worker", lambda worker_id: None)
    monkeypatch.setattr("src.training.loaders.get_worker_info", lambda: SimpleNamespace(dataset=dataset))
    monkeypatch.setattr(torch, "initial_seed", lambda: 123456)
    seed_training_worker(2)
    assert dataset.augment.seed == 123456


def test_validation_loader_is_not_shuffled(tmp_path):
    items = records(tmp_path)
    config = TrainingConfig(batch_size=2)
    train, val = build_datasets(
        items[:6], items[6:], preprocess=build_preprocess("none", 8), config=config, seed=1
    )
    _, val_loader = build_loaders(train, val, config, seed=1)
    observed = []
    for batch in val_loader:
        observed.extend(batch["source_id"])
    assert observed == [record.source_id for record in items[6:]]
