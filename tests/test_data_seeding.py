"""Rule 20.9: seeding Python, NumPy and PyTorch reproducibly."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.data import dataloader_kwargs, seed_everything, seed_worker
from src.data.seeding import make_generator

torch = pytest.importorskip("torch")


def test_seed_everything_reports_what_it_seeded():
    state = seed_everything(123)
    assert state["seed"] == 123
    assert state["python"] and state["numpy"] and state["torch"]
    assert "cuda" in state and "deterministic" in state


def test_python_random_is_reproducible():
    seed_everything(7)
    first = [random.random() for _ in range(5)]
    seed_everything(7)
    assert [random.random() for _ in range(5)] == first


def test_numpy_is_reproducible():
    seed_everything(7)
    first = np.random.rand(5).tolist()
    seed_everything(7)
    assert np.random.rand(5).tolist() == first


def test_torch_is_reproducible():
    seed_everything(7)
    first = torch.rand(5)
    seed_everything(7)
    assert torch.equal(torch.rand(5), first)


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.rand(5)
    seed_everything(2)
    assert not torch.equal(torch.rand(5), a)


def test_deterministic_flag_is_recorded():
    assert seed_everything(0, deterministic_torch=True)["deterministic"] is True
    assert seed_everything(0)["deterministic"] is False


def test_pythonhashseed_is_exported():
    import os

    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_make_generator_is_seeded():
    a = torch.randperm(10, generator=make_generator(5))
    b = torch.randperm(10, generator=make_generator(5))
    assert torch.equal(a, b)


def test_dataloader_kwargs_shape():
    kwargs = dataloader_kwargs(seed=0, num_workers=0)
    assert "generator" in kwargs and "worker_init_fn" not in kwargs
    kwargs = dataloader_kwargs(seed=0, num_workers=2)
    assert kwargs["worker_init_fn"] is seed_worker and kwargs["num_workers"] == 2


def test_seeded_shuffling_is_reproducible():
    """The property training needs: identical batch order across runs."""
    from torch.utils.data import DataLoader

    from src.data import ManifestDataset, ManifestRecord

    records = [ManifestRecord("p%d.png" % i, i % 2, "s%d" % i) for i in range(16)]
    dataset = ManifestDataset(records, loader=lambda path: None, preprocess=lambda img: 0)

    def order(seed):
        loader = DataLoader(dataset, batch_size=4, shuffle=True, **dataloader_kwargs(seed))
        return [sid for batch in loader for sid in batch["source_id"]]

    assert order(0) == order(0)
    assert order(0) != order(1)


def test_seed_worker_makes_workers_independent():
    """Without this, NumPy draws can repeat identically in every worker."""
    seeds = []
    for worker_id in range(4):
        torch.manual_seed(1000 + worker_id)
        seed_worker(worker_id)
        seeds.append(np.random.rand())
    assert len(set(seeds)) == len(seeds), "workers must not share a NumPy stream"
