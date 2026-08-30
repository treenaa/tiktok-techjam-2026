"""Training/validation datasets and reproducible DataLoader construction."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, get_worker_info

from src.data import (
    ManifestDataset,
    PairedViewDataset,
    RandomCompetitionTransform,
    make_generator,
    seed_worker,
)

from .config import TrainingConfig


def seed_training_worker(worker_id: int) -> None:
    """Seed global RNGs and the dataset's private augmentation stream.

    ``RandomCompetitionTransform`` owns a ``random.Random`` instance. Without
    reseeding that object after worker spawn, every worker begins from an
    identical copied state and silently repeats the same corruption sequence.
    """
    seed_worker(worker_id)
    info = get_worker_info()
    if info is None:
        return
    augment = getattr(info.dataset, "augment", None)
    if hasattr(augment, "set_seed"):
        augment.set_seed(torch.initial_seed() % (2 ** 32))


def build_datasets(
    train_records: Iterable[Any],
    val_records: Iterable[Any],
    *,
    preprocess: Callable,
    config: TrainingConfig,
    seed: int,
) -> Tuple[Any, ManifestDataset]:
    """Construct a true single-view baseline or paired robustness dataset."""
    train_records = list(train_records)
    val_records = list(val_records)
    if config.augment == "competition":
        augment = RandomCompetitionTransform(
            families=config.augment_families,
            weights=config.augment_weights,
            p_identity=config.augment_identity_probability,
            n_ops=config.augment_operations,
            seed=int(seed),
        )
        train_dataset = PairedViewDataset(
            train_records,
            augment=augment,
            preprocess=preprocess,
            check_paths_exist=True,
        )
    else:
        train_dataset = ManifestDataset(
            train_records,
            preprocess=preprocess,
            check_paths_exist=True,
        )
    validation_dataset = ManifestDataset(
        val_records,
        preprocess=preprocess,
        check_paths_exist=True,
    )
    return train_dataset, validation_dataset


def build_loaders(
    train_dataset: Any,
    validation_dataset: Any,
    config: TrainingConfig,
    *,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic loaders; validation ordering is never shuffled."""
    train_generator = make_generator(seed)
    validation_generator = make_generator(seed + 1)
    shared = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_training_worker if config.num_workers > 0 else None,
        **shared,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        worker_init_fn=seed_worker if config.num_workers > 0 else None,
        **shared,
    )
    return train_loader, validation_loader


def capture_loader_state(loader: DataLoader) -> Mapping[str, Any]:
    state = {}
    generator = getattr(loader, "generator", None)
    if generator is not None:
        state["generator"] = generator.get_state()
    augment = getattr(getattr(loader, "dataset", None), "augment", None)
    rng = getattr(augment, "rng", None)
    if rng is not None:
        state["augmentation_rng"] = rng.getstate()
    return state


def restore_loader_state(loader: DataLoader, state: Mapping[str, Any]) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None and state.get("generator") is not None:
        generator.set_state(state["generator"])
    augment = getattr(getattr(loader, "dataset", None), "augment", None)
    rng = getattr(augment, "rng", None)
    if rng is not None and state.get("augmentation_rng") is not None:
        rng.setstate(state["augmentation_rng"])
