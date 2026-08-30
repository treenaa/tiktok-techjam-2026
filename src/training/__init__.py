"""Robust, leakage-conscious training for binary AIGC detection."""

from .checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    checkpoint_payload,
    read_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig, TrainingError
from .engine import (
    EarlyStopping,
    Trainer,
    TrainingResult,
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    validate_one_epoch,
)
from .loaders import (
    build_datasets,
    build_loaders,
    capture_loader_state,
    restore_loader_state,
    seed_training_worker,
)
from .losses import LossOutput, RobustBinaryObjective
from .thresholds import select_threshold

__all__ = [
    "TrainingError",
    "TrainingConfig",
    "LossOutput",
    "RobustBinaryObjective",
    "select_threshold",
    "seed_training_worker",
    "build_datasets",
    "build_loaders",
    "capture_loader_state",
    "restore_loader_state",
    "CHECKPOINT_SCHEMA_VERSION",
    "capture_rng_state",
    "restore_rng_state",
    "checkpoint_payload",
    "save_checkpoint",
    "read_checkpoint",
    "restore_training_checkpoint",
    "EarlyStopping",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "validate_one_epoch",
    "Trainer",
    "TrainingResult",
]
