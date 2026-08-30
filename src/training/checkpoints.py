"""Atomic, strict, resumable training checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn

from .config import TrainingError


CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Any,
    training_config: Mapping[str, Any],
    threshold: float,
    best_metric: Optional[float],
    best_epoch: Optional[int],
    early_stopping_state: Mapping[str, Any],
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    loader_state: Optional[Mapping[str, Any]] = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": list(history),
        "training_config": dict(training_config),
        "threshold": float(threshold),
        "threshold_source": "validation",
        "best_metric": None if best_metric is None else float(best_metric),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "early_stopping_state": dict(early_stopping_state),
        "rng_state": capture_rng_state(),
        "loader_state": dict(loader_state or {}),
        "run_metadata": dict(run_metadata or {}),
    }


def save_checkpoint(payload: Mapping[str, Any], path: str) -> str:
    """Atomically replace ``path`` so interruption cannot leave a half-file."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def read_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise TrainingError("checkpoint not found: %s" % path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TrainingError("checkpoint root must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(
            "unsupported checkpoint schema %r (expected %d)"
            % (payload.get("schema_version"), CHECKPOINT_SCHEMA_VERSION)
        )
    return payload


def restore_training_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    payload = read_checkpoint(path)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        restore_rng_state(payload.get("rng_state", {}))
    return payload
