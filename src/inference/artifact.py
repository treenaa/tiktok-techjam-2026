"""Strict reconstruction of a trained detector and its preprocessing."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
from torch import nn

from src.evaluation import resolve_device
from src.models import PARAMETER_LIMIT, count_parameters


class InferenceError(ValueError):
    """Raised when product inference cannot produce a trustworthy score."""


FactoryLike = Union[str, Callable[..., Any]]


def import_callable(value: FactoryLike) -> Callable[..., Any]:
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise InferenceError("factory must be callable or written as 'module.path:function'")
    module_name, attribute = value.rsplit(":", 1)
    try:
        result = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise InferenceError("could not import factory %r: %s" % (value, exc)) from exc
    if not callable(result):
        raise InferenceError("imported object %r is not callable" % value)
    return result


def read_checkpoint_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise InferenceError("checkpoint not found: %s" % path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise InferenceError("checkpoint root must be a mapping")
    return dict(payload)


def extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state_dict", "state_dict"):
        state = payload.get(key)
        if state is not None:
            if not isinstance(state, Mapping):
                raise InferenceError("checkpoint %s is not a state-dict mapping" % key)
            return state
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise InferenceError(
        "checkpoint has no model_state_dict/state_dict; provide a compatible trained checkpoint"
    )


@dataclass
class LoadedArtifact:
    model: nn.Module
    preprocess: Callable[[Any], Any]
    device: str
    threshold: float
    threshold_source: str
    checkpoint_path: str
    metadata: Dict[str, Any]


def load_artifact(
    checkpoint_path: str,
    *,
    model_factory: Optional[FactoryLike] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    preprocess_factory: Optional[FactoryLike] = None,
    preprocess_kwargs: Optional[Mapping[str, Any]] = None,
    device: str = "auto",
    threshold: Optional[float] = None,
) -> LoadedArtifact:
    """Restore model + preprocessing, preferring self-describing metadata.

    Explicit factory/kwargs values override checkpoint metadata. State loading
    is always strict; a mismatched architecture must never reach inference.
    """
    payload = read_checkpoint_file(checkpoint_path)
    metadata = dict(payload.get("run_metadata") or {})

    selected_model_factory = model_factory or metadata.get("model_factory")
    if selected_model_factory is None:
        raise InferenceError(
            "checkpoint is not self-describing; pass --model-factory and --model-kwargs"
        )
    resolved_model_kwargs = dict(metadata.get("model_kwargs") or {})
    resolved_model_kwargs.update(dict(model_kwargs or {}))
    model = import_callable(selected_model_factory)(**resolved_model_kwargs)
    if not isinstance(model, nn.Module):
        raise InferenceError("model factory returned %r, expected torch.nn.Module" % type(model))
    model.load_state_dict(extract_state_dict(payload), strict=True)
    parameter_count = count_parameters(model)
    if parameter_count >= PARAMETER_LIMIT:
        raise InferenceError("model violates the competition's <2B parameter limit")

    selected_preprocess_factory = preprocess_factory or metadata.get("preprocess_factory")
    resolved_preprocess_kwargs = dict(metadata.get("preprocess_kwargs") or {})
    resolved_preprocess_kwargs.update(dict(preprocess_kwargs or {}))
    if selected_preprocess_factory is None:
        backbone = getattr(model, "backbone_name", None)
        if backbone is None:
            raise InferenceError(
                "checkpoint has no preprocessing metadata; pass --preprocess-factory"
            )
        selected_preprocess_factory = "src.models:create_preprocess"
        resolved_preprocess_kwargs.setdefault("backbone", backbone)
    preprocess = import_callable(selected_preprocess_factory)(**resolved_preprocess_kwargs)
    if not callable(preprocess):
        raise InferenceError("preprocess factory returned a non-callable object")

    selected_threshold = (
        float(threshold)
        if threshold is not None
        else float(payload.get("threshold", payload.get("best_threshold", 0.5)))
    )
    if not 0.0 <= selected_threshold <= 1.0:
        raise InferenceError("decision threshold must be in [0, 1]")
    threshold_source = (
        "explicit_override" if threshold is not None else str(payload.get("threshold_source", "default_0.5"))
    )
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    artifact_metadata = {
        **metadata,
        "parameter_count": parameter_count,
        "model_factory": (
            selected_model_factory if isinstance(selected_model_factory, str) else repr(selected_model_factory)
        ),
        "model_kwargs": resolved_model_kwargs,
        "preprocess_factory": (
            selected_preprocess_factory
            if isinstance(selected_preprocess_factory, str)
            else repr(selected_preprocess_factory)
        ),
        "preprocess_kwargs": resolved_preprocess_kwargs,
    }
    return LoadedArtifact(
        model=model,
        preprocess=preprocess,
        device=resolved_device,
        threshold=selected_threshold,
        threshold_source=threshold_source,
        checkpoint_path=os.path.abspath(checkpoint_path),
        metadata=artifact_metadata,
    )
