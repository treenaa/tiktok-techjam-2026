"""Model execution over the deterministic robustness grid."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Tuple

from src.data import dataloader_kwargs

from .types import EvaluationError, EvaluationRun, PredictionTable, RuntimeStats


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in that order."""
    import torch

    requested = str(requested).lower()
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise EvaluationError("CUDA was requested but is not available")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise EvaluationError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _forward(model: Any, images: Any) -> Any:
    return model(**images) if isinstance(images, Mapping) else model(images)


def extract_logits(output: Any, expected_batch_size: Optional[int] = None):
    """Extract one raw binary logit per image from common model outputs.

    Accepted shapes are ``(B,)`` and ``(B, 1)``.  A two-class softmax head is
    rejected because this project's canonical contract is one AIGC logit and
    silently choosing a column could invert the label convention.
    """
    import torch

    if isinstance(output, Mapping):
        if "logits" not in output:
            raise EvaluationError("model returned a mapping without a 'logits' key")
        output = output["logits"]
    elif hasattr(output, "logits"):
        output = output.logits
    elif isinstance(output, (tuple, list)):
        if not output:
            raise EvaluationError("model returned an empty tuple/list")
        output = output[0]
    if not torch.is_tensor(output):
        raise EvaluationError("model output must contain a torch.Tensor of raw logits")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1:
        raise EvaluationError(
            "binary detector must return logits shaped (B,) or (B, 1), got %s"
            % (tuple(output.shape),)
        )
    if expected_batch_size is not None and len(output) != expected_batch_size:
        raise EvaluationError(
            "model returned %d logits for a batch of %d" % (len(output), expected_batch_size)
        )
    if not bool(torch.isfinite(output).all()):
        raise EvaluationError("model returned NaN or infinite logits")
    return output


def _strings(batch: Mapping[str, Any], key: str, size: int, default: str = "") -> list[str]:
    values = batch.get(key)
    if values is None:
        return [default] * size
    if isinstance(values, str):
        values = [values]
    values = list(values)
    if len(values) != size:
        raise EvaluationError("batch metadata %r has length %d, expected %d" % (key, len(values), size))
    return [str(value) for value in values]


def predict_dataset(
    model: Any,
    dataset: Any,
    *,
    transform_name: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
    pin_memory: Optional[bool] = None,
    seed: int = 0,
) -> Tuple[PredictionTable, RuntimeStats]:
    """Run one transformed dataset and return aligned probabilities + timing."""
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise EvaluationError("batch_size must be at least 1")
    if num_workers < 0:
        raise EvaluationError("num_workers cannot be negative")
    if len(dataset) == 0:
        raise EvaluationError("cannot evaluate an empty dataset")
    device = resolve_device(device)
    if pin_memory is None:
        pin_memory = device.startswith("cuda")
    loader_options = dataloader_kwargs(seed=seed, num_workers=num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=bool(pin_memory),
        **loader_options,
    )

    model.to(device)
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    source_ids: list[str] = []
    image_paths: list[str] = []
    datasets: list[str] = []
    generators: list[str] = []
    batch_times: list[float] = []
    observed_names: set[str] = set()

    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            if "image" not in batch or "label" not in batch:
                raise EvaluationError("evaluation batch requires 'image' and 'label'")
            batch_labels = batch["label"]
            if torch.is_tensor(batch_labels):
                batch_labels_list = batch_labels.detach().cpu().reshape(-1).tolist()
            else:
                batch_labels_list = list(batch_labels)
            n = len(batch_labels_list)
            images = _to_device(batch["image"], device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = _forward(model, images)
            logits = extract_logits(output, expected_batch_size=n)
            probs = torch.sigmoid(logits)  # exactly once: logits -> P(AIGC)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - start)

            labels.extend(int(value) for value in batch_labels_list)
            probabilities.extend(float(value) for value in probs.detach().cpu().tolist())
            source_ids.extend(_strings(batch, "source_id", n))
            image_paths.extend(_strings(batch, "image_path", n))
            datasets.extend(_strings(batch, "dataset", n))
            generators.extend(_strings(batch, "generator", n))
            observed_names.update(_strings(batch, "transform_name", n, default="clean"))
    total_seconds = time.perf_counter() - wall_start

    name = transform_name or getattr(dataset, "transform_name", None)
    if name is None:
        if len(observed_names) != 1:
            raise EvaluationError("could not infer one transform name from batch metadata")
        name = next(iter(observed_names))
    if observed_names and observed_names != {str(name)}:
        raise EvaluationError(
            "dataset requested transform %r but emitted transform names %s"
            % (name, sorted(observed_names))
        )

    table = PredictionTable(
        labels=labels,
        probabilities=probabilities,
        source_ids=source_ids,
        image_paths=image_paths,
        datasets=datasets,
        generators=generators,
        transform_name=str(name),
    )
    runtime = RuntimeStats(
        n_samples=len(table),
        n_batches=len(batch_times),
        total_seconds=total_seconds,
        batch_seconds=tuple(batch_times),
    )
    return table, runtime


def evaluate_grid(
    model: Any,
    datasets: Mapping[str, Any],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
    pin_memory: Optional[bool] = None,
    seed: int = 0,
) -> EvaluationRun:
    """Evaluate a model on clean plus deterministic transformed datasets."""
    if "clean" not in datasets:
        raise EvaluationError("evaluation grid requires a 'clean' dataset")
    predictions: Dict[str, PredictionTable] = {}
    runtimes: Dict[str, RuntimeStats] = {}
    for name, dataset in datasets.items():
        table, runtime = predict_dataset(
            model,
            dataset,
            transform_name=name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            pin_memory=pin_memory,
            seed=seed,
        )
        predictions[name] = table
        runtimes[name] = runtime
    return EvaluationRun(predictions=predictions, runtimes=runtimes)
