"""Typed, validated containers shared by the evaluation subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


class EvaluationError(ValueError):
    """Raised when evaluation inputs would make a result invalid or ambiguous."""


def _as_1d(values: Iterable[Any], dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=dtype)
    if array.ndim != 1:
        raise EvaluationError("%s must be one-dimensional, got shape %s" % (name, array.shape))
    return array


@dataclass
class PredictionTable:
    """Predictions and provenance for one dataset view, in dataset order.

    ``probabilities`` always means P(AIGC), never logits or hard predictions.
    Identity is the pair ``(source_id, image_path)`` because a manifest may
    legitimately contain several files derived from one source image.
    """

    labels: Sequence[int]
    probabilities: Sequence[float]
    source_ids: Sequence[str]
    image_paths: Sequence[str]
    datasets: Sequence[str] = field(default_factory=list)
    generators: Sequence[str] = field(default_factory=list)
    transform_name: str = "clean"

    def __post_init__(self) -> None:
        self.labels = _as_1d(self.labels, np.int64, "labels")
        self.probabilities = _as_1d(self.probabilities, np.float64, "probabilities")
        self.source_ids = [str(value) for value in self.source_ids]
        self.image_paths = [str(value) for value in self.image_paths]
        n = len(self.labels)
        if n == 0:
            raise EvaluationError("prediction table is empty")
        lengths = {
            "probabilities": len(self.probabilities),
            "source_ids": len(self.source_ids),
            "image_paths": len(self.image_paths),
        }
        bad = {name: size for name, size in lengths.items() if size != n}
        if bad:
            raise EvaluationError("prediction table length mismatch: labels=%d, %s" % (n, bad))
        labels = set(self.labels.tolist())
        if not labels <= {0, 1}:
            raise EvaluationError("labels must be 0 (real) or 1 (AIGC), got %s" % sorted(labels))
        if not np.all(np.isfinite(self.probabilities)):
            raise EvaluationError("probabilities contain NaN or infinity")
        if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
            raise EvaluationError("probabilities must be in [0, 1]")
        if not all(self.source_ids):
            raise EvaluationError("source_ids must be non-empty")
        if not all(self.image_paths):
            raise EvaluationError("image_paths must be non-empty")

        self.datasets = self._metadata(self.datasets, n, "datasets")
        self.generators = self._metadata(self.generators, n, "generators")
        self.transform_name = str(self.transform_name)
        if not self.transform_name:
            raise EvaluationError("transform_name must be non-empty")

    @staticmethod
    def _metadata(values: Sequence[str], n: int, name: str) -> List[str]:
        if not values:
            return [""] * n
        if len(values) != n:
            raise EvaluationError("%s length %d does not match labels length %d" % (name, len(values), n))
        return [str(value) for value in values]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def identity(self) -> List[Tuple[str, str]]:
        return list(zip(self.source_ids, self.image_paths))

    def assert_aligned(self, other: "PredictionTable") -> None:
        """Require row-for-row identity and labels before computing drift."""
        if len(self) != len(other):
            raise EvaluationError(
                "prediction tables are not aligned: %s has %d rows, %s has %d"
                % (self.transform_name, len(self), other.transform_name, len(other))
            )
        if self.identity != other.identity:
            for index, (left, right) in enumerate(zip(self.identity, other.identity)):
                if left != right:
                    raise EvaluationError(
                        "prediction tables first differ at row %d: %r != %r" % (index, left, right)
                    )
            raise EvaluationError("prediction table identities differ")
        if not np.array_equal(self.labels, other.labels):
            raise EvaluationError("aligned prediction tables have different labels")

    def subset(self, indices: Iterable[int]) -> "PredictionTable":
        idx = np.asarray(list(indices), dtype=np.int64)
        return PredictionTable(
            labels=self.labels[idx],
            probabilities=self.probabilities[idx],
            source_ids=[self.source_ids[i] for i in idx],
            image_paths=[self.image_paths[i] for i in idx],
            datasets=[self.datasets[i] for i in idx],
            generators=[self.generators[i] for i in idx],
            transform_name=self.transform_name,
        )

    def rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "image_path": self.image_paths[i],
                "source_id": self.source_ids[i],
                "label": int(self.labels[i]),
                "prob_aigc": float(self.probabilities[i]),
                "dataset": self.datasets[i],
                "generator": self.generators[i],
                "transform_name": self.transform_name,
            }
            for i in range(len(self))
        ]


@dataclass(frozen=True)
class RuntimeStats:
    """Wall-clock inference measurements for one transformed dataset."""

    n_samples: int
    n_batches: int
    total_seconds: float
    batch_seconds: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        batches = np.asarray(self.batch_seconds, dtype=np.float64)
        return {
            "n_samples": int(self.n_samples),
            "n_batches": int(self.n_batches),
            "total_seconds": float(self.total_seconds),
            "samples_per_second": (
                float(self.n_samples / self.total_seconds) if self.total_seconds > 0 else None
            ),
            "mean_model_batch_ms": float(batches.mean() * 1000.0) if len(batches) else None,
            "p50_model_batch_ms": float(np.percentile(batches, 50) * 1000.0) if len(batches) else None,
            "p95_model_batch_ms": float(np.percentile(batches, 95) * 1000.0) if len(batches) else None,
        }


@dataclass
class EvaluationRun:
    """Prediction tables and timing for a complete robustness grid."""

    predictions: Mapping[str, PredictionTable]
    runtimes: Mapping[str, RuntimeStats]

    def __post_init__(self) -> None:
        self.predictions = dict(self.predictions)
        self.runtimes = dict(self.runtimes)
        if "clean" not in self.predictions:
            raise EvaluationError("evaluation run requires a 'clean' prediction table")
        if set(self.predictions) != set(self.runtimes):
            raise EvaluationError("runtime keys must exactly match prediction keys")
        clean = self.predictions["clean"]
        for name, table in self.predictions.items():
            if name != table.transform_name:
                raise EvaluationError(
                    "prediction key %r does not match transform_name %r" % (name, table.transform_name)
                )
            clean.assert_aligned(table)
