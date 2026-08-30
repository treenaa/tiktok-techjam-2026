"""Subgroup and representative-error analysis for AIGC predictions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .metrics import binary_classification_metrics
from .types import EvaluationError, PredictionTable


def subgroup_metrics(
    table: PredictionTable,
    field: str,
    threshold: float = 0.5,
    *,
    pair_generators_with_all_real: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Metrics by dataset or generator with explicit support counts.

    Generator slices contain only AIGC rows by definition.  By default each
    generator is paired with every real row, producing a meaningful two-class
    cross-generator diagnostic.  The report labels this protocol explicitly.
    """
    if field not in {"dataset", "generator"}:
        raise EvaluationError("subgroup field must be 'dataset' or 'generator'")
    values = table.datasets if field == "dataset" else table.generators
    groups: Dict[str, Dict[str, Any]] = {}
    if field == "generator" and pair_generators_with_all_real:
        real_indices = np.flatnonzero(table.labels == 0).tolist()
        names = sorted({value for value, label in zip(values, table.labels) if label == 1 and value})
        for name in names:
            aigc_indices = [
                index
                for index, (value, label) in enumerate(zip(values, table.labels))
                if label == 1 and value == name
            ]
            subset = table.subset(real_indices + aigc_indices)
            metrics = binary_classification_metrics(subset.labels, subset.probabilities, threshold)
            metrics["protocol"] = "all_real_vs_generator_aigc"
            groups[name] = metrics
        return groups

    names = sorted(set(values))
    for name in names:
        display_name = name or "unknown"
        indices = [index for index, value in enumerate(values) if value == name]
        subset = table.subset(indices)
        metrics = binary_classification_metrics(subset.labels, subset.probabilities, threshold)
        metrics["protocol"] = "within_%s" % field
        groups[display_name] = metrics
    return groups


def representative_errors(
    table: PredictionTable, threshold: float = 0.5, max_per_type: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """Most confident false positives and false negatives for inspection."""
    if max_per_type < 0:
        raise EvaluationError("max_per_type cannot be negative")
    pred = table.probabilities >= threshold
    false_positive = np.flatnonzero((table.labels == 0) & pred)
    false_negative = np.flatnonzero((table.labels == 1) & ~pred)
    false_positive = sorted(false_positive, key=lambda i: (-table.probabilities[i], i))
    false_negative = sorted(false_negative, key=lambda i: (table.probabilities[i], i))

    def rows(indices: Iterable[int], kind: str) -> List[Dict[str, Any]]:
        return [
            {
                "error_type": kind,
                "image_path": table.image_paths[i],
                "source_id": table.source_ids[i],
                "label": int(table.labels[i]),
                "prob_aigc": float(table.probabilities[i]),
                "predicted_label": int(pred[i]),
                "dataset": table.datasets[i],
                "generator": table.generators[i],
                "transform_name": table.transform_name,
            }
            for i in list(indices)[:max_per_type]
        ]

    return {
        "false_positives": rows(false_positive, "false_positive"),
        "false_negatives": rows(false_negative, "false_negative"),
    }
