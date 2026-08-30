"""Validation-only decision-threshold selection."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from .config import TrainingError


def select_threshold(
    labels: Iterable[int],
    probabilities: Iterable[float],
    metric: str = "f1",
) -> Dict[str, Any]:
    """Select a threshold in O(N log N) using validation predictions.

    Ties prefer the candidate closest to 0.5, then the lower threshold. This
    makes the result deterministic without quietly favouring extreme cutoffs.
    """
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise TrainingError("threshold selection needs equally sized non-empty 1D arrays")
    if set(y.tolist()) != {0, 1}:
        raise TrainingError("threshold selection requires both validation labels {0, 1}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise TrainingError("threshold selection requires finite probabilities in [0, 1]")
    if metric not in {"f1", "balanced_accuracy", "accuracy"}:
        raise TrainingError("threshold metric must be f1, balanced_accuracy, or accuracy")

    positives = int(y.sum())
    negatives = len(y) - positives

    def value(tp: int, fp: int) -> float:
        fn = positives - tp
        tn = negatives - fp
        if metric == "f1":
            denominator = 2 * tp + fp + fn
            return float(2 * tp / denominator) if denominator else 0.0
        if metric == "balanced_accuracy":
            return float(0.5 * (tp / positives + tn / negatives))
        return float((tp + tn) / len(y))

    candidates = []
    maximum = float(p.max())
    if maximum < 1.0:
        # This is the only partition not represented by thresholding at an
        # observed score: predict every sample as real.
        candidates.append((value(0, 0), (maximum + 1.0) / 2.0))
    # Include the canonical threshold even when it lies between observed scores.
    pred_at_half = p >= 0.5
    candidates.append(
        (
            value(int(np.sum((y == 1) & pred_at_half)), int(np.sum((y == 0) & pred_at_half))),
            0.5,
        )
    )

    order = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    tp = fp = 0
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and sorted_p[end] == sorted_p[index]:
            end += 1
        group = sorted_y[index:end]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        candidates.append((value(tp, fp), float(sorted_p[index])))
        index = end

    best_value, best_threshold = min(
        candidates,
        key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]),
    )
    return {
        "threshold": float(best_threshold),
        "metric": metric,
        "metric_value": float(best_value),
        "source": "validation",
        "n_samples": int(len(y)),
    }
