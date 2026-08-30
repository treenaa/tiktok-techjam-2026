"""Numerically explicit binary classification and robustness metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .types import EvaluationError, PredictionTable


def _validated_arrays(labels: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise EvaluationError("labels and scores must be one-dimensional and equally sized")
    if len(y) == 0:
        raise EvaluationError("metrics require at least one sample")
    if not set(y.tolist()) <= {0, 1}:
        raise EvaluationError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(p)):
        raise EvaluationError("scores contain NaN or infinity")
    return y, p


def auroc(labels: Iterable[int], scores: Iterable[float]) -> Optional[float]:
    """Area under the ROC curve from continuous scores, with tie handling.

    Returns ``None`` for a single-class slice.  That is expected for some
    subgroup reports; callers must not silently average those missing values.
    """
    y, p = _validated_arrays(labels, scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(p, kind="mergesort")
    sorted_scores = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_classification_metrics(
    labels: Iterable[int], scores: Iterable[float], threshold: float = 0.5
) -> Dict[str, Any]:
    """Core project metrics for label 1 = AIGC.

    The threshold is an input selected outside this function (normally on the
    validation set).  This module intentionally contains no test-set threshold
    optimiser.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise EvaluationError("threshold must be in [0, 1]")
    y, p = _validated_arrays(labels, scores)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise EvaluationError(
            "binary classification scores must be probabilities in [0, 1], not logits"
        )
    pred = (p >= float(threshold)).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        "n_samples": int(len(y)),
        "n_real": int(np.sum(y == 0)),
        "n_aigc": int(np.sum(y == 1)),
        "threshold": float(threshold),
        "auroc": auroc(y, p),
        "accuracy": divide(tp + tn, len(y)),
        "f1": divide(2.0 * precision * recall, precision + recall),
        "precision": precision,
        "recall": recall,
        "specificity": divide(tn, tn + fp),
        "false_positive_rate": divide(fp, fp + tn),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def score_stability(
    clean: PredictionTable, transformed: PredictionTable, threshold: float = 0.5
) -> Dict[str, Any]:
    """Prediction drift for aligned clean/transformed samples."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise EvaluationError("threshold must be in [0, 1]")
    clean.assert_aligned(transformed)
    drift = np.abs(clean.probabilities - transformed.probabilities)
    clean_class = clean.probabilities >= threshold
    transformed_class = transformed.probabilities >= threshold
    return {
        "mean_absolute_drift": float(drift.mean()),
        "median_absolute_drift": float(np.median(drift)),
        "p95_absolute_drift": float(np.percentile(drift, 95)),
        "max_absolute_drift": float(drift.max()),
        "class_flip_rate": float(np.mean(clean_class != transformed_class)),
        "n_class_flips": int(np.sum(clean_class != transformed_class)),
    }


def robustness_summary(transform_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate clean-to-corruption AUROC degradation without hiding gaps."""
    if "clean" not in transform_metrics:
        raise EvaluationError("transform metrics require a 'clean' entry")
    clean_auc = transform_metrics["clean"].get("auroc")
    if clean_auc is None:
        raise EvaluationError("clean AUROC is undefined; main evaluation needs both labels")

    transformed = {
        name: values.get("auroc")
        for name, values in transform_metrics.items()
        if name != "clean"
    }
    valid = {name: float(value) for name, value in transformed.items() if value is not None}
    missing = sorted(name for name, value in transformed.items() if value is None)
    if not valid:
        return {
            "clean_auroc": float(clean_auc),
            "mean_transformed_auroc": None,
            "worst_case_transformed_auroc": None,
            "worst_case_transform": None,
            "mean_auroc_drop": None,
            "worst_auroc_drop": None,
            "missing_auroc_transforms": missing,
        }
    worst_name = min(valid, key=lambda name: (valid[name], name))
    drops = {name: float(clean_auc) - value for name, value in valid.items()}
    worst_drop_name = min(drops, key=lambda name: (-drops[name], name))
    return {
        "clean_auroc": float(clean_auc),
        "mean_transformed_auroc": float(np.mean(list(valid.values()))),
        "worst_case_transformed_auroc": valid[worst_name],
        "worst_case_transform": worst_name,
        "mean_auroc_drop": float(np.mean(list(drops.values()))),
        "worst_auroc_drop": drops[worst_drop_name],
        "worst_drop_transform": worst_drop_name,
        "missing_auroc_transforms": missing,
    }


def aggregate_stability(stability_by_transform: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Headline stability across corruptions while retaining the worst view."""
    if not stability_by_transform:
        return {
            "mean_of_mean_absolute_drift": None,
            "worst_mean_absolute_drift": None,
            "worst_drift_transform": None,
            "mean_class_flip_rate": None,
            "worst_class_flip_rate": None,
            "worst_flip_transform": None,
        }
    required = {"mean_absolute_drift", "class_flip_rate"}
    for name, values in stability_by_transform.items():
        missing = required - set(values)
        if missing:
            raise EvaluationError("stability entry %r is missing %s" % (name, sorted(missing)))
    mean_drifts = {
        name: float(values["mean_absolute_drift"])
        for name, values in stability_by_transform.items()
    }
    flip_rates = {
        name: float(values["class_flip_rate"])
        for name, values in stability_by_transform.items()
    }
    worst_drift = min(mean_drifts, key=lambda name: (-mean_drifts[name], name))
    worst_flip = min(flip_rates, key=lambda name: (-flip_rates[name], name))
    return {
        "mean_of_mean_absolute_drift": float(np.mean(list(mean_drifts.values()))),
        "worst_mean_absolute_drift": mean_drifts[worst_drift],
        "worst_drift_transform": worst_drift,
        "mean_class_flip_rate": float(np.mean(list(flip_rates.values()))),
        "worst_class_flip_rate": flip_rates[worst_flip],
        "worst_flip_transform": worst_flip,
    }
