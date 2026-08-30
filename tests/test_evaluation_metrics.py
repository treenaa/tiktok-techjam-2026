from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    EvaluationError,
    PredictionTable,
    aggregate_stability,
    auroc,
    binary_classification_metrics,
    robustness_summary,
    score_stability,
)


def table(scores, labels=(0, 0, 1, 1), name="clean"):
    n = len(labels)
    return PredictionTable(
        labels=labels,
        probabilities=scores,
        source_ids=["source-%d" % i for i in range(n)],
        image_paths=["image-%d.png" % i for i in range(n)],
        datasets=["set-a"] * n,
        generators=["", "", "g1", "g2"][:n],
        transform_name=name,
    )


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([0, 1, 0, 1, 1], [0.5, 0.5, 0.2, 0.7, 0.5]),
        ([1, 0], [0.9, 0.1]),
    ],
)
def test_auroc_matches_sklearn_and_handles_ties(labels, scores):
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_single_class_auroc_is_explicitly_undefined():
    assert auroc([1, 1], [0.2, 0.8]) is None


def test_binary_metrics_use_label_one_as_aigc_and_threshold_ge():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.5, 0.49, 0.9, 0.1], threshold=0.5)
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)


def test_metrics_reject_invalid_inputs():
    with pytest.raises(EvaluationError, match="equally sized"):
        binary_classification_metrics([0], [0.1, 0.2])
    with pytest.raises(EvaluationError, match="NaN"):
        binary_classification_metrics([0, 1], [0.1, math.nan])
    with pytest.raises(EvaluationError, match="threshold"):
        binary_classification_metrics([0, 1], [0.1, 0.9], 1.1)
    with pytest.raises(EvaluationError, match="not logits"):
        binary_classification_metrics([0, 1], [-2.0, 2.0])


def test_score_stability_and_class_flips():
    clean = table([0.1, 0.49, 0.51, 0.9])
    transformed = table([0.2, 0.6, 0.4, 0.8], name="jpeg_30")
    result = score_stability(clean, transformed)
    assert result["mean_absolute_drift"] == pytest.approx(0.105)
    assert result["n_class_flips"] == 2
    assert result["class_flip_rate"] == pytest.approx(0.5)


def test_drift_refuses_misaligned_rows_even_when_length_matches():
    clean = table([0.1, 0.2, 0.8, 0.9])
    transformed = table([0.1, 0.2, 0.8, 0.9], name="blur_2.0")
    transformed.image_paths[1] = "different.png"
    with pytest.raises(EvaluationError, match="first differ"):
        score_stability(clean, transformed)


def test_robustness_summary_uses_transforms_only_and_reports_worst():
    metrics = {
        "clean": {"auroc": 0.95},
        "jpeg_30": {"auroc": 0.8},
        "blur_2.0": {"auroc": 0.9},
    }
    summary = robustness_summary(metrics)
    assert summary["mean_transformed_auroc"] == pytest.approx(0.85)
    assert summary["worst_case_transform"] == "jpeg_30"
    assert summary["mean_auroc_drop"] == pytest.approx(0.10)
    assert summary["worst_auroc_drop"] == pytest.approx(0.15)


def test_aggregate_stability_reports_worst_corruption():
    summary = aggregate_stability(
        {
            "jpeg_30": {"mean_absolute_drift": 0.2, "class_flip_rate": 0.1},
            "blur_2.0": {"mean_absolute_drift": 0.1, "class_flip_rate": 0.3},
        }
    )
    assert summary["mean_of_mean_absolute_drift"] == pytest.approx(0.15)
    assert summary["worst_drift_transform"] == "jpeg_30"
    assert summary["worst_flip_transform"] == "blur_2.0"


def test_prediction_table_validates_probability_semantics():
    with pytest.raises(EvaluationError, match=r"\[0, 1\]"):
        table([-0.1, 0.2, 0.8, 1.1])
    with pytest.raises(EvaluationError, match="labels"):
        PredictionTable([0, 2], [0.1, 0.2], ["a", "b"], ["a.png", "b.png"])
