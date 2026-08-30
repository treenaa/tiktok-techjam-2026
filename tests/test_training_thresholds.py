from __future__ import annotations

import itertools

import pytest

from src.evaluation import binary_classification_metrics
from src.training import EarlyStopping, TrainingError, select_threshold


def test_threshold_selection_maximizes_validation_f1():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.6, 0.55, 0.9]
    selected = select_threshold(labels, probabilities, metric="f1")
    candidates = [0.5] + probabilities
    brute_best = max(
        binary_classification_metrics(labels, probabilities, threshold=value)["f1"]
        for value in candidates
    )
    assert selected["metric_value"] == pytest.approx(brute_best)
    assert selected["source"] == "validation"


@pytest.mark.parametrize("metric", ["f1", "balanced_accuracy", "accuracy"])
def test_threshold_selection_is_deterministic_for_ties(metric):
    first = select_threshold([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], metric)
    second = select_threshold([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], metric)
    assert first == second


def test_threshold_selection_refuses_single_class_or_logits():
    with pytest.raises(TrainingError, match="both validation labels"):
        select_threshold([1, 1], [0.2, 0.8])
    with pytest.raises(TrainingError, match="probabilities"):
        select_threshold([0, 1], [-2.0, 2.0])


def test_accuracy_selection_can_choose_the_all_real_partition():
    selected = select_threshold(
        [0, 0, 0, 1],
        [0.6, 0.7, 0.8, 0.1],
        metric="accuracy",
    )
    assert selected["threshold"] > 0.8
    assert selected["metric_value"] == pytest.approx(0.75)


def test_early_stopping_tracks_direction_delta_and_patience():
    stopper = EarlyStopping(mode="max", patience=2, min_delta=0.01)
    assert stopper.step(0.7) == (True, False)
    assert stopper.step(0.705) == (False, False)
    assert stopper.step(0.704) == (False, True)
    assert stopper.best == 0.7


def test_patience_zero_does_not_stop_on_an_improvement():
    stopper = EarlyStopping(mode="min", patience=0)
    assert stopper.step(1.0) == (True, False)
    assert stopper.step(1.1) == (False, True)
