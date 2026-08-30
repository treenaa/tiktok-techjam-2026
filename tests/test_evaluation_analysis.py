from __future__ import annotations

import pytest

from src.evaluation import PredictionTable, representative_errors, subgroup_metrics


def make_table():
    return PredictionTable(
        labels=[0, 0, 1, 1, 1],
        probabilities=[0.9, 0.6, 0.1, 0.4, 0.8],
        source_ids=["r1", "r2", "a1", "a2", "a3"],
        image_paths=["r1.png", "r2.png", "a1.png", "a2.png", "a3.png"],
        datasets=["d1", "d2", "d1", "d1", "d2"],
        generators=["", "", "g1", "g1", "g2"],
    )


def test_generator_metrics_pair_each_generator_with_all_real():
    groups = subgroup_metrics(make_table(), "generator")
    assert set(groups) == {"g1", "g2"}
    assert groups["g1"]["n_real"] == 2
    assert groups["g1"]["n_aigc"] == 2
    assert groups["g2"]["n_real"] == 2
    assert groups["g2"]["n_aigc"] == 1
    assert groups["g1"]["protocol"] == "all_real_vs_generator_aigc"


def test_dataset_metrics_keep_undefined_auroc_as_none():
    groups = subgroup_metrics(make_table(), "dataset")
    assert groups["d1"]["n_samples"] == 3
    assert groups["d2"]["n_samples"] == 2
    assert groups["d2"]["auroc"] is not None


def test_representative_errors_are_most_confident_first():
    errors = representative_errors(make_table(), max_per_type=2)
    assert [row["image_path"] for row in errors["false_positives"]] == ["r1.png", "r2.png"]
    assert [row["image_path"] for row in errors["false_negatives"]] == ["a1.png", "a2.png"]
    assert all(row["predicted_label"] != row["label"] for values in errors.values() for row in values)


def test_error_limit_zero_is_supported():
    errors = representative_errors(make_table(), max_per_type=0)
    assert errors == {"false_positives": [], "false_negatives": []}
