"""Shared experiment config and comparability (rule 21)."""

from __future__ import annotations

import json
import os

import pytest

from src.data.config import DatasetConfigError
from src.data.experiment import (
    COMPARABILITY_KEYS,
    ExperimentConfig,
    assert_comparable,
    comparability_report,
    load_experiment,
    save_experiment,
)


def base(name="run", model=None, **overrides):
    raw = {
        "name": name,
        "seed": 42,
        "data": {
            "seed": 42,
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "datasets": [{"name": "cifake", "adapter": "cifake", "root": "/data/cifake"}],
        },
        "model": model or {"backbone": "clip_vit_b16"},
        "training": {"epochs": 10, "lr": 3e-4, "batch_size": 32},
        "evaluation": {"metrics": ["auroc"]},
    }
    raw.update(overrides)
    return ExperimentConfig(raw)


# -- structure -------------------------------------------------------------
def test_sections_are_exposed():
    config = base()
    assert config.name == "run" and config.seed == 42
    assert config.model["backbone"] == "clip_vit_b16"
    assert config.training["epochs"] == 10
    assert config.evaluation["metrics"] == ["auroc"]


def test_data_seed_inherits_the_run_seed():
    """A split must not silently use a different seed from training."""
    config = ExperimentConfig({
        "seed": 7,
        "data": {"datasets": [{"adapter": "folder", "root": "/d"}]},
    })
    assert config.data["seed"] == 7


def test_explicit_data_seed_is_respected():
    config = ExperimentConfig({
        "seed": 7,
        "data": {"seed": 1, "datasets": [{"adapter": "folder", "root": "/d"}]},
    })
    assert config.data["seed"] == 1


def test_unknown_section_is_rejected():
    with pytest.raises(DatasetConfigError, match="unknown experiment section"):
        ExperimentConfig({"data": {"datasets": [{"root": "/d"}]}, "optimiser": {}})


def test_section_must_be_a_mapping():
    with pytest.raises(DatasetConfigError, match="must be a mapping"):
        ExperimentConfig({"data": {"datasets": [{"root": "/d"}]}, "model": ["clip"]})


def test_missing_datasets_is_rejected():
    with pytest.raises(DatasetConfigError, match="no data.datasets"):
        ExperimentConfig({"data": {}})


# -- fingerprints ----------------------------------------------------------
def test_fingerprint_ignores_the_model():
    """The whole point: swapping backbones must not change the fingerprint."""
    a = base("a", model={"backbone": "clip"})
    b = base("b", model={"backbone": "dinov2"})
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_the_split():
    a = base()
    b = base()
    b.raw["data"]["ratios"] = {"train": 0.8, "val": 0.1, "test": 0.1}
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_changes_with_training_settings():
    a = base()
    b = base()
    b.raw["training"]["lr"] = 1e-3
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_changes_with_the_seed():
    assert base().fingerprint() != base(seed=1).fingerprint()


def test_fingerprint_is_stable_across_calls():
    config = base()
    assert config.fingerprint() == config.fingerprint()


def test_comparability_keys_exclude_the_model():
    assert "model" not in COMPARABILITY_KEYS
    assert set(COMPARABILITY_KEYS) == {"data", "training", "evaluation", "seed"}


# -- comparability ---------------------------------------------------------
def test_backbone_only_difference_is_comparable():
    configs = [
        base("clip", model={"backbone": "clip"}),
        base("dino", model={"backbone": "dinov2"}),
        base("ijepa", model={"backbone": "ijepa"}),
    ]
    report = comparability_report(configs)
    assert report["comparable"]
    assert len(set(report["fingerprints"].values())) == 1
    assert_comparable(configs)


def test_differing_learning_rate_is_not_comparable():
    a = base("a", model={"backbone": "clip"})
    b = base("b", model={"backbone": "dino"})
    b.raw["training"]["lr"] = 1e-3
    report = comparability_report([a, b])
    assert not report["comparable"]
    assert any("training.lr" in d for d in report["differences"]["a vs b"])


def test_differing_split_is_not_comparable():
    a = base("a", model={"backbone": "clip"})
    b = base("b", model={"backbone": "dino"})
    b.raw["data"]["ratios"]["train"] = 0.8
    assert not comparability_report([a, b])["comparable"]


def test_assert_comparable_message_names_the_offending_key():
    a = base("a", model={"backbone": "clip"})
    b = base("b", model={"backbone": "dino"})
    b.raw["training"]["epochs"] = 20
    with pytest.raises(AssertionError) as excinfo:
        assert_comparable([a, b])
    message = str(excinfo.value)
    assert "rule 21" in message and "training.epochs" in message


def test_identical_models_are_a_vacuous_comparison():
    configs = [base("a"), base("b")]  # same model section
    assert comparability_report(configs)["identical_models"]
    with pytest.raises(AssertionError, match="vacuous"):
        assert_comparable(configs)


def test_added_key_is_detected():
    a = base("a", model={"backbone": "clip"})
    b = base("b", model={"backbone": "dino"})
    b.raw["training"]["scheduler"] = "cosine"
    diffs = comparability_report([a, b])["differences"]["a vs b"]
    assert any("scheduler" in d for d in diffs)


def test_comparing_one_config_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        comparability_report([base()])


# -- persistence -----------------------------------------------------------
def test_json_round_trip(tmp_path):
    path = str(tmp_path / "exp.json")
    original = base("saved")
    save_experiment(original, path)
    restored = load_experiment(path)
    assert restored.name == "saved"
    assert restored.fingerprint() == original.fingerprint()


def test_yaml_round_trip(tmp_path):
    pytest.importorskip("yaml")
    path = str(tmp_path / "exp.yaml")
    original = base("saved")
    save_experiment(original, path)
    assert load_experiment(path).fingerprint() == original.fingerprint()


def test_missing_file_is_reported():
    with pytest.raises(DatasetConfigError, match="not found"):
        load_experiment("/nonexistent/exp.yaml")


# -- the shipped baseline configs -----------------------------------------
def test_shipped_baselines_are_mutually_comparable():
    """The Phase-1 configs must stay comparable as they are edited."""
    pytest.importorskip("yaml")
    configs = [
        load_experiment("configs/baseline_%s.yaml" % name)
        for name in ("clip", "dino", "ijepa")
    ]
    assert_comparable(configs)


def test_shipped_baselines_use_distinct_backbones():
    pytest.importorskip("yaml")
    backbones = {
        load_experiment("configs/baseline_%s.yaml" % name).model["backbone"]
        for name in ("clip", "dino", "ijepa")
    }
    assert len(backbones) == 3


def test_shipped_baselines_keep_the_demo_subset_out_of_training():
    pytest.importorskip("yaml")
    config = load_experiment("configs/baseline_clip.yaml")
    demo_names = {entry["name"] for entry in config.data["demo"]}
    assert demo_names == {"coco_val2017", "dalle_advanced"}
    train_names = {entry["name"] for entry in config.data["datasets"]}
    assert not (train_names & demo_names)


def test_shipped_baselines_select_on_validation():
    """Rule 11.D: never tune on test."""
    pytest.importorskip("yaml")
    for name in ("clip", "dino", "ijepa"):
        config = load_experiment("configs/baseline_%s.yaml" % name)
        assert config.evaluation["threshold_source"] == "val"
        assert config.evaluation["primary_metric"].startswith("val")
        assert config.training["early_stopping"]["monitor"].startswith("val")
