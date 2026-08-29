"""Config-driven builds (rule 20.8) and generic field holdout (rule 11.C)."""

from __future__ import annotations

import json
import os

import pytest

from src.data import (
    DataError,
    DatasetConfigError,
    LeakageError,
    ManifestRecord,
    ProtectedDataError,
    assert_field_disjoint,
    build_from_config,
    list_field_values,
    load_config,
    split_by_field_holdout,
)
from test_data_fixtures import build_cifake_tree, build_wildfake_tree, write_image


# ==========================================================================
# generic field holdout
# ==========================================================================
def multi_dataset_records(n=12):
    records = []
    for dataset in ("cifake", "wildfake", "sid_set"):
        for i in range(n):
            label = i % 2
            records.append(ManifestRecord(
                "%s/%d.png" % (dataset, i), label, "%s_%d" % (dataset, i),
                dataset=dataset, generator=("gen_%d" % (i % 3)) if label else ""))
    return records


def test_list_field_values_works_on_any_column():
    records = multi_dataset_records()
    assert list_field_values(records, "dataset") == ["cifake", "sid_set", "wildfake"]
    assert list_field_values(records, "generator") == ["gen_0", "gen_1", "gen_2"]


def test_list_field_values_can_restrict_to_a_label():
    records = multi_dataset_records()
    assert list_field_values(records, "generator", label=0) == []


def test_unseen_dataset_holdout():
    """Cross-source protocol: train on two corpora, test on a third."""
    splits = split_by_field_holdout(
        multi_dataset_records(), field="dataset", holdout=["sid_set"],
        holdout_label=None, seed=0,
    )
    assert list_field_values(splits["test"], "dataset") == ["sid_set"]
    assert "sid_set" not in list_field_values(splits["train"], "dataset")
    assert "sid_set" not in list_field_values(splits["val"], "dataset")


def test_unseen_dataset_holdout_keeps_both_labels():
    splits = split_by_field_holdout(
        multi_dataset_records(), field="dataset", holdout=["sid_set"],
        holdout_label=None, seed=0,
    )
    for name, members in splits.items():
        assert {r.label for r in members} == {0, 1}, name


def test_field_holdout_preserves_source_id_grouping():
    records = []
    for dataset in ("a", "b", "c"):
        for i in range(8):
            for view in ("clean", "jpeg_70"):
                records.append(ManifestRecord(
                    "%s/%d_%s.png" % (dataset, i, view), i % 2, "%s_%d" % (dataset, i),
                    dataset=dataset))
    splits = split_by_field_holdout(records, field="dataset", holdout=["c"],
                                    holdout_label=None, seed=0)
    seen = {}
    for name, members in splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name


def test_field_holdout_is_deterministic():
    records = multi_dataset_records()
    a = split_by_field_holdout(records, field="dataset", holdout=["sid_set"], holdout_label=None, seed=3)
    b = split_by_field_holdout(records, field="dataset", holdout=["sid_set"], holdout_label=None, seed=3)
    assert {k: sorted(r.image_path for r in v) for k, v in a.items()} == {
        k: sorted(r.image_path for r in v) for k, v in b.items()
    }


def test_field_holdout_by_count():
    splits = split_by_field_holdout(
        multi_dataset_records(), field="dataset", n_holdout=1, holdout_label=None, seed=1
    )
    assert len(list_field_values(splits["test"], "dataset")) == 1


def test_field_holdout_rejects_unknown_value():
    with pytest.raises(DataError, match="not present"):
        split_by_field_holdout(multi_dataset_records(), field="dataset", holdout=["nope"])


def test_field_holdout_rejects_holding_out_everything():
    with pytest.raises(DataError, match="nothing to train on"):
        split_by_field_holdout(
            multi_dataset_records(), field="dataset",
            holdout=["cifake", "wildfake", "sid_set"], holdout_label=None,
        )


def test_field_holdout_requires_the_column():
    records = [ManifestRecord("a%d.png" % i, i % 2, "s%d" % i) for i in range(6)]
    with pytest.raises(DataError, match="needs that column populated"):
        split_by_field_holdout(records, field="generator", n_holdout=1)


def test_assert_field_disjoint_detects_a_leak():
    splits = split_by_field_holdout(
        multi_dataset_records(), field="dataset", holdout=["sid_set"],
        holdout_label=None, seed=0,
    )
    splits["train"].append(ManifestRecord("sid_set/leak.png", 1, "leak", dataset="sid_set"))
    with pytest.raises(LeakageError, match="dataset leakage"):
        assert_field_disjoint(splits, field="dataset", holdout=["sid_set"])


# ==========================================================================
# config
# ==========================================================================
@pytest.fixture
def roots(tmp_path):
    return {
        "cifake": build_cifake_tree(tmp_path / "cifake", n_per_class=8),
        "wildfake": build_wildfake_tree(tmp_path / "wildfake", n_per_leaf=4),
    }


def write_config(tmp_path, config, name="data.json"):
    path = str(tmp_path / name)
    with open(path, "w") as fh:
        json.dump(config, fh)
    return path


def test_build_from_config_produces_valid_splits(roots, tmp_path):
    config = {
        "seed": 1,
        "datasets": [
            {"name": "cifake", "adapter": "cifake", "root": roots["cifake"]},
            {"name": "wildfake", "adapter": "wildfake", "root": roots["wildfake"]},
        ],
    }
    splits = build_from_config(config)
    assert set(splits) == {"train", "val", "test"}
    for members in splits.values():
        assert {r.label for r in members} == {0, 1}
    assert {r.dataset for r in splits["train"]} == {"cifake", "wildfake"}


def test_config_splits_are_leakage_free(roots):
    splits = build_from_config({
        "seed": 0,
        "datasets": [{"name": "cifake", "adapter": "cifake", "root": roots["cifake"]}],
    })
    seen = {}
    for name, members in splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name


def test_config_is_deterministic(roots):
    config = {"seed": 5, "datasets": [{"name": "cifake", "adapter": "cifake", "root": roots["cifake"]}]}
    a, b = build_from_config(config), build_from_config(config)
    assert {k: sorted(r.image_path for r in v) for k, v in a.items()} == {
        k: sorted(r.image_path for r in v) for k, v in b.items()
    }


def test_demo_section_is_kept_out_of_training(tmp_path, roots):
    """The whole point: demo data lands in its own split, not train/val/test."""
    demo_root = str(tmp_path / "coco" / "val2017")
    for i in range(6):
        write_image(os.path.join(demo_root, "%d.jpg" % i), i)

    splits = build_from_config({
        "seed": 0,
        "datasets": [{"name": "cifake", "adapter": "cifake", "root": roots["cifake"]}],
        "demo": [{"name": "coco_val2017", "adapter": "folder", "root": demo_root,
                  "fixed_label": 0}],
    })
    assert "demo" in splits and len(splits["demo"]) == 6
    assert all(r.label == 0 for r in splits["demo"])
    for name in ("train", "val", "test"):
        assert not any("val2017" in r.image_path for r in splits[name])


def test_protected_data_listed_as_trainable_is_refused(tmp_path, roots):
    """Putting the demo subset under 'datasets' must be a hard error."""
    demo_root = str(tmp_path / "coco" / "val2017")
    for i in range(4):
        write_image(os.path.join(demo_root, "%d.jpg" % i), i)
    config = {
        "datasets": [
            {"name": "cifake", "adapter": "cifake", "root": roots["cifake"]},
            {"name": "coco_val2017", "adapter": "folder", "root": demo_root, "fixed_label": 0},
        ],
    }
    with pytest.raises(ProtectedDataError, match="11.B"):
        build_from_config(config)


def test_custom_ratios_and_stratification(roots):
    splits = build_from_config({
        "seed": 0,
        "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
        "stratify_keys": ["dataset", "label"],
        "datasets": [
            {"name": "cifake", "adapter": "cifake", "root": roots["cifake"]},
            {"name": "wildfake", "adapter": "wildfake", "root": roots["wildfake"]},
        ],
    })
    assert len(splits["train"]) > len(splits["val"])


def test_load_config_from_json(tmp_path, roots):
    path = write_config(tmp_path, {
        "seed": 2,
        "datasets": [{"name": "cifake", "adapter": "cifake", "root": roots["cifake"]}],
    })
    config = load_config(path)
    assert config["seed"] == 2 and config["ratios"]["train"] == 0.7  # default filled in
    assert set(build_from_config(path)) == {"train", "val", "test"}


def test_missing_config_file():
    with pytest.raises(DatasetConfigError, match="not found"):
        load_config("/nonexistent/data.json")


def test_config_without_datasets_is_rejected(tmp_path):
    path = write_config(tmp_path, {"seed": 0, "datasets": []})
    with pytest.raises(DatasetConfigError, match="no datasets"):
        load_config(path)


def test_unknown_top_level_key_is_rejected(tmp_path, roots):
    path = write_config(tmp_path, {
        "datasets": [{"adapter": "cifake", "root": roots["cifake"]}],
        "learning_rate": 0.1,
    })
    with pytest.raises(DatasetConfigError, match="unknown config key"):
        load_config(path)


def test_entry_without_root_is_rejected(tmp_path):
    path = write_config(tmp_path, {"datasets": [{"name": "x", "adapter": "folder"}]})
    with pytest.raises(DatasetConfigError, match="no 'root'"):
        load_config(path)


def test_unknown_entry_key_is_rejected(roots):
    with pytest.raises(DatasetConfigError, match="unknown key"):
        build_from_config({
            "datasets": [{"adapter": "cifake", "root": roots["cifake"], "batch_size": 4}]
        })


def test_demo_split_name_must_be_recognised(roots):
    with pytest.raises(DatasetConfigError, match="protected-data guard"):
        build_from_config(
            {"datasets": [{"adapter": "cifake", "root": roots["cifake"]}]},
            demo_split_name="heldout",
        )


def test_bad_fixed_label_is_rejected(tmp_path, roots):
    demo_root = str(tmp_path / "misc")
    write_image(os.path.join(demo_root, "a.jpg"), 1)
    with pytest.raises(DatasetConfigError, match="fixed_label"):
        build_from_config({
            "datasets": [{"adapter": "cifake", "root": roots["cifake"]}],
            "demo": [{"adapter": "folder", "root": demo_root, "fixed_label": 7}],
        })


def test_shipped_example_config_is_parseable():
    """configs/data.example.yaml must stay valid as the schema evolves."""
    yaml = pytest.importorskip("yaml")
    with open("configs/data.example.yaml") as fh:
        raw = yaml.safe_load(fh)
    assert set(raw) <= {"seed", "ratios", "stratify_keys", "group_keys", "datasets", "demo"}
    assert raw["datasets"] and raw["demo"]
    demo_names = {entry["name"] for entry in raw["demo"]}
    assert demo_names == {"coco_val2017", "dalle_advanced"}
