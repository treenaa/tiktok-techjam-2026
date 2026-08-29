"""Leakage-safe splitting: grouping, determinism, stratification, checks."""

from __future__ import annotations

import pytest

from src.data import (
    ManifestRecord,
    LeakageError,
    assert_no_path_overlap,
    assert_no_source_id_leakage,
    assign_splits,
    check_split_integrity,
    split_by_source_id,
    split_records,
    split_report,
)
from src.data.splitting import format_split_report

VIEWS = ("clean", "jpeg70", "blur_1.0", "noise_0.05")


def make_records(n_sources=60, views=VIEWS, datasets=("dsA", "dsB")):
    """``n_sources`` originals, each present as several transformed views."""
    records = []
    for i in range(n_sources):
        # label and dataset must vary independently, or "stratify by dataset
        # and label" would collapse to two strata instead of four.
        label = i % 2
        dataset = datasets[(i // 2) % len(datasets)]
        for view in views:
            records.append(
                ManifestRecord(
                    image_path="%s/img%03d_%s.png" % (dataset, i, view),
                    label=label,
                    source_id="img%03d" % i,
                    dataset=dataset,
                    generator="sdxl" if label else "",
                )
            )
    return records


# -- the core guarantee ----------------------------------------------------
def test_no_source_id_crosses_a_split():
    splits = split_records(make_records(), seed=0)
    seen = {}
    for name, members in splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name, rec.source_id


def test_all_views_of_an_original_stay_together():
    """The derivative test: every view of an image lands in one split."""
    splits = split_records(make_records(), seed=0)
    for name, members in splits.items():
        by_source = {}
        for rec in members:
            by_source.setdefault(rec.source_id, set()).add(rec.image_path)
        for source_id, paths in by_source.items():
            assert len(paths) == len(VIEWS), (name, source_id, paths)


def test_split_is_a_partition_of_the_input():
    records = make_records()
    splits = split_records(records, seed=0)
    assert sum(len(m) for m in splits.values()) == len(records)
    assert {r.image_path for m in splits.values() for r in m} == {
        r.image_path for r in records
    }


def test_ratios_are_honoured_at_group_level():
    records = make_records(n_sources=100)
    splits = split_records(records, ratios=(0.7, 0.15, 0.15), seed=0)
    groups = {n: len({r.source_id for r in m}) for n, m in splits.items()}
    assert groups == {"train": 70, "val": 15, "test": 15}


def test_ratios_need_not_sum_to_one():
    splits = split_records(make_records(100), ratios=(7, 2, 1), seed=0)
    assert {n: len({r.source_id for r in m}) for n, m in splits.items()} == {
        "train": 70, "val": 20, "test": 10
    }


def test_custom_split_names_and_counts():
    splits = split_records(
        make_records(100), ratios={"fit": 0.5, "calib": 0.2, "holdout": 0.2, "probe": 0.1}, seed=0
    )
    assert list(splits) == ["fit", "calib", "holdout", "probe"]
    assert sum(len({r.source_id for r in m}) for m in splits.values()) == 100


# -- determinism -----------------------------------------------------------
def test_same_seed_gives_the_same_split():
    records = make_records()
    a = split_records(records, seed=42)
    b = split_records(records, seed=42)
    assert {k: sorted(r.image_path for r in v) for k, v in a.items()} == {
        k: sorted(r.image_path for r in v) for k, v in b.items()
    }


def test_different_seeds_give_different_splits():
    records = make_records()
    a = split_records(records, seed=1)["test"]
    b = split_records(records, seed=2)["test"]
    assert {r.source_id for r in a} != {r.source_id for r in b}


def test_split_is_independent_of_input_order():
    """Shuffled input must not change the assignment -- reproducible manifests."""
    records = make_records()
    a = split_records(records, seed=7)
    b = split_records(list(reversed(records)), seed=7)
    assert {k: sorted(r.image_path for r in v) for k, v in a.items()} == {
        k: sorted(r.image_path for r in v) for k, v in b.items()
    }


# -- stratification --------------------------------------------------------
def test_stratification_by_label_preserves_balance():
    """Each split carries both labels in proportion, up to one group of rounding.

    Exact per-stratum equality is not achievable when a stratum's share is
    fractional (50 groups x 0.15 = 7.5), and the splitter prioritises hitting
    the *global* ratios -- so the guarantee is balance within +-1 group.
    """
    records = make_records(n_sources=100)
    splits = split_records(records, seed=0, stratify_keys=("label",))
    for name, members in splits.items():
        groups = {0: set(), 1: set()}
        for rec in members:
            groups[rec.label].add(rec.source_id)
        counts = [len(groups[0]), len(groups[1])]
        assert min(counts) > 0, name
        assert max(counts) - min(counts) <= 1, (name, counts)


def test_stratification_by_dataset_and_label():
    """All four (dataset, label) strata appear in every split, evenly."""
    records = make_records(n_sources=100)
    splits = split_records(records, seed=0, stratify_keys=("dataset", "label"))
    for name, members in splits.items():
        groups = {}
        for rec in members:
            groups.setdefault((rec.dataset, rec.label), set()).add(rec.source_id)
        assert len(groups) == 4, (name, sorted(groups))
        counts = [len(v) for v in groups.values()]
        assert max(counts) - min(counts) <= 1, (name, groups)


def test_stratification_by_generator():
    records = make_records(n_sources=80)
    splits = split_records(records, seed=0, stratify_keys=("generator",))
    for members in splits.values():
        assert {r.generator for r in members} == {"", "sdxl"}


def test_stratification_can_be_disabled():
    splits = split_records(make_records(), seed=0, stratify_keys=None)
    assert sum(len(m) for m in splits.values()) == len(make_records())


def test_a_source_id_spanning_both_labels_stays_intact():
    """A real image and its tampered derivative share a source_id (SID_Set)."""
    records = []
    for i in range(40):
        sid = "img%03d" % i
        records.append(ManifestRecord("real/%s.png" % sid, 0, sid))
        records.append(ManifestRecord("tampered/%s_tampered.png" % sid, 1, sid))
    splits = split_records(records, seed=0)
    for members in splits.values():
        by_source = {}
        for rec in members:
            by_source.setdefault(rec.source_id, []).append(rec.label)
        assert all(sorted(v) == [0, 1] for v in by_source.values())


# -- grouping --------------------------------------------------------------
def test_grouping_by_dataset_and_source_id_separates_colliding_ids():
    """Identical source_ids in two datasets are different originals."""
    records = []
    for dataset in ("dsA", "dsB"):
        for i in range(30):
            records.append(
                ManifestRecord("%s/img%03d.png" % (dataset, i), i % 2, "img%03d" % i, dataset=dataset)
            )
    splits = split_records(records, seed=0, group_keys=("dataset", "source_id"))
    assert_no_source_id_leakage(splits, group_keys=("dataset", "source_id"))
    # With the default grouping the two datasets are forced together instead.
    default = split_records(records, seed=0)
    for members in default.values():
        by_source = {}
        for rec in members:
            by_source.setdefault(rec.source_id, set()).add(rec.dataset)
        assert all(v == {"dsA", "dsB"} for v in by_source.values())


def test_empty_group_keys_are_rejected():
    with pytest.raises(ValueError, match="group_keys"):
        split_records(make_records(), group_keys=())


# -- explicit leakage checks ----------------------------------------------
def test_assert_no_source_id_leakage_detects_an_injected_overlap():
    splits = split_records(make_records(), seed=0)
    leaked = splits["train"][0]
    splits["test"].append(leaked.with_fields(image_path="copy_" + leaked.image_path))
    with pytest.raises(LeakageError, match="source_id leakage"):
        assert_no_source_id_leakage(splits)


def test_leakage_error_names_the_offending_splits_and_ids():
    rec = ManifestRecord("a.png", 0, "shared")
    splits = {"train": [rec], "test": [ManifestRecord("b.png", 0, "shared")]}
    with pytest.raises(LeakageError) as excinfo:
        assert_no_source_id_leakage(splits)
    message = str(excinfo.value)
    assert "train" in message and "test" in message and "shared" in message


def test_assert_no_path_overlap_detects_a_duplicated_file():
    rec = ManifestRecord("a.png", 0, "a")
    splits = {"train": [rec], "test": [ManifestRecord("a.png", 0, "b")]}
    with pytest.raises(LeakageError, match="duplicate image paths"):
        assert_no_path_overlap(splits)


def test_verify_runs_by_default_inside_split_records():
    """The splitter checks its own output -- a regression cannot pass silently."""
    splits = split_records(make_records(), seed=0, verify=True)
    assert_no_source_id_leakage(splits)


def test_check_split_integrity_reports_a_lost_record():
    records = make_records()
    splits = split_records(records, seed=0)
    splits["train"].pop()
    with pytest.raises(LeakageError, match="not a partition"):
        check_split_integrity(splits, original=records)


def test_check_split_integrity_flags_empty_splits():
    records = [ManifestRecord("a%d.png" % i, i % 2, "s%d" % i) for i in range(2)]
    splits = split_records(records, ratios=(0.98, 0.01, 0.01), seed=0, verify=False)
    with pytest.raises(LeakageError, match="empty split"):
        check_split_integrity(splits)


def test_check_split_integrity_returns_a_report():
    records = make_records()
    report = check_split_integrity(split_records(records, seed=0), original=records)
    assert report["n_images"] == len(records)
    assert set(report["splits"]) == {"train", "val", "test"}
    assert report["splits"]["train"]["n_groups"] == 42


def test_split_report_contents():
    splits = split_records(make_records(n_sources=100), seed=0)
    report = split_report(splits)
    train = report["splits"]["train"]
    assert train["n_images"] == 280 and train["n_groups"] == 70
    assert train["n_real"] == train["n_aigc"] == 140
    assert abs(train["fraction"] - 0.7) < 1e-9
    assert set(train["by_dataset"]) == {"dsA", "dsB"}
    assert "train" in format_split_report(report)


# -- assign_splits ---------------------------------------------------------
def test_assign_splits_labels_records_without_mutating_the_input():
    records = make_records()
    assigned = assign_splits(records, seed=0)
    assert len(assigned) == len(records)
    assert all(rec.split == "" for rec in records), "input must not be mutated"
    assert {rec.split for rec in assigned} == {"train", "val", "test"}
    # Order is preserved, so the two lists correspond element-wise.
    assert [a.image_path for a in assigned] == [r.image_path for r in records]


def test_assign_splits_agrees_with_split_records():
    records = make_records()
    assigned = assign_splits(records, seed=11)
    splits = split_records(records, seed=11)
    for name, members in splits.items():
        assert {r.image_path for r in members} == {
            a.image_path for a in assigned if a.split == name
        }


def test_assigned_source_ids_never_span_splits():
    assigned = assign_splits(make_records(), seed=3)
    by_source = {}
    for rec in assigned:
        by_source.setdefault(rec.source_id, set()).add(rec.split)
    assert all(len(v) == 1 for v in by_source.values())


# -- misc ------------------------------------------------------------------
def test_alias_matches_split_records():
    records = make_records()
    assert {k: [r.image_path for r in v] for k, v in split_by_source_id(records, seed=0).items()} == {
        k: [r.image_path for r in v] for k, v in split_records(records, seed=0).items()
    }


def test_splitting_an_empty_collection_fails():
    with pytest.raises(ValueError, match="empty"):
        split_records([])


@pytest.mark.parametrize("ratios", [(0, 0, 0), {"train": -1.0}])
def test_invalid_ratios_are_rejected(ratios):
    with pytest.raises(ValueError):
        split_records(make_records(), ratios=ratios)
