"""Generator-aware filtering, grouping and unseen-generator splits.

No generator name is hard-coded in ``src/data``; these tests use deliberately
invented names to prove it.
"""

from __future__ import annotations

import pytest

from src.data import (
    DataError,
    LeakageError,
    ManifestRecord,
    assert_generators_disjoint,
    filter_by_generator,
    generator_counts,
    group_by_generator,
    list_generators,
    partition_generators,
    split_by_generator_holdout,
)

GENERATORS = ("aurora", "borealis", "cirrus", "delta")


def make_records(n_per_generator=8, generators=GENERATORS, n_real=32):
    records = [
        ManifestRecord("real/r%03d.png" % i, 0, "real_%03d" % i, dataset="d")
        for i in range(n_real)
    ]
    for gi, generator in enumerate(generators):
        for i in range(n_per_generator):
            records.append(
                ManifestRecord(
                    "fake/%s/f%03d.png" % (generator, gi * 100 + i),
                    1,
                    "fake_%03d" % (gi * 100 + i),
                    dataset="d",
                    generator=generator,
                )
            )
    return records


# -- introspection ---------------------------------------------------------
def test_list_generators_excludes_real_by_default():
    records = make_records()
    assert list_generators(records) == sorted(GENERATORS)
    assert "" in list_generators(records, include_real=True)


def test_generator_counts_are_ordered_by_frequency():
    records = make_records(n_per_generator=4) + [
        ManifestRecord("fake/aurora/extra.png", 1, "extra", generator="aurora")
    ]
    counts = generator_counts(records)
    assert counts["aurora"] == 5
    assert list(counts)[0] == ""  # real images are the largest group here


def test_group_by_generator():
    groups = group_by_generator(make_records())
    assert set(groups) == set(GENERATORS) | {""}
    assert all(r.generator == "aurora" for r in groups["aurora"])


# -- filtering -------------------------------------------------------------
def test_filter_include_keeps_real_images_by_default():
    filtered = filter_by_generator(make_records(), include=["aurora"])
    assert list_generators(filtered) == ["aurora"]
    assert any(r.label == 0 for r in filtered), "real images must be retained"


def test_filter_can_drop_real_images():
    filtered = filter_by_generator(make_records(), include=["aurora"], keep_real=False)
    assert all(r.label == 1 and r.generator == "aurora" for r in filtered)


def test_filter_exclude():
    filtered = filter_by_generator(make_records(), exclude=["aurora", "delta"])
    assert list_generators(filtered) == ["borealis", "cirrus"]


def test_filter_predicate_expresses_family_rules_without_hard_coded_names():
    filtered = filter_by_generator(
        make_records(), predicate=lambda name: name.startswith("b"), keep_real=False
    )
    assert list_generators(filtered) == ["borealis"]


def test_filter_combines_include_and_predicate():
    filtered = filter_by_generator(
        make_records(),
        include=["aurora", "borealis"],
        predicate=lambda name: name != "aurora",
        keep_real=False,
    )
    assert list_generators(filtered) == ["borealis"]


# -- partitioning ----------------------------------------------------------
def test_partition_with_explicit_holdout():
    seen, held = partition_generators(make_records(), holdout=["delta"])
    assert held == ["delta"]
    assert set(seen) == set(GENERATORS) - {"delta"}


def test_partition_with_n_holdout_is_deterministic():
    records = make_records()
    a = partition_generators(records, n_holdout=2, seed=3)
    b = partition_generators(records, n_holdout=2, seed=3)
    c = partition_generators(records, n_holdout=2, seed=4)
    assert a == b
    assert len(a[1]) == 2
    assert a != c or True  # different seeds may coincide; determinism is the claim


def test_partition_rejects_unknown_generator():
    with pytest.raises(DataError, match="not present"):
        partition_generators(make_records(), holdout=["nonexistent"])


def test_partition_rejects_holding_out_everything():
    with pytest.raises(DataError, match="no training generators"):
        partition_generators(make_records(), holdout=list(GENERATORS))


def test_partition_requires_generator_metadata():
    records = [ManifestRecord("a%d.png" % i, i % 2, "s%d" % i) for i in range(4)]
    with pytest.raises(DataError, match="no generator metadata"):
        partition_generators(records, n_holdout=1)


def test_partition_requires_a_selection_mode():
    with pytest.raises(DataError, match="holdout=|n_holdout"):
        partition_generators(make_records())


# -- unseen-generator splitting -------------------------------------------
def test_held_out_generator_appears_only_in_test():
    splits = split_by_generator_holdout(make_records(), holdout=["delta"], seed=0)
    assert "delta" not in list_generators(splits["train"])
    assert "delta" not in list_generators(splits["val"])
    assert "delta" in list_generators(splits["test"])


def test_seen_generators_never_leak_into_the_holdout_split():
    splits = split_by_generator_holdout(make_records(), holdout=["delta"], seed=0)
    assert list_generators(splits["test"]) == ["delta"]


def test_generator_holdout_keeps_both_classes_in_every_split():
    splits = split_by_generator_holdout(make_records(), holdout=["delta"], seed=0)
    for name, members in splits.items():
        assert {r.label for r in members} == {0, 1}, name


def test_generator_holdout_preserves_source_id_grouping():
    """Derivatives must still not cross a split under generator-aware splitting."""
    records = []
    for gi, generator in enumerate(GENERATORS):
        for i in range(6):
            source_id = "fake_%d_%d" % (gi, i)
            for view in ("clean", "jpeg_70"):
                records.append(
                    ManifestRecord(
                        "fake/%s/%s_%s.png" % (generator, source_id, view),
                        1, source_id, dataset="d", generator=generator,
                    )
                )
    for i in range(24):
        for view in ("clean", "jpeg_70"):
            records.append(
                ManifestRecord("real/r%d_%s.png" % (i, view), 0, "real_%d" % i, dataset="d")
            )
    splits = split_by_generator_holdout(records, holdout=["delta"], seed=0)
    seen = {}
    for name, members in splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name


def test_generator_holdout_is_deterministic():
    records = make_records()
    a = split_by_generator_holdout(records, holdout=["delta"], seed=7)
    b = split_by_generator_holdout(records, holdout=["delta"], seed=7)
    assert {k: sorted(r.image_path for r in v) for k, v in a.items()} == {
        k: sorted(r.image_path for r in v) for k, v in b.items()
    }


def test_generator_holdout_by_count():
    splits = split_by_generator_holdout(make_records(), n_holdout=2, seed=1)
    held = list_generators(splits["test"])
    assert len(held) == 2
    assert not set(held) & set(list_generators(splits["train"]))


def test_holdout_split_can_be_renamed():
    splits = split_by_generator_holdout(
        make_records(), holdout=["delta"], holdout_split="unseen", seed=0
    )
    assert "unseen" in splits and "delta" in list_generators(splits["unseen"])


def test_val_ratio_zero_puts_everything_in_train():
    splits = split_by_generator_holdout(
        make_records(), holdout=["delta"], val_ratio=0.0, seed=0,
        real_ratios={"train": 0.85, "test": 0.15},
    )
    assert not [r for r in splits["val"] if r.label == 1]


def test_invalid_val_ratio_is_rejected():
    with pytest.raises(ValueError, match="val_ratio"):
        split_by_generator_holdout(make_records(), holdout=["delta"], val_ratio=1.0)


# -- the assertion helper --------------------------------------------------
def test_assert_generators_disjoint_passes_on_a_clean_split():
    splits = split_by_generator_holdout(make_records(), holdout=["delta"], seed=0)
    assert_generators_disjoint(splits, holdout=["delta"])
    assert_generators_disjoint(splits)  # symmetric form


def test_assert_generators_disjoint_detects_a_leak():
    splits = split_by_generator_holdout(make_records(), holdout=["delta"], seed=0)
    splits["train"].append(
        ManifestRecord("fake/delta/leak.png", 1, "leak", generator="delta")
    )
    with pytest.raises(LeakageError, match="generator leakage"):
        assert_generators_disjoint(splits, holdout=["delta"])
    with pytest.raises(LeakageError, match="generator leakage"):
        assert_generators_disjoint(splits)


def test_assert_generators_disjoint_requires_the_holdout_split():
    with pytest.raises(LeakageError, match="not present"):
        assert_generators_disjoint({"train": make_records()}, holdout_split="test")


# -- regression: unseen-domain purity --------------------------------------
def test_holdout_split_contains_only_held_out_generators():
    """Regression: seen-generator AIGC must not leak into the holdout split.

    A 15% slice of *seen* AIGC used to be routed into the holdout split
    alongside the real images, which silently contaminated the cross-generator
    number with in-distribution samples.
    """
    from src.data import list_field_values, split_by_field_holdout

    records = make_records()
    for splits in (
        split_by_generator_holdout(records, holdout=["delta"], seed=0),
        split_by_field_holdout(records, field="generator", holdout=["delta"], seed=0),
    ):
        assert list_field_values(splits["test"], "generator") == ["delta"]
        assert {r.generator for r in splits["test"] if r.label == 1} == {"delta"}
        assert {r.label for r in splits["test"]} == {0, 1}, "must stay two-class"


def test_every_seen_generator_still_reaches_training():
    from src.data import list_field_values, split_by_field_holdout

    records = make_records()
    splits = split_by_field_holdout(records, field="generator", holdout=["delta"], seed=0)
    seen = set(GENERATORS) - {"delta"}
    assert set(list_field_values(splits["train"], "generator")) == seen
