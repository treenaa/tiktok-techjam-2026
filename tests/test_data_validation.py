"""`validate_splits` -- the pre-training gate."""

from __future__ import annotations

import pytest

from src.data import (
    LeakageError,
    ManifestRecord,
    split_records,
    validate_splits,
    write_manifest,
)
from src.data.validation import (
    find_derivative_leakage,
    find_forbidden_combinations,
    normalized_stem,
)


def make_records(n_sources=40, views=("clean", "jpeg_70"), dataset="demo", offset=0):
    """Independent record blocks; ``offset`` keeps source_ids from colliding."""
    records = []
    for i in range(offset, offset + n_sources):
        for view in views:
            records.append(
                ManifestRecord(
                    image_path="%s/img%03d_%s.png" % (dataset, i, view),
                    label=i % 2,
                    source_id="img%03d" % i,
                    dataset=dataset,
                    generator="gen_%d" % (i % 3) if i % 2 else "",
                )
            )
    return records


@pytest.fixture
def clean_splits():
    return split_records(make_records(), seed=0)


# -- the happy path --------------------------------------------------------
def test_a_clean_split_passes(clean_splits):
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"]
    )
    assert report.ok
    assert "PASSED" in report.summary()
    assert set(report.stats["splits"]) == {"train", "val", "test"}


def test_accepts_manifest_paths(tmp_path, clean_splits):
    paths = {}
    for name, records in clean_splits.items():
        path = str(tmp_path / ("%s.csv" % name))
        write_manifest(records, path)
        paths[name] = path
    assert validate_splits(paths["train"], paths["val"], paths["test"]).ok


def test_extra_splits_are_supported(clean_splits):
    report = validate_splits(
        clean_splits["train"],
        clean_splits["val"],
        clean_splits["test"],
        extra_splits={"holdout": make_records(5, dataset="other", offset=900)},
        check_derivatives=False,
    )
    assert "holdout" in report.stats["splits"]


# -- source_id leakage -----------------------------------------------------
def test_duplicate_source_id_across_splits_is_detected(clean_splits):
    leaked = clean_splits["train"][0]
    clean_splits["test"].append(leaked.with_fields(image_path="other/copy.png"))
    with pytest.raises(LeakageError, match="source_id"):
        validate_splits(clean_splits["train"], clean_splits["val"], clean_splits["test"])


def test_leakage_message_names_splits_and_ids(clean_splits):
    leaked = clean_splits["train"][0]
    clean_splits["test"].append(leaked.with_fields(image_path="other/copy.png"))
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        raise_on_failure=False,
    )
    assert not report.ok
    message = report.summary()
    assert "train" in message and "test" in message and leaked.source_id in message
    assert "FAILED" in message


# -- path leakage ----------------------------------------------------------
def test_duplicate_path_across_splits_is_detected(clean_splits):
    duplicated = clean_splits["train"][0]
    clean_splits["test"].append(duplicated.with_fields(source_id="different_id"))
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        raise_on_failure=False,
    )
    assert "path_overlap" in report.problems


def test_duplicate_path_within_one_split_is_detected():
    records = make_records(20)
    records.append(records[0])
    report = validate_splits(records, make_records(4, dataset="v", offset=500),
                             make_records(4, dataset="t", offset=600),
                             check_derivatives=False, raise_on_failure=False)
    assert any("duplicate path" in m for m in report.problems.get("structure", []))


# -- derivative leakage inferred from filenames ---------------------------
def test_derivative_leakage_is_caught_even_when_source_ids_disagree():
    """The dangerous case: a bad source_id policy hides a real overlap."""
    train = [ManifestRecord("d/cat_017.png", 1, "cat_017", generator="g")]
    train.append(ManifestRecord("d/dog_001.png", 0, "dog_001"))
    test = [ManifestRecord("d/cat_017_jpeg_30.png", 1, "cat_017_jpeg_30", generator="g")]
    test.append(ManifestRecord("d/fox_002.png", 0, "fox_002"))
    # source_ids look disjoint...
    assert not ({r.source_id for r in train} & {r.source_id for r in test})
    # ...but the filenames give it away.
    with pytest.raises(LeakageError, match="normalized stem"):
        validate_splits(train, None, test)


def test_derivative_check_can_be_disabled():
    train = [
        ManifestRecord("d/cat_017.png", 1, "a", generator="g"),
        ManifestRecord("d/dog.png", 0, "b"),
    ]
    test = [
        ManifestRecord("d/cat_017_jpeg_30.png", 1, "c", generator="g"),
        ManifestRecord("d/fox.png", 0, "d"),
    ]
    assert validate_splits(train, None, test, check_derivatives=False).ok


def test_normalized_stem_collapses_transform_suffixes():
    assert normalized_stem("/x/Cat_017_jpeg_30.png") == "cat_017"
    assert normalized_stem("/x/cat_017.png") == "cat_017"


def test_find_derivative_leakage_returns_messages_not_exceptions():
    splits = {
        "train": [ManifestRecord("d/a.png", 0, "a")],
        "test": [ManifestRecord("d/a_blur_1.0.png", 0, "b")],
    }
    problems = find_derivative_leakage(splits)
    assert len(problems) == 1 and "train" in problems[0] and "test" in problems[0]


# -- forbidden combinations ------------------------------------------------
def test_forbidden_generator_in_test_is_detected(clean_splits):
    present = sorted({r.generator for r in clean_splits["test"] if r.generator})
    assert present, "fixture must place at least one generator in test"
    with pytest.raises(LeakageError, match="forbidden"):
        validate_splits(
            clean_splits["train"], clean_splits["val"], clean_splits["test"],
            forbidden={"test": {"generator": [present[0]]}},
        )


def test_forbidden_dataset_rule(clean_splits):
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        forbidden={"train": {"dataset": ["demo"]}},
        raise_on_failure=False,
    )
    assert "forbidden_combination" in report.problems


def test_forbidden_rules_that_hold_are_silent(clean_splits):
    assert validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        forbidden={"test": {"generator": ["never_used_generator"]}},
    ).ok


def test_forbidden_rule_naming_an_unknown_split_is_reported(clean_splits):
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        forbidden={"nosuchsplit": {"dataset": ["demo"]}},
        raise_on_failure=False,
    )
    assert any("unknown split" in m for m in report.problems["forbidden_combination"])


def test_find_forbidden_combinations_directly():
    splits = {"test": [ManifestRecord("a.png", 1, "a", generator="sdxl")]}
    assert find_forbidden_combinations(splits, {"test": {"generator": ["sdxl"]}})
    assert not find_forbidden_combinations(splits, {"test": {"generator": ["other"]}})


# -- structural problems ---------------------------------------------------
def test_empty_split_is_reported():
    with pytest.raises(LeakageError, match="is empty"):
        validate_splits(make_records(10), [], make_records(4, dataset="t", offset=600))


def test_single_class_split_is_reported():
    real_only = [ManifestRecord("r/%d.png" % i, 0, "r%d" % i) for i in range(4)]
    with pytest.raises(LeakageError, match="only label"):
        validate_splits(make_records(10), real_only, make_records(4, dataset="t", offset=600))


def test_single_class_check_can_be_relaxed():
    real_only = [ManifestRecord("r/%d.png" % i, 0, "r%d" % i) for i in range(4)]
    assert validate_splits(
        make_records(10), real_only, make_records(4, dataset="t", offset=600),
        require_both_labels=False,
    ).ok


def test_missing_files_are_reported_when_requested(tmp_path):
    records = [ManifestRecord(str(tmp_path / "gone.png"), 0, "a"),
               ManifestRecord(str(tmp_path / "gone2.png"), 1, "b")]
    with pytest.raises(LeakageError, match="missing file"):
        validate_splits(records, None, None, check_paths_exist=True,
                        require_both_labels=False, check_derivatives=False)


# -- API behaviour ---------------------------------------------------------
def test_all_problems_are_reported_not_just_the_first():
    """One run should surface every category, so a fix cycle is not iterative."""
    shared = ManifestRecord("d/img000.png", 1, "img000", dataset="demo", generator="gen_1")
    train = [shared, ManifestRecord("d/img001.png", 0, "img001", dataset="demo")]
    test = [shared, ManifestRecord("d/img002.png", 0, "img002", dataset="demo")]
    report = validate_splits(
        train, None, test,
        forbidden={"test": {"generator": ["gen_1"]}},
        raise_on_failure=False,
    )
    assert {"source_id_overlap", "path_overlap", "forbidden_combination"} <= set(report.problems)


def test_raise_if_failed_is_explicit(clean_splits):
    leaked = clean_splits["train"][0]
    clean_splits["test"].append(leaked.with_fields(image_path="other/copy.png"))
    report = validate_splits(
        clean_splits["train"], clean_splits["val"], clean_splits["test"],
        raise_on_failure=False,
    )
    with pytest.raises(LeakageError):
        report.raise_if_failed()


def test_at_least_one_manifest_is_required():
    with pytest.raises(ValueError, match="at least one manifest"):
        validate_splits()


def test_rejects_non_record_input():
    with pytest.raises(TypeError, match="ManifestRecord"):
        validate_splits([{"image_path": "a.png"}], None, None)
