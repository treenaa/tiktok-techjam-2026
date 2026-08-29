"""Rule 11.B: demonstration-only data must never reach training.

The competition's demo subset is COCO val2017 (4998 real) and DALL-E Advanced
(8843 AIGC).  These tests pin that the guard is on *by default* -- the failure
mode being defended against is a teammate forgetting to enable it.
"""

from __future__ import annotations

import pytest

from src.data import (
    DEMO_SPLIT_NAMES,
    LeakageError,
    ManifestRecord,
    ProtectedDataError,
    assert_not_trainable,
    classify_protected,
    find_protected_records,
    partition_protected,
    protected_report,
    split_records,
    validate_splits,
    write_manifest,
)


def demo_records(n=6):
    records = [
        ManifestRecord("/data/coco/val2017/%012d.jpg" % i, 0, "coco_%d" % i,
                       dataset="coco_val2017")
        for i in range(n)
    ]
    records += [
        ManifestRecord("/data/dalle_advanced/img_%d.png" % i, 1, "dalle_%d" % i,
                       dataset="dalle_advanced")
        for i in range(n)
    ]
    return records


def normal_records(n=20, offset=0):
    """Distinct blocks; ``offset`` keeps splits from sharing source_ids."""
    return [
        ManifestRecord("/data/cifake/train/%s/%d.png" % ("REAL" if i % 2 == 0 else "FAKE", i),
                       i % 2, "cifake_%d" % i, dataset="cifake",
                       generator="sdxl" if i % 2 else "")
        for i in range(offset, offset + n)
    ]


# -- recognition -----------------------------------------------------------
@pytest.mark.parametrize(
    "record,expected",
    [
        (ManifestRecord("/d/coco/val2017/1.jpg", 0, "a"), "coco_val2017"),
        (ManifestRecord("/d/x.jpg", 0, "a", dataset="COCO val2017"), "coco_val2017"),
        (ManifestRecord("/d/x.jpg", 0, "a", dataset="coco_val_2017"), "coco_val2017"),
        (ManifestRecord("/d/dalle_advanced/1.png", 1, "a"), "dalle_advanced"),
        (ManifestRecord("/d/x.png", 1, "a", dataset="DALL-E Advanced"), "dalle_advanced"),
        (ManifestRecord("/d/x.png", 1, "a", dataset="dalle_advanced"), "dalle_advanced"),
        (ManifestRecord("/d/cifake/1.png", 0, "a", dataset="cifake"), None),
        (ManifestRecord("/d/wildfake/fake/sdxl/1.png", 1, "a", dataset="wildfake"), None),
    ],
)
def test_recognition_by_path_or_dataset_column(record, expected):
    assert classify_protected(record) == expected


def test_recognition_survives_a_wrong_dataset_column():
    """A mislabelled dataset column must not defeat the guard."""
    record = ManifestRecord("/data/coco/val2017/9.jpg", 0, "a", dataset="my_training_data")
    assert classify_protected(record) == "coco_val2017"


def test_windows_style_paths_are_recognised():
    record = ManifestRecord(r"C:\data\coco\val2017\1.jpg", 0, "a")
    assert classify_protected(record) == "coco_val2017"


def test_unicode_dalle_spelling_is_recognised():
    assert classify_protected(ManifestRecord("/d/x.png", 1, "a", dataset="DALL·E Advanced"))


def test_ordinary_datasets_are_never_flagged():
    assert not find_protected_records(normal_records())


# -- the guard -------------------------------------------------------------
def test_assert_not_trainable_raises_on_demo_data():
    with pytest.raises(ProtectedDataError, match="must NEVER be used for training"):
        assert_not_trainable(demo_records())


def test_assert_not_trainable_message_is_actionable():
    with pytest.raises(ProtectedDataError) as excinfo:
        assert_not_trainable(demo_records())
    message = str(excinfo.value)
    assert "11.B" in message
    assert "coco_val2017" in message and "dalle_advanced" in message
    assert "val2017" in message  # names an offending file


def test_assert_not_trainable_passes_on_clean_data():
    assert_not_trainable(normal_records()) is None


def test_partition_protected_separates_the_subset():
    trainable, protected = partition_protected(normal_records() + demo_records())
    assert len(trainable) == 20 and len(protected) == 12
    assert not find_protected_records(trainable)


def test_protected_report_flags_count_drift():
    report = protected_report(demo_records(n=6))
    assert report["n_protected"] == 12
    assert "count_mismatch" in report["subsets"]["coco_val2017"]
    assert "4998" in report["subsets"]["coco_val2017"]["count_mismatch"]


def test_protected_report_flags_label_drift():
    """COCO val2017 is real; anything labelled AIGC there is a mistake."""
    mislabelled = [ManifestRecord("/d/coco/val2017/1.jpg", 1, "a", dataset="coco_val2017")]
    report = protected_report(mislabelled)
    assert "label_mismatch" in report["subsets"]["coco_val2017"]


# -- integration with validate_splits (the part that must be default-on) ---
def test_demo_data_in_training_fails_by_default():
    """The regression this module exists for."""
    splits = split_records(demo_records(n=10), seed=0)
    with pytest.raises(LeakageError, match="demonstration-only"):
        validate_splits(splits["train"], splits["val"], splits["test"])


def test_demo_data_in_val_or_test_also_fails():
    """val/test drive model selection and the final number -- equally forbidden."""
    for split_position in ("val", "test"):
        kwargs = {"train_manifest": normal_records()}
        kwargs["%s_manifest" % split_position] = demo_records(n=4)
        other = "test" if split_position == "val" else "val"
        kwargs["%s_manifest" % other] = normal_records(6)
        with pytest.raises(LeakageError, match="demonstration-only"):
            validate_splits(check_derivatives=False, **kwargs)


def test_error_names_the_rule_and_the_subset():
    splits = split_records(demo_records(n=10), seed=0)
    report = validate_splits(
        splits["train"], splits["val"], splits["test"], raise_on_failure=False
    )
    assert "protected_data" in report.problems
    joined = " ".join(report.problems["protected_data"])
    assert "11.B" in joined and "coco_val2017" in joined


def test_demo_data_is_allowed_in_a_demo_split():
    report = validate_splits(
        normal_records(20),
        normal_records(6, offset=100),
        normal_records(6, offset=200),
        extra_splits={"demo": demo_records()},
        check_derivatives=False,
    )
    assert report.ok


@pytest.mark.parametrize("name", DEMO_SPLIT_NAMES)
def test_every_demo_split_alias_is_accepted(name):
    report = validate_splits(
        normal_records(20), normal_records(6, offset=100), normal_records(6, offset=200),
        extra_splits={name: demo_records()},
        check_derivatives=False,
    )
    assert report.ok


def test_opt_out_requires_an_explicit_flag():
    """Escape hatch exists, but must be deliberate and visible at the call site."""
    splits = split_records(demo_records(n=10), seed=0)
    report = validate_splits(
        splits["train"], splits["val"], splits["test"], allow_protected=True
    )
    assert report.ok


def test_guard_works_through_manifest_files(tmp_path):
    """Paths are the only signal when the dataset column is absent."""
    records = [
        ManifestRecord("/data/coco/val2017/%d.jpg" % i, 0, "c%d" % i) for i in range(4)
    ] + [ManifestRecord("/data/gen/%d.png" % i, 1, "g%d" % i) for i in range(4)]
    path = str(tmp_path / "train.csv")
    write_manifest(records, path)
    with pytest.raises(LeakageError, match="demonstration-only"):
        validate_splits(path, None, None, require_both_labels=False)


def test_report_stats_include_protected_summary():
    report = validate_splits(
        normal_records(20), normal_records(6, offset=100), normal_records(6, offset=200),
        check_derivatives=False,
    )
    assert report.stats["protected"]["n_protected"] == 0
