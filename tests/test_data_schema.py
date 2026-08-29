"""Records, labels, validation and manifest I/O."""

from __future__ import annotations

import json
import os

import pytest

from src.data import (
    LABEL_AIGC,
    LABEL_REAL,
    DataError,
    ManifestRecord,
    describe_records,
    filter_records,
    label_counts,
    merge_manifests,
    read_manifest,
    records_from_dataframe,
    records_to_dataframe,
    validate_records,
    write_manifest,
)


def make_records(n=6):
    return [
        ManifestRecord(
            image_path="images/img%03d.png" % i,
            label=i % 2,
            source_id="img%03d" % i,
            dataset="demo",
            generator="sdxl" if i % 2 else "",
        )
        for i in range(n)
    ]


# -- labels ----------------------------------------------------------------
def test_binary_label_convention():
    assert (LABEL_REAL, LABEL_AIGC) == (0, 1)
    real = ManifestRecord("a.png", 0, "a")
    aigc = ManifestRecord("b.png", 1, "b")
    assert real.label_name == "real" and not real.is_aigc
    assert aigc.label_name == "aigc" and aigc.is_aigc


@pytest.mark.parametrize("bad", [2, -1, 99])
def test_non_binary_labels_are_rejected(bad):
    with pytest.raises(DataError, match="label must be"):
        ManifestRecord("a.png", bad, "a")


def test_string_labels_are_coerced():
    assert ManifestRecord("a.png", "1", "a").label == 1


def test_label_zero_survives_row_round_trip():
    """Regression: ``label=0`` is falsy and must not read as 'missing'."""
    rec = ManifestRecord.from_row({"image_path": "a.png", "label": 0, "source_id": "a"})
    assert rec.label == 0


def test_label_counts_and_describe():
    records = make_records(10)
    assert label_counts(records) == {0: 5, 1: 5}
    info = describe_records(records)
    assert info["n_images"] == 10
    assert info["n_source_ids"] == 10
    assert info["aigc_fraction"] == 0.5
    assert info["generators"] == {"sdxl": 5}


# -- source ids ------------------------------------------------------------
def test_empty_source_id_is_rejected():
    with pytest.raises(DataError, match="source_id"):
        ManifestRecord("a.png", 0, "")


def test_records_are_not_mutated_by_with_fields():
    rec = ManifestRecord("a.png", 0, "a")
    other = rec.with_fields(split="train")
    assert rec.split == "" and other.split == "train"
    assert other.source_id == "a"


# -- validation ------------------------------------------------------------
def test_duplicate_paths_are_rejected():
    records = [ManifestRecord("a.png", 0, "a"), ManifestRecord("a.png", 1, "b")]
    with pytest.raises(DataError, match="duplicate image_path"):
        validate_records(records)


def test_missing_files_are_reported_when_requested():
    with pytest.raises(DataError, match="do not exist"):
        validate_records([ManifestRecord("nope.png", 0, "a")], check_paths_exist=True)


def test_empty_collection_is_rejected():
    with pytest.raises(DataError):
        validate_records([])


# -- manifest I/O ----------------------------------------------------------
@pytest.mark.parametrize("ext", [".csv", ".json", ".jsonl", ".tsv"])
def test_manifest_round_trip(tmp_path, ext):
    records = make_records()
    path = str(tmp_path / ("manifest" + ext))
    write_manifest(records, path)
    loaded = read_manifest(path)
    assert len(loaded) == len(records)
    for original, restored in zip(records, loaded):
        assert restored.image_path == original.image_path
        assert restored.label == original.label
        assert restored.source_id == original.source_id
        assert restored.dataset == original.dataset
        assert restored.generator == original.generator


def test_manifest_csv_has_the_documented_columns(tmp_path):
    path = str(tmp_path / "m.csv")
    write_manifest(make_records(), path)
    header = open(path).readline().strip().split(",")
    assert header[:5] == ["image_path", "label", "source_id", "dataset", "generator"]
    # ``split`` is unpopulated here and therefore dropped.
    assert "split" not in header


def test_manifest_keeps_split_column_when_populated(tmp_path):
    records = [r.with_fields(split="train") for r in make_records()]
    path = str(tmp_path / "m.csv")
    write_manifest(records, path)
    assert "split" in open(path).readline()
    assert read_manifest(path, split="train")
    with pytest.raises(DataError, match="no rows with split"):
        read_manifest(path, split="test")


def test_manifest_preserves_extra_columns(tmp_path):
    records = [ManifestRecord("a.png", 0, "a", extra={"camera": "iphone"})]
    path = str(tmp_path / "m.csv")
    write_manifest(records, path)
    assert read_manifest(path)[0].extra["camera"] == "iphone"


def test_relative_paths_resolve_against_root(tmp_path):
    path = str(tmp_path / "m.csv")
    write_manifest(make_records(), path)
    loaded = read_manifest(path, root="/data/demo")
    assert loaded[0].image_path == os.path.join("/data/demo", "images/img000.png")


def test_write_relative_to_makes_manifest_portable(tmp_path):
    records = [ManifestRecord(str(tmp_path / "a" / "x.png"), 0, "x")]
    path = str(tmp_path / "m.csv")
    write_manifest(records, path, relative_to=str(tmp_path))
    assert read_manifest(path)[0].image_path == os.path.join("a", "x.png")


def test_reading_a_manifest_without_required_columns_fails(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as fh:
        json.dump([{"image_path": "a.png", "label": 1}], fh)
    with pytest.raises(DataError, match="missing required column"):
        read_manifest(path)


def test_missing_manifest_file_fails_clearly(tmp_path):
    with pytest.raises(DataError, match="not found"):
        read_manifest(str(tmp_path / "absent.csv"))


def test_writing_an_empty_manifest_is_refused(tmp_path):
    with pytest.raises(DataError, match="empty manifest"):
        write_manifest([], str(tmp_path / "m.csv"))


# -- dataframe interop / querying -----------------------------------------
def test_dataframe_round_trip():
    records = make_records()
    restored = records_from_dataframe(records_to_dataframe(records))
    assert [r.source_id for r in restored] == [r.source_id for r in records]
    assert [r.label for r in restored] == [r.label for r in records]


def test_filter_records():
    records = make_records(10)
    assert len(filter_records(records, label=1)) == 5
    assert len(filter_records(records, generator="sdxl")) == 5
    assert len(filter_records(records, dataset=["demo", "other"])) == 10
    assert len(filter_records(records, predicate=lambda r: r.source_id.endswith("0"))) == 1


def test_merge_namespaces_source_ids_across_datasets():
    """Same filename in two datasets must not become one source_id."""
    a = [ManifestRecord("a/img.png", 0, "img", dataset="cifake")]
    b = [ManifestRecord("b/img.png", 1, "img", dataset="wildfake")]
    merged = merge_manifests(a, b)
    assert {r.source_id for r in merged} == {"cifake:img", "wildfake:img"}
    assert len({r.source_id for r in merge_manifests(a, b, namespace_source_ids=False)}) == 1
