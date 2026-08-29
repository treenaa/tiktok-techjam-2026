"""Dataset adapters: folder layouts -> records, for CIFAKE / SID_Set / WildFake."""

from __future__ import annotations

import os

import pytest

from src.data import (
    DataError,
    build_manifest,
    cifake_adapter,
    describe_records,
    from_class_folders,
    from_folder,
    list_adapters,
    merge_manifests,
    read_manifest,
    sid_set_adapter,
    split_records,
    wildfake_adapter,
    write_manifest,
)
from test_data_fixtures import (
    build_cifake_tree,
    build_derivative_tree,
    build_sid_set_tree,
    build_wildfake_tree,
    write_image,
)


# -- generic folder adapter ------------------------------------------------
def test_from_folder_labels_by_directory_name(tmp_path):
    root = build_derivative_tree(tmp_path / "ds", n_originals=4)
    records = from_folder(root, dataset="demo")
    assert len(records) == 16  # 4 originals x 4 views
    assert {r.label for r in records} == {0, 1}
    assert all(r.dataset == "demo" for r in records)
    assert all(os.path.isabs(r.image_path) for r in records)


def test_from_folder_gives_derivatives_a_shared_source_id(tmp_path):
    """The core contract: the 4 on-disk views of an image share one source_id."""
    root = build_derivative_tree(tmp_path / "ds", n_originals=6)
    records = from_folder(root, dataset="demo")
    by_source = {}
    for rec in records:
        by_source.setdefault(rec.source_id, []).append(rec)
    assert len(by_source) == 6
    for source_id, group in by_source.items():
        assert len(group) == 4, source_id
        assert len({r.label for r in group}) == 1, "a group must not span labels here"


def test_from_folder_rejects_unlabelled_directories(tmp_path):
    root = tmp_path / "ds"
    write_image(str(root / "mystery" / "a.png"), 1)
    with pytest.raises(DataError, match="not covered by class_map"):
        from_folder(str(root))


def test_from_folder_can_skip_unlabelled_directories(tmp_path):
    root = tmp_path / "ds"
    write_image(str(root / "real" / "a.png"), 1)
    write_image(str(root / "mystery" / "b.png"), 2)
    records = from_folder(str(root), on_unlabelled="skip")
    assert len(records) == 1 and records[0].label == 0


def test_from_folder_custom_class_map(tmp_path):
    root = tmp_path / "ds"
    write_image(str(root / "camera" / "a.png"), 1)
    write_image(str(root / "midjourney" / "b.png"), 2)
    records = from_folder(str(root), class_map={"camera": 0, "midjourney": 1})
    assert sorted(r.label for r in records) == [0, 1]


def test_deepest_directory_segment_wins(tmp_path):
    """``fake/.../real_photos/`` must not be mislabelled by an ancestor."""
    root = tmp_path / "ds"
    write_image(str(root / "fake" / "sdxl" / "a.png"), 1)
    write_image(str(root / "real" / "camera" / "b.png"), 2)
    records = {os.path.basename(r.image_path): r.label for r in from_folder(str(root))}
    assert records == {"a.png": 1, "b.png": 0}


def test_empty_directory_fails_clearly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError, match="no images"):
        from_folder(str(empty))


def test_missing_root_fails_clearly(tmp_path):
    with pytest.raises(DataError, match="not a directory"):
        from_folder(str(tmp_path / "absent"))


def test_from_class_folders(tmp_path):
    root = tmp_path / "ds"
    write_image(str(root / "photos" / "a.png"), 1)
    write_image(str(root / "aigen" / "b.png"), 2)
    records = from_class_folders(str(root / "photos"), str(root / "aigen"), dataset="x")
    assert sorted(r.label for r in records) == [0, 1]


# -- CIFAKE ----------------------------------------------------------------
def test_cifake_adapter(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=5)
    records = cifake_adapter(root)
    assert len(records) == 20  # 2 splits x 2 classes x 5
    assert {r.dataset for r in records} == {"cifake"}
    assert {r.split for r in records} == {"train", "test"}
    info = describe_records(records)
    assert info["n_real"] == info["n_aigc"] == 10


def test_cifake_repeated_filenames_stay_distinct(tmp_path):
    """CIFAKE reuses ``0000.png`` in REAL and FAKE -- different originals."""
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=5)
    records = cifake_adapter(root)
    assert len({r.source_id for r in records}) == len(records)


def test_cifake_flat_layout_without_split_dirs(tmp_path):
    root = tmp_path / "cifake_flat"
    for cls in ("REAL", "FAKE"):
        for i in range(3):
            write_image(str(root / cls / ("%02d.png" % i)), i)
    records = cifake_adapter(str(root))
    assert len(records) == 6 and {r.split for r in records} == {""}


# -- SID_Set ---------------------------------------------------------------
def test_sid_set_maps_three_classes_to_binary(tmp_path):
    root = build_sid_set_tree(tmp_path / "sid", n_per_class=4)
    records = sid_set_adapter(root)
    assert len(records) == 36  # 3 splits x 4 x (real + tampered + synthetic)
    info = describe_records(records)
    assert info["n_real"] == 12
    assert info["n_aigc"] == 24, "synthetic AND tampered are both AIGC"
    assert {r.split for r in records} == {"train", "val", "test"}


def test_sid_set_class_map_is_overridable(tmp_path):
    """Treating tampered images as real is a one-argument change."""
    root = build_sid_set_tree(tmp_path / "sid", n_per_class=4)
    records = sid_set_adapter(root, class_map={"tampered": 0})
    info = describe_records(records)
    assert info["n_real"] == 24 and info["n_aigc"] == 12


def test_sid_set_can_pair_tampered_images_with_their_real_source(tmp_path):
    """Opt-in: a tampered image inherits the source_id it was edited from."""
    root = build_sid_set_tree(tmp_path / "sid", n_per_class=4)
    unpaired = sid_set_adapter(root)
    assert len({r.source_id for r in unpaired}) == len(unpaired)

    paired = sid_set_adapter(root, tampered_share_real_source_id=True)
    by_source = {}
    for rec in paired:
        by_source.setdefault(rec.source_id, set()).add(rec.label)
    mixed = [sid for sid, labels in by_source.items() if labels == {0, 1}]
    assert len(mixed) == 12, "each real image pairs with its tampered version"

    # ...and the splitter keeps those pairs on one side.
    splits = split_records(paired, seed=0)
    seen = {}
    for name, members in splits.items():
        for rec in members:
            assert seen.setdefault(rec.source_id, name) == name


# -- WildFake --------------------------------------------------------------
def test_wildfake_records_the_generator(tmp_path):
    root = build_wildfake_tree(tmp_path / "wild", n_per_leaf=3)
    records = wildfake_adapter(root)
    generators = describe_records(records)["generators"]
    assert set(generators) == {"gan", "diffusion"}
    assert all(r.generator == "" for r in records if r.label == 0)


def test_wildfake_generator_depth_selects_the_model(tmp_path):
    root = build_wildfake_tree(tmp_path / "wild", n_per_leaf=3)
    records = wildfake_adapter(root, generator_depth=1)
    generators = describe_records(records)["generators"]
    assert set(generators) == {"stylegan2", "biggan", "sdxl"}


def test_wildfake_can_stratify_by_generator(tmp_path):
    root = build_wildfake_tree(tmp_path / "wild", n_per_leaf=8)
    records = wildfake_adapter(root, generator_depth=1)
    splits = split_records(records, seed=0, stratify_keys=("generator", "label"))
    for members in splits.values():
        assert {r.generator for r in members} >= {"stylegan2", "biggan", "sdxl"}


# -- registry & pooling ----------------------------------------------------
def test_build_manifest_dispatches_by_name(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=3)
    assert len(build_manifest("cifake", root)) == 12
    assert "wildfake" in list_adapters() and "folder" in list_adapters()


def test_build_manifest_rejects_unknown_datasets(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset adapter"):
        build_manifest("imagenet", str(tmp_path))


def test_pooling_datasets_keeps_source_ids_namespaced(tmp_path):
    cifake = cifake_adapter(build_cifake_tree(tmp_path / "cifake", n_per_class=4))
    wild = wildfake_adapter(build_wildfake_tree(tmp_path / "wild", n_per_leaf=3))
    pooled = merge_manifests(cifake, wild)
    assert len(pooled) == len(cifake) + len(wild)
    assert all(":" in r.source_id for r in pooled)
    assert {r.dataset for r in pooled} == {"cifake", "wildfake"}
    splits = split_records(pooled, seed=0, stratify_keys=("dataset", "label"))
    for members in splits.values():
        assert {r.dataset for r in members} == {"cifake", "wildfake"}


# -- end-to-end: folders -> manifest -> split -> manifest ------------------
def test_adapter_to_manifest_round_trip(tmp_path):
    root = build_wildfake_tree(tmp_path / "wild", n_per_leaf=4)
    records = wildfake_adapter(root, generator_depth=1)
    path = str(tmp_path / "wildfake.csv")
    write_manifest(records, path, relative_to=root)
    loaded = read_manifest(path, root=root, check_paths_exist=True)
    assert len(loaded) == len(records)
    assert [r.source_id for r in loaded] == [r.source_id for r in records]
    assert [r.generator for r in loaded] == [r.generator for r in records]
    assert [r.label for r in loaded] == [r.label for r in records]
