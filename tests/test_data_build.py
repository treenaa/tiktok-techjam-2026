"""Manifest generation utility (library API + CLI)."""

from __future__ import annotations

import csv
import os

import pytest

from src.data import (
    ManifestRecord,
    generate_manifest,
    generate_split_manifests,
    read_manifest,
    validate_splits,
)
from src.data.build import main
from test_data_fixtures import (
    build_cifake_tree,
    build_derivative_tree,
    build_sid_set_tree,
    build_wildfake_tree,
)


# -- library API -----------------------------------------------------------
def test_generate_manifest_writes_the_standard_columns(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=4)
    out = str(tmp_path / "cifake.csv")
    records = generate_manifest(root, adapter="cifake", out_path=out)

    with open(out) as fh:
        header = next(csv.reader(fh))
    for column in ("image_path", "label", "source_id", "dataset", "generator"):
        assert column in header
    assert len(read_manifest(out, root=root)) == len(records) == 16


def test_generate_manifest_defaults_dataset_to_the_adapter_name(tmp_path):
    root = build_wildfake_tree(tmp_path / "wf", n_per_leaf=2)
    records = generate_manifest(root, adapter="wildfake")
    assert {r.dataset for r in records} == {"wildfake"}


def test_generate_manifest_paths_are_relative_and_resolvable(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=3)
    out = str(tmp_path / "m.csv")
    generate_manifest(root, adapter="cifake", out_path=out)
    raw = read_manifest(out)
    assert not os.path.isabs(raw[0].image_path), "manifests should be portable"
    resolved = read_manifest(out, root=root, check_paths_exist=True)
    assert os.path.exists(resolved[0].image_path)


def test_generate_manifest_can_assign_splits(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=10)
    records = generate_manifest(root, adapter="cifake", split=True, seed=0)
    assert {r.split for r in records} == {"train", "val", "test"}
    by_source = {}
    for rec in records:
        by_source.setdefault(rec.source_id, set()).add(rec.split)
    assert all(len(v) == 1 for v in by_source.values())


def test_generate_manifest_without_out_path_writes_nothing(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=2)
    records = generate_manifest(root, adapter="cifake")
    assert records and not list((tmp_path).glob("*.csv"))


def test_generate_split_manifests_produces_one_file_per_split(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=12)
    out_dir = str(tmp_path / "manifests")
    paths = generate_split_manifests(root, out_dir, adapter="cifake", seed=0)

    assert set(paths) == {"train", "val", "test"}
    for name, path in paths.items():
        assert os.path.exists(path)
        records = read_manifest(path, root=root)
        assert records and all(r.split == name for r in records)


def test_generated_split_manifests_pass_validation(tmp_path):
    """The generator's output must satisfy the pre-training gate."""
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=16)
    paths = generate_split_manifests(root, str(tmp_path / "m"), adapter="cifake", seed=0)
    report = validate_splits(paths["train"], paths["val"], paths["test"])
    assert report.ok


def test_generated_manifests_keep_derivatives_together(tmp_path):
    """A tree containing transformed copies must not split them apart."""
    root = build_derivative_tree(tmp_path / "deriv", n_originals=12)
    paths = generate_split_manifests(root, str(tmp_path / "m"), adapter="folder",
                                     dataset="deriv", seed=0)
    assert validate_splits(paths["train"], paths["val"], paths["test"]).ok


def test_generate_works_for_each_adapter(tmp_path):
    for adapter, builder in (
        ("cifake", lambda p: build_cifake_tree(p, n_per_class=4)),
        ("wildfake", lambda p: build_wildfake_tree(p, n_per_leaf=3)),
        ("sid_set", lambda p: build_sid_set_tree(p, n_per_class=4)),
        ("folder", lambda p: build_derivative_tree(p, n_originals=6)),
    ):
        root = builder(tmp_path / adapter)
        records = generate_manifest(root, adapter=adapter, dataset=adapter)
        assert records, adapter
        assert {r.label for r in records} == {0, 1}, adapter


def test_custom_ratios_are_respected(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=20)
    paths = generate_split_manifests(
        root, str(tmp_path / "m"), adapter="cifake", ratios=(0.5, 0.25, 0.25), seed=0
    )
    counts = {name: len(read_manifest(path)) for name, path in paths.items()}
    assert counts["train"] > counts["val"] == counts["test"]


# -- CLI -------------------------------------------------------------------
def test_cli_writes_a_manifest(tmp_path, capsys):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=4)
    out = str(tmp_path / "cli.csv")
    assert main(["--root", root, "--adapter", "cifake", "--out", out]) == 0

    captured = capsys.readouterr().out
    assert "16 images" in captured and "written:" in captured
    assert len(read_manifest(out)) == 16


def test_cli_split_mode(tmp_path, capsys):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=10)
    out_dir = str(tmp_path / "manifests")
    assert main(["--root", root, "--adapter", "cifake", "--out-dir", out_dir]) == 0

    printed = capsys.readouterr().out
    assert "train" in printed and "test" in printed
    for name in ("train", "val", "test"):
        assert os.path.exists(os.path.join(out_dir, "%s.csv" % name))


def test_cli_rejects_an_unknown_adapter(tmp_path):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--adapter", "nosuchdataset"])


def test_cli_skip_unlabelled(tmp_path):
    root = build_cifake_tree(tmp_path / "cifake", n_per_class=3)
    stray = os.path.join(root, "train", "UNKNOWN")
    os.makedirs(stray, exist_ok=True)
    from test_data_fixtures import write_image

    write_image(os.path.join(stray, "x.png"), 999)
    out = str(tmp_path / "m.csv")
    assert main(["--root", root, "--adapter", "cifake", "--out", out, "--skip-unlabelled"]) == 0
    assert all("UNKNOWN" not in r.image_path for r in read_manifest(out))
