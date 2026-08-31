"""The audit CLI: exit codes must be CI-usable."""

from __future__ import annotations

import json
import os

import pytest

from src.data import ManifestRecord, write_manifest
from src.data.audit_cli import main
from src.data.synthetic import make_synthetic_dataset
from test_data_fixtures import write_image

OK, FAIL, USAGE = 0, 1, 2


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("cli"), n_per_class=12)


# -- splits ----------------------------------------------------------------
def test_clean_splits_exit_zero(bundle, capsys):
    code = main(["splits", "--train", bundle.train_manifest,
                 "--val", bundle.val_manifest, "--test", bundle.test_manifest])
    assert code == OK
    assert "PASSED" in capsys.readouterr().out


def test_leaked_splits_exit_one(tmp_path, bundle, capsys):
    leaked = str(tmp_path / "leaky.csv")
    write_manifest(bundle.train[:4] + bundle.test[:2], leaked)
    code = main(["splits", "--train", leaked, "--test", bundle.test_manifest])
    assert code == FAIL
    assert "FAILED" in capsys.readouterr().out


def test_protected_data_in_training_exits_one(tmp_path, capsys):
    """Rule 11.B through the CLI."""
    records = [ManifestRecord("/data/coco/val2017/%d.jpg" % i, 0, "c%d" % i) for i in range(6)]
    records += [ManifestRecord("/data/gen/%d.png" % i, 1, "g%d" % i) for i in range(6)]
    path = str(tmp_path / "train.csv")
    write_manifest(records, path)
    assert main(["splits", "--train", path]) == FAIL
    assert "demonstration-only" in capsys.readouterr().out


def test_splits_json_output_is_machine_readable(bundle, capsys):
    main(["splits", "--train", bundle.train_manifest, "--val", bundle.val_manifest,
          "--test", bundle.test_manifest, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "stats" in payload and "problems" in payload


def test_splits_requires_input(capsys):
    assert main(["splits"]) == USAGE


# -- shortcut --------------------------------------------------------------
def test_shortcut_audit_runs(bundle, capsys):
    assert main(["shortcut", "--manifest", bundle.train_manifest]) == OK
    assert "shortcut audit" in capsys.readouterr().out


def test_shortcut_strict_fails_on_critical(tmp_path, capsys):
    """real=one dataset/format, AIGC=another -> critical -> non-zero under --strict."""
    records = []
    for i in range(8):
        path = str(tmp_path / "real" / ("r%d.jpg" % i))
        write_image(path, i, size=(64, 48))
        records.append(ManifestRecord(path, 0, "r%d" % i, dataset="coco_like"))
    for i in range(8):
        path = str(tmp_path / "fake" / ("a%d.png" % i))
        write_image(path, 100 + i, size=(128, 128))
        records.append(ManifestRecord(path, 1, "a%d" % i, dataset="gen_dump", generator="g"))
    manifest = str(tmp_path / "m.csv")
    write_manifest(records, manifest)

    assert main(["shortcut", "--manifest", manifest]) == OK          # advisory
    assert main(["shortcut", "--manifest", manifest, "--strict"]) == FAIL
    assert "CRIT" in capsys.readouterr().out


def test_shortcut_metadata_only_mode(bundle):
    assert main(["shortcut", "--manifest", bundle.train_manifest, "--no-files"]) == OK


def test_shortcut_requires_input():
    assert main(["shortcut"]) == USAGE


# -- verify ----------------------------------------------------------------
def test_verify_clean_directory(bundle, capsys):
    assert main(["verify", "--input", bundle.root]) == OK
    assert "readable" in capsys.readouterr().out


def test_verify_reports_corrupt_files(tmp_path, capsys):
    write_image(str(tmp_path / "good.png"), 1)
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert main(["verify", "--input", str(tmp_path)]) == FAIL
    output = capsys.readouterr().out
    assert "UNREADABLE" in output and "bad.png" in output


def test_verify_json_lists_reasons(tmp_path, capsys):
    write_image(str(tmp_path / "good.png"), 1)
    (tmp_path / "bad.png").write_bytes(b"")
    main(["verify", "--input", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_checked"] == 2 and len(payload["unreadable"]) == 1
    assert payload["unreadable"][0]["reason"]


def test_verify_manifest_applies_root_once(tmp_path, capsys):
    root = tmp_path / "images"
    image_path = root / "real" / "sample.png"
    write_image(str(image_path), 1)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [ManifestRecord("real/sample.png", 0, "sample")],
        str(manifest),
    )

    assert main(
        ["verify", "--manifest", str(manifest), "--root", str(root)]
    ) == OK
    assert "1/1 images readable" in capsys.readouterr().out


def test_verify_requires_input():
    assert main(["verify"]) == USAGE


# -- compare ---------------------------------------------------------------
def test_shipped_baselines_compare_cleanly(capsys):
    pytest.importorskip("yaml")
    code = main(["compare", "configs/baseline_clip.yaml", "configs/baseline_dino.yaml",
                 "configs/baseline_ijepa.yaml"])
    assert code == OK
    assert "COMPARABLE" in capsys.readouterr().out


def test_compare_detects_drift(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml

    with open("configs/baseline_dino.yaml") as fh:
        raw = yaml.safe_load(fh)
    raw["name"] = "drifted"
    raw["training"]["lr"] = 0.1
    drifted = str(tmp_path / "drifted.yaml")
    with open(drifted, "w") as fh:
        yaml.safe_dump(raw, fh)

    assert main(["compare", "configs/baseline_clip.yaml", drifted]) == FAIL
    output = capsys.readouterr().out
    assert "NOT COMPARABLE" in output and "training.lr" in output


def test_compare_accepts_globs(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml

    # Glob over a fixture directory rather than `configs/`: the shipped configs
    # legitimately include dataset-specific and intervention runs that are not
    # mutually comparable, and this test is about pattern expansion, not about
    # which shipped runs happen to share a fingerprint. The comparability of the
    # backbone-comparison set is asserted by
    # `test_shipped_baselines_compare_cleanly`.
    with open("configs/baseline_dino.yaml") as fh:
        raw = yaml.safe_load(fh)

    for name, backbone in (("globbed_a", "dinov2_vitb14"), ("globbed_b", "clip_vitb16")):
        cfg = dict(raw, name=name)
        cfg["model"] = dict(raw["model"], backbone=backbone)
        with open(tmp_path / ("%s.yaml" % name), "w") as fh:
            yaml.safe_dump(cfg, fh)

    assert main(["compare", str(tmp_path / "globbed_*.yaml")]) == OK
    output = capsys.readouterr().out
    assert "comparing 2 runs" in output and "COMPARABLE" in output


def test_compare_needs_two_configs():
    assert main(["compare", "configs/baseline_clip.yaml"]) == USAGE


# -- general ---------------------------------------------------------------
def test_no_command_prints_help(capsys):
    assert main([]) == USAGE
    assert "usage" in capsys.readouterr().out.lower()


def test_missing_manifest_is_reported_not_traced(capsys):
    assert main(["splits", "--train", "/nonexistent/train.csv"]) == FAIL
    assert "error:" in capsys.readouterr().err
