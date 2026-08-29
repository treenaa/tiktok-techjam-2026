"""Rule 11.C: dataset-shortcut auditing.

The trap: real=COCO / AIGC=one generator lets a model score ~98% by recognising
the corpus rather than AI-ness, then collapse on new sources.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from src.data import ManifestRecord, audit_shortcuts, format_audit_report
from src.data.audit import (
    ShortcutFinding,
    encoding_shortcut,
    generator_concentration,
    provenance_shortcut,
)


def write(path, size=(64, 64), fmt="PNG", seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = (np.random.default_rng(seed).random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(array).save(path, format=fmt)
    return path


@pytest.fixture
def shortcut_corpus(tmp_path):
    """The dangerous setup: real=COCO jpegs 640x480, AIGC=one generator PNG 1024."""
    records = []
    for i in range(12):
        records.append(ManifestRecord(
            write(str(tmp_path / "coco" / ("r%d.jpg" % i)), (640, 480), "JPEG", i),
            0, "r%d" % i, dataset="mscoco_train"))
    for i in range(12):
        records.append(ManifestRecord(
            write(str(tmp_path / "gen" / ("a%d.png" % i)), (1024, 1024), "PNG", i),
            1, "a%d" % i, dataset="sdxl_dump", generator="sdxl"))
    return records


@pytest.fixture
def healthy_corpus(tmp_path):
    """Both datasets contribute both labels; sizes and formats overlap."""
    records = []
    for i in range(24):
        dataset = ["cifake", "wildfake"][(i // 2) % 2]   # independent of label
        label = i % 2
        records.append(ManifestRecord(
            write(str(tmp_path / "mix" / ("m%d.jpg" % i)), (256, 256), "JPEG", i),
            label, "m%d" % i, dataset=dataset,
            generator=["sdxl", "midjourney", "flux"][i % 3] if label else ""))
    return records


def kinds(report, severity=None):
    return {
        f["kind"] for f in report["findings"]
        if severity is None or f["severity"] == severity
    }


# -- the trap is caught ----------------------------------------------------
def test_shortcut_corpus_is_flagged_critical(shortcut_corpus):
    report = audit_shortcuts(shortcut_corpus)
    assert report["worst_severity"] == "critical"
    assert report["n_critical"] >= 2


def test_provenance_shortcut_is_detected(shortcut_corpus):
    assert "provenance" in kinds(audit_shortcuts(shortcut_corpus), "critical")


def test_resolution_shortcut_is_detected(shortcut_corpus):
    assert "resolution" in kinds(audit_shortcuts(shortcut_corpus), "critical")


def test_encoding_shortcut_is_detected(shortcut_corpus):
    assert "encoding" in kinds(audit_shortcuts(shortcut_corpus), "critical")


def test_single_generator_is_flagged(shortcut_corpus):
    assert "generator_concentration" in kinds(audit_shortcuts(shortcut_corpus), "warning")


def test_report_explains_the_risk_in_words(shortcut_corpus):
    text = format_audit_report(audit_shortcuts(shortcut_corpus))
    assert "CRIT" in text
    assert "dataset identity" in text or "provenance" in text


def test_raise_on_critical_is_opt_in(shortcut_corpus):
    audit_shortcuts(shortcut_corpus)  # advisory by default
    with pytest.raises(AssertionError):
        audit_shortcuts(shortcut_corpus, raise_on_critical=True)


# -- no false alarms on a healthy corpus ----------------------------------
def test_healthy_corpus_is_clean(healthy_corpus):
    report = audit_shortcuts(healthy_corpus)
    assert report["worst_severity"] == "info", format_audit_report(report)
    assert report["n_critical"] == 0


def test_healthy_corpus_reports_every_axis(healthy_corpus):
    assert kinds(audit_shortcuts(healthy_corpus)) == {
        "provenance", "generator_concentration", "resolution", "encoding", "file_size",
    }


# -- individual detectors --------------------------------------------------
def test_provenance_detects_partial_shortcuts():
    """Most records in single-label datasets -> warning, not critical."""
    records = [ManifestRecord("a%d.png" % i, 0, "a%d" % i, dataset="pure_real") for i in range(20)]
    records += [ManifestRecord("b%d.png" % i, 1, "b%d" % i, dataset="pure_fake") for i in range(20)]
    records += [ManifestRecord("c%d.png" % i, i % 2, "c%d" % i, dataset="mixed") for i in range(4)]
    severities = {f.kind: f.severity for f in provenance_shortcut(records)}
    assert severities["provenance"] == "warning"


def test_single_dataset_is_flagged_as_unassessable():
    records = [ManifestRecord("a%d.png" % i, i % 2, "a%d" % i, dataset="only") for i in range(10)]
    finding = provenance_shortcut(records)[0]
    assert finding.severity == "warning"
    assert "single dataset" in finding.summary


def test_encoding_detector_on_synthetic_probe():
    probe = {0: {"formats": ["JPEG"] * 20}, 1: {"formats": ["PNG"] * 20}}
    assert encoding_shortcut(probe)[0].severity == "critical"
    probe = {0: {"formats": ["JPEG"] * 20}, 1: {"formats": ["JPEG"] * 18 + ["PNG"] * 2}}
    assert encoding_shortcut(probe)[0].severity == "info"


def test_generator_concentration_thresholds():
    records = [ManifestRecord("a%d.png" % i, 1, "a%d" % i, generator="sdxl") for i in range(19)]
    records.append(ManifestRecord("b.png", 1, "b", generator="flux"))
    assert generator_concentration(records)[0].severity == "warning"


def test_no_aigc_records_yields_no_generator_finding():
    records = [ManifestRecord("a%d.png" % i, 0, "a%d" % i) for i in range(4)]
    assert generator_concentration(records) == []


# -- API behaviour ---------------------------------------------------------
def test_metadata_only_mode_skips_file_probing():
    """Works when the images are not on disk."""
    records = [ManifestRecord("/nonexistent/%d.png" % i, i % 2, "s%d" % i, dataset="d")
               for i in range(10)]
    report = audit_shortcuts(records, inspect_files=False)
    assert kinds(report) == {"provenance", "generator_concentration"}


def test_missing_files_are_skipped_not_fatal(tmp_path, healthy_corpus):
    records = healthy_corpus + [ManifestRecord("/nope/x.png", 1, "gone", dataset="cifake")]
    assert audit_shortcuts(records)["n_records"] == len(records)


def test_sampling_caps_the_work(shortcut_corpus):
    report = audit_shortcuts(shortcut_corpus, sample_size=4)
    assert report["n_records"] == len(shortcut_corpus)


def test_audit_is_deterministic(shortcut_corpus):
    a = audit_shortcuts(shortcut_corpus, sample_size=8, seed=3)
    b = audit_shortcuts(shortcut_corpus, sample_size=8, seed=3)
    assert a == b


def test_report_is_json_serialisable(shortcut_corpus):
    import json

    json.dumps(audit_shortcuts(shortcut_corpus))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        audit_shortcuts([])


def test_invalid_severity_is_rejected():
    with pytest.raises(ValueError, match="severity"):
        ShortcutFinding("k", "catastrophic", "s")
