from __future__ import annotations

import json

import pytest

from src.gpu import (
    GPU_REPORT_SCHEMA_VERSION,
    CheckResult,
    GpuReport,
    overall_status,
    read_report,
    render_text,
    write_report,
)
from src.gpu.report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN


def _report() -> GpuReport:
    return GpuReport(
        config={"device": "cuda"},
        environment={
            "torch": {"version": "2.4.0", "cuda_version": "12.1", "cuda_available": True},
            "driver": {"version": "535.104.05"},
            "devices": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "total_memory_bytes": 24 * 1024 ** 3,
                    "capability": "8.6",
                    "multi_processor_count": 84,
                }
            ],
            "resolved_device": "cuda",
            "python": {"version": "3.11.0"},
            "platform": {"system": "Linux"},
        },
        parameters=[
            {
                "backbone": "dinov2",
                "architecture": "visual",
                "total": 86_000_000,
                "trainable": 200_000,
                "limit": 2_000_000_000,
                "within_limit": True,
            }
        ],
        benchmarks=[
            {
                "backbone": "dinov2",
                "architecture": "visual",
                "mode": "train",
                "precision": "amp",
                "batch_size": 32,
                "status": "ok",
                "images_per_second": 250.5,
                "peak_memory_allocated_bytes": 3 * 1024 ** 3,
                "peak_memory_reserved_bytes": 4 * 1024 ** 3,
                "vram_budget_bytes": 8172 * 1024 * 1024,
                "budget_fraction": 0.50,
                "fits_vram_budget": True,
                "time_to_first_batch_seconds": 1.25,
            },
            {
                "backbone": "ijepa",
                "architecture": "fusion",
                "mode": "train",
                "precision": "fp32",
                "batch_size": 64,
                "status": "oom",
                "error": "CUDA out of memory",
                "vram_budget_bytes": 8172 * 1024 * 1024,
                "fits_vram_budget": False,
            },
        ],
        recommendations=[
            {
                "backbone": "ijepa",
                "architecture": "fusion",
                "mode": "train",
                "precision": "fp32",
                "batch_size": 64,
                "status": "oom",
                "peak_memory_reserved_bytes": None,
                "vram_budget_bytes": 8172 * 1024 * 1024,
                "budget_fraction": None,
                "fallbacks": ["drop batch_size 64 -> 8", "enable mixed precision (fp16)"],
            }
        ],
        determinism={"caveats": ["cuDNN picks algorithms by benchmarking"]},
        notes=["stub backbones in use"],
        duration_seconds=42.5,
    )


def test_check_status_must_be_a_known_value():
    with pytest.raises(ValueError, match="check status"):
        CheckResult("x", "maybe", "summary")


def test_overall_status_takes_the_worst_and_strict_promotes_warnings():
    checks = [
        CheckResult("a", STATUS_PASS, ""),
        CheckResult("b", STATUS_WARN, ""),
        CheckResult("c", STATUS_SKIP, ""),
    ]
    assert overall_status(checks) == STATUS_WARN
    assert overall_status(checks, strict=True) == STATUS_FAIL
    assert overall_status(checks + [CheckResult("d", STATUS_FAIL, "")]) == STATUS_FAIL
    assert overall_status([]) == STATUS_SKIP


def test_text_report_shows_the_facts_a_human_needs():
    report = _report()
    report.add(CheckResult("device.resolved", STATUS_PASS, "resolved to cuda"))
    report.add(CheckResult("benchmark.oom", STATUS_WARN, "1 configuration OOM"))
    text = render_text(report)
    assert "Test GPU" in text
    assert "24.00 GiB" in text
    assert "86,000,000" in text
    assert "250.5" in text
    assert "4.00 GiB" in text          # peak reserved, the number that decides fit
    assert "50%" in text               # share of the 8 GB budget
    assert "OOM" in text
    assert "Fallbacks for configurations that do not fit in 7.98 GiB" in text
    assert "-> drop batch_size 64 -> 8" in text
    assert "Reproducibility caveats" in text
    assert "stub backbones in use" in text
    assert "OVERALL: WARN" in text
    assert "OVERALL: FAIL" in render_text(report, strict=True)


def test_json_and_text_reports_round_trip_on_disk(tmp_path):
    report = _report()
    report.add(CheckResult("device.resolved", STATUS_PASS, "resolved to cuda"))
    paths = write_report(report, str(tmp_path), basename="gpu_report")
    assert paths["json"].endswith("gpu_report.json")
    assert paths["text"].endswith("gpu_report.txt")

    payload = read_report(paths["json"])
    assert payload["schema_version"] == GPU_REPORT_SCHEMA_VERSION
    assert payload["status"] == STATUS_PASS
    assert payload["checks"][0]["name"] == "device.resolved"
    assert payload["benchmarks"][1]["status"] == "oom"
    assert payload["determinism"]["caveats"]
    assert payload["recommendations"][0]["fallbacks"]
    # Reparsing proves nothing unserialisable leaked into the payload.
    assert json.loads(open(paths["json"], encoding="utf-8").read()) == payload


def test_strict_flag_is_recorded_in_the_written_report(tmp_path):
    report = _report()
    report.add(CheckResult("benchmark.oom", STATUS_WARN, "1 configuration OOM"))
    paths = write_report(report, str(tmp_path), strict=True)
    payload = read_report(paths["json"])
    assert payload["strict"] is True
    assert payload["status"] == STATUS_FAIL


def test_writing_creates_the_output_directory(tmp_path):
    target = tmp_path / "nested" / "gpu"
    write_report(_report(), str(target))
    assert (target / "gpu_report.json").exists()
