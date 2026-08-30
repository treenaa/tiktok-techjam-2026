from __future__ import annotations

import pytest

from src.gpu import GpuCheckError, benchmark_checks, benchmark_variant, build_detector, measure_config
from src.gpu.report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN

torch = pytest.importorskip("torch")


def _status(results, name):
    for result in results:
        if result.name == name:
            return result.status
    raise AssertionError("no check named %r" % name)


def test_measure_config_reports_timing_and_memory_fields(tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    row = measure_config(
        model,
        tiny_config,
        backbone="dinov2",
        architecture="visual",
        device="cpu",
        batch_size=2,
        precision="fp32",
        mode="train",
    )
    assert row["status"] == "ok"
    assert row["images_per_second"] > 0
    assert row["time_to_first_batch_seconds"] >= 0
    assert row["seconds_per_step"] > 0
    # CPU has no VRAM accounting; the fields exist and are explicitly null.
    assert row["peak_memory_allocated_bytes"] is None
    assert row["peak_reserved_fraction"] is None


def test_amp_rows_are_skipped_rather_than_faked_off_cuda(tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    row = measure_config(
        model,
        tiny_config,
        backbone="dinov2",
        architecture="visual",
        device="cpu",
        batch_size=2,
        precision="amp",
        mode="inference",
    )
    assert row["status"] == "skipped"
    assert "CUDA" in row["reason"]


def test_out_of_memory_becomes_a_reported_row_not_a_crash(tiny_config, monkeypatch):
    model = build_detector("dinov2", "visual", tiny_config.model)

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    monkeypatch.setattr(model, "forward", explode)
    row = measure_config(
        model,
        tiny_config,
        backbone="dinov2",
        architecture="visual",
        device="cpu",
        batch_size=512,
        precision="fp32",
        mode="train",
    )
    assert row["status"] == "oom"
    # The context that produced it survives into the report.
    assert "batch_size=512" in row["error"]
    assert "out of memory" in row["error"]


def test_a_non_oom_cuda_error_aborts_with_full_context(tiny_config, monkeypatch):
    model = build_detector("dinov2", "visual", tiny_config.model)

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(model, "forward", explode)
    with pytest.raises(GpuCheckError) as info:
        measure_config(
            model,
            tiny_config,
            backbone="dinov2",
            architecture="visual",
            device="cpu",
            batch_size=8,
            precision="fp32",
            mode="inference",
        )
    assert "batch_size=8" in str(info.value)
    assert "device-side assert" in str(info.value)


def test_benchmark_variant_covers_the_configured_grid(tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    rows = benchmark_variant(
        model, tiny_config, backbone="dinov2", architecture="visual", device="cpu"
    )
    assert len(rows) == 2  # two modes x one precision x one batch size
    assert {row["mode"] for row in rows} == {"train", "inference"}


def test_checks_flag_oom_and_vram_pressure(tiny_config):
    rows = [
        {
            "backbone": "clip",
            "architecture": "fusion",
            "mode": "train",
            "precision": "amp",
            "batch_size": 64,
            "status": "oom",
            "error": "CUDA out of memory [batch_size=64]",
        },
        {
            "backbone": "clip",
            "architecture": "fusion",
            "mode": "train",
            "precision": "amp",
            "batch_size": 32,
            "status": "ok",
            "images_per_second": 120.0,
            "time_to_first_batch_seconds": 1.0,
            "peak_reserved_fraction": 0.97,
        },
    ]
    results = benchmark_checks(rows, tiny_config, device="cuda")
    assert _status(results, "benchmark.oom") == STATUS_WARN
    assert _status(results, "benchmark.vram_headroom") == STATUS_WARN
    assert _status(results, "benchmark.completed") == STATUS_PASS
    assert _status(results, "benchmark.throughput.inference") == STATUS_SKIP


def test_throughput_floor_fails_only_when_nothing_clears_it(config_factory):
    config = config_factory(budget={"min_train_images_per_second": 500.0})
    rows = [
        {
            "backbone": "dinov2",
            "architecture": "visual",
            "mode": "train",
            "precision": "fp32",
            "batch_size": 8,
            "status": "ok",
            "images_per_second": 100.0,
            "time_to_first_batch_seconds": 0.5,
            "peak_reserved_fraction": None,
        }
    ]
    assert _status(benchmark_checks(rows, config, device="cuda"), "benchmark.throughput.train") == (
        STATUS_FAIL
    )
    rows.append(dict(rows[0], batch_size=32, images_per_second=900.0))
    assert _status(benchmark_checks(rows, config, device="cuda"), "benchmark.throughput.train") == (
        STATUS_PASS
    )


def test_slow_first_batch_is_a_warning_with_the_offending_config(config_factory):
    config = config_factory(budget={"max_time_to_first_batch_seconds": 1.0})
    rows = [
        {
            "backbone": "ijepa",
            "architecture": "fusion",
            "mode": "train",
            "precision": "fp32",
            "batch_size": 32,
            "status": "ok",
            "images_per_second": 50.0,
            "time_to_first_batch_seconds": 45.0,
            "peak_reserved_fraction": 0.5,
        }
    ]
    results = benchmark_checks(rows, config, device="cuda")
    result = next(r for r in results if r.name == "benchmark.time_to_first_batch")
    assert result.status == STATUS_WARN
    assert "ijepa/fusion" in result.summary
