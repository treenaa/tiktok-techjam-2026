"""Checks that only mean anything on a real CUDA device.

Every test here is skipped -- never quietly passed -- when no GPU is present,
so a green suite on a laptop is not mistaken for a validated GPU instance. On
the rented instance these are the ones that matter.
"""

from __future__ import annotations

import pytest

from src.gpu import (
    assert_on_device,
    build_detector,
    determinism_report,
    measure_config,
    run_gpu_checks,
    smoke_variant,
)
from src.gpu.report import STATUS_PASS, STATUS_WARN

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.cuda


def test_model_data_and_loss_all_land_on_the_gpu(cuda_device, tiny_config):
    results = smoke_variant("dinov2", "fusion", tiny_config, device=cuda_device)
    names = {result.name: result for result in results}
    assert names["placement.weights.dinov2/fusion"].status == STATUS_PASS
    placement = names["placement.compute.dinov2/fusion"]
    assert placement.status == STATUS_PASS
    assert placement.details["logits"].startswith("cuda")
    assert placement.details["loss"].startswith("cuda")


def test_a_model_left_on_the_host_is_caught(cuda_device, tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    from src.gpu import GpuCheckError

    with pytest.raises(GpuCheckError, match="are not on cuda"):
        assert_on_device(model, cuda_device)
    model.to(cuda_device)
    assert assert_on_device(model, cuda_device)["misplaced"] == []


def test_mixed_precision_stays_nan_free(cuda_device, config_factory):
    config = config_factory(smoke={"batch_size": 4, "steps": 3, "amp": True})
    results = smoke_variant("dinov2", "fusion", config, device=cuda_device)
    amp = next(result for result in results if result.name.startswith("amp."))
    assert amp.status in {STATUS_PASS, STATUS_WARN}
    assert all(loss == loss for loss in amp.details["losses"])  # no NaN
    assert "max_abs_logit_divergence_vs_fp32" in amp.details


def test_vram_is_actually_measured(cuda_device, tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    row = measure_config(
        model,
        tiny_config,
        backbone="dinov2",
        architecture="visual",
        device=cuda_device,
        batch_size=4,
        precision="fp32",
        mode="train",
    )
    assert row["status"] == "ok"
    assert row["peak_memory_allocated_bytes"] > 0
    assert row["peak_memory_reserved_bytes"] >= row["peak_memory_allocated_bytes"]
    assert 0 < row["peak_reserved_fraction"] <= 1
    assert row["device_total_memory_bytes"] > 0


def test_an_impossible_batch_is_reported_as_oom_not_a_crash(cuda_device, config_factory):
    """The batch is sized past any current device, so this must not crash."""
    config = config_factory(model={"image_size": 512, "stub": {"patch_size": 16}})
    model = build_detector("dinov2", "visual", config.model)
    row = measure_config(
        model,
        config,
        backbone="dinov2",
        architecture="visual",
        device=cuda_device,
        batch_size=1 << 16,
        precision="fp32",
        mode="train",
    )
    assert row["status"] == "oom"
    assert "batch_size=65536" in row["error"]


def test_same_seed_reproduces_on_the_gpu(cuda_device, tiny_config):
    payload, results = determinism_report(tiny_config, device=cuda_device)
    variant = payload["variants"][0]
    assert variant["max_abs_init_diff"] == 0.0
    assert variant["max_abs_logit_diff"] <= tiny_config.determinism.logit_tolerance
    controls = next(r for r in results if r.name == "determinism.controls")
    # Whatever the verdict, the caveats must be recorded rather than implied.
    assert controls.details["caveats"]


def test_a_full_gpu_run_reports_devices_and_driver(cuda_device, config_factory):
    config = config_factory(device=cuda_device, require_cuda=True, allow_cpu=False)
    report = run_gpu_checks(config)
    assert report.environment["devices"], "a CUDA run must enumerate its devices"
    names = {result.name: result.status for result in report.checks}
    assert names["torch.cuda_build"] == STATUS_PASS
    assert names["torch.cuda_available"] == STATUS_PASS
    assert names["device.resolved"] == STATUS_PASS
    assert report.benchmarks[0]["peak_memory_allocated_bytes"] > 0
