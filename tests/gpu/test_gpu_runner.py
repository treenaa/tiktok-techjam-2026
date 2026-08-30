from __future__ import annotations

import pytest

from src.gpu import run_benchmark_report, run_gpu_checks
from src.gpu.report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP
from src.gpu.runner import CPU_NOTE, STUB_NOTE

torch = pytest.importorskip("torch")


def _names(report):
    return {result.name: result.status for result in report.checks}


def test_a_full_cpu_dry_run_produces_a_complete_report(tiny_config):
    report = run_gpu_checks(tiny_config)
    names = _names(report)
    assert names["placement.weights.dinov2/visual"] == STATUS_PASS
    assert names["smoke.fp32.dinov2/visual"] == STATUS_PASS
    assert names["model.parameter_budget"] == STATUS_PASS
    assert names["benchmark.completed"] == STATUS_PASS
    assert names["determinism.repeat_run"] == STATUS_PASS
    assert names["budget.wall_clock"] == STATUS_PASS
    assert report.parameters and report.benchmarks
    assert report.determinism["caveats"]
    assert report.duration_seconds > 0


def test_a_cpu_run_says_so_instead_of_implying_a_gpu_was_tested(tiny_config):
    report = run_gpu_checks(tiny_config, include_benchmarks=False, include_determinism=False)
    assert CPU_NOTE in report.notes
    assert STUB_NOTE in report.notes
    names = _names(report)
    assert names["benchmark.completed"] == STATUS_SKIP
    assert names["determinism.repeat_run"] == STATUS_SKIP


def test_requiring_cuda_without_cuda_fails_and_runs_nothing(config_factory):
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA, so the missing-GPU path cannot be exercised")
    config = config_factory(device="auto", require_cuda=True, allow_cpu=False)
    report = run_gpu_checks(config)
    names = _names(report)
    assert report.status() == STATUS_FAIL
    assert names["torch.cuda_available"] == STATUS_FAIL
    # Nothing may masquerade as a validated GPU: the work is skipped, loudly.
    assert names["run.execution"] == STATUS_SKIP
    assert not report.benchmarks
    assert not report.parameters


def test_parameter_budget_failure_is_reported_per_variant(config_factory):
    config = config_factory(budget={"parameter_limit": 10})
    report = run_gpu_checks(config, include_benchmarks=False, include_determinism=False)
    result = next(r for r in report.checks if r.name == "model.parameter_budget")
    assert result.status == STATUS_FAIL
    assert "dinov2/visual" in result.summary
    assert result.details["limit"] == 10


def test_parameter_counts_are_reported_for_every_variant(config_factory):
    config = config_factory(
        model={"backbones": ["ijepa", "clip"], "architectures": ["visual", "fusion"]}
    )
    report = run_gpu_checks(config, include_benchmarks=False, include_determinism=False)
    reported = {(row["backbone"], row["architecture"]) for row in report.parameters}
    assert reported == {
        ("ijepa", "visual"),
        ("ijepa", "fusion"),
        ("clip", "visual"),
        ("clip", "fusion"),
    }
    for row in report.parameters:
        assert row["total"] > 0
        assert row["within_limit"] is True
        assert row["backbone_source"] == "stub"


def test_wall_clock_warning_fires_when_the_budget_is_unrealistic(config_factory):
    config = config_factory(budget={"max_total_seconds": 1e-6})
    report = run_gpu_checks(config, include_benchmarks=False, include_determinism=False)
    result = next(r for r in report.checks if r.name == "budget.wall_clock")
    assert result.status == "warn"
    assert "over the" in result.summary


def test_benchmark_only_entry_point_skips_smoke_and_determinism(tiny_config):
    report = run_benchmark_report(tiny_config)
    names = _names(report)
    assert report.benchmarks
    assert names["determinism.repeat_run"] == STATUS_SKIP
    assert not any(name.startswith("smoke.") for name in names)


def test_a_config_over_the_vram_budget_reaches_the_report_with_fallbacks(
    config_factory, monkeypatch
):
    """Simulated allocator readings, so the 8 GB path is covered without a GPU."""
    from src.gpu import benchmark as benchmark_module

    budget_bytes = 8172 * 1024 * 1024

    state = {"peak": 0}

    def fake_snapshot(device):
        # Peak scales with the batch currently in flight; the 64-batch row is
        # sized past the budget on purpose.
        return {
            "peak_memory_allocated_bytes": state["peak"],
            "peak_memory_reserved_bytes": state["peak"],
            "steady_memory_allocated_bytes": state["peak"] // 2,
            "device_total_memory_bytes": budget_bytes,
        }

    original_measure = benchmark_module.measure_config

    def measure(model, config, **kwargs):
        state["peak"] = int(budget_bytes * (0.2 if kwargs["batch_size"] <= 4 else 0.95))
        return original_measure(model, config, **kwargs)

    monkeypatch.setattr(benchmark_module, "_memory_snapshot", fake_snapshot)
    monkeypatch.setattr(benchmark_module, "measure_config", measure)

    config = config_factory(
        benchmark={
            "batch_sizes": [4, 64],
            "warmup_steps": 0,
            "measure_steps": 1,
            "precisions": ["fp32"],
            "modes": ["train"],
        },
        budget={
            "vram_budget_mb": 8172,
            "vram_headroom_fraction": 0.85,
            "target_train_batch_size": 32,
        },
    )
    report = run_gpu_checks(config, include_smoke=False, include_determinism=False)

    assert [row["fits_vram_budget"] for row in report.benchmarks] == [True, False]
    assert len(report.recommendations) == 1
    recommendation = report.recommendations[0]
    assert recommendation["batch_size"] == 64
    joined = " | ".join(recommendation["fallbacks"])
    assert "drop batch_size 64 -> 4" in joined
    assert "gradient accumulation: 8 micro-batches of 4" in joined
    assert "gradient checkpointing" in joined

    check = next(r for r in report.checks if r.name == "benchmark.vram_budget")
    assert check.status == "warn"
    assert "8172 MiB" in check.summary
