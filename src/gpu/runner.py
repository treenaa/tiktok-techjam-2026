"""Orchestration: environment -> smoke -> benchmark -> determinism -> report.

This is the entry point both CLI scripts and the pytest suite call. It is
independently runnable: with ``backbone_source='stub'`` it needs no dataset, no
pretrained weights, and no other subsystem to be finished.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from .benchmark import benchmark_checks, benchmark_variant
from .fallbacks import budget_recommendations
from .builders import build_detector, parameter_row
from .config import GpuCheckConfig
from .determinism import determinism_report
from .environment import collect_environment, environment_checks
from .probes import is_cuda, smoke_variant
from .report import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    CheckResult,
    GpuReport,
)

STUB_NOTE = (
    "Backbones are download-free stand-ins (backbone_source='stub'): the plumbing, device "
    "placement and precision behaviour are real, but throughput and VRAM figures are NOT "
    "those of I-JEPA, DINOv2 or CLIP. Re-run with backbone_source='pretrained' on the GPU "
    "instance before sizing a real training run."
)

CPU_NOTE = (
    "This run did not execute on a CUDA device, so throughput and VRAM figures describe the "
    "host, not the training GPU. It validates the code path only."
)


def _parameter_check(rows: List[Dict[str, Any]], config: GpuCheckConfig) -> CheckResult:
    """The competition's <2B parameter ceiling, per variant."""
    if not rows:
        return CheckResult("model.parameter_budget", STATUS_SKIP, "no variants built", {})
    over = [row for row in rows if not row["within_limit"]]
    largest = max(rows, key=lambda row: row["total"])
    limit = config.budget.parameter_limit
    if over:
        return CheckResult(
            "model.parameter_budget",
            STATUS_FAIL,
            "%d variant(s) at or above the %s parameter limit: %s"
            % (
                len(over),
                f"{limit:,}",
                ", ".join(
                    "%s/%s (%s)" % (row["backbone"], row["architecture"], f"{row['total']:,}")
                    for row in over
                ),
            ),
            {"limit": limit, "rows": rows},
        )
    return CheckResult(
        "model.parameter_budget",
        STATUS_PASS,
        "largest variant %s/%s has %s parameters, under the %s limit"
        % (
            largest["backbone"],
            largest["architecture"],
            f"{largest['total']:,}",
            f"{limit:,}",
        ),
        {"limit": limit, "rows": rows},
    )


def _wall_clock_check(elapsed: float, config: GpuCheckConfig) -> CheckResult:
    limit = config.budget.max_total_seconds
    details = {"elapsed_seconds": elapsed, "limit_seconds": limit}
    if limit is None:
        return CheckResult(
            "budget.wall_clock", STATUS_PASS, "sanity run took %.1fs (no limit set)" % elapsed, details
        )
    if elapsed > limit:
        return CheckResult(
            "budget.wall_clock",
            STATUS_WARN,
            "sanity run took %.1fs, over the %.0fs hackathon budget; shrink batch_sizes, "
            "measure_steps or the variant list" % (elapsed, limit),
            details,
        )
    return CheckResult(
        "budget.wall_clock",
        STATUS_PASS,
        "sanity run took %.1fs, within the %.0fs budget" % (elapsed, limit),
        details,
    )


def run_gpu_checks(
    config: GpuCheckConfig,
    *,
    include_smoke: bool = True,
    include_benchmarks: bool = True,
    include_determinism: bool = True,
) -> GpuReport:
    """Run the configured checks and return the assembled report.

    The report is returned even when checks fail -- failures are data, not
    exceptions. A genuine CUDA fault (anything other than a measured OOM) still
    propagates as a :class:`~src.gpu.errors.GpuCheckError` with full context,
    because a broken environment must not be reported as a soft finding.
    """
    started = time.perf_counter()
    environment = collect_environment(config.device)
    report = GpuReport(config=config.to_dict(), environment=environment)
    for result in environment_checks(environment, config):
        report.add(result)

    device = environment["resolved_device"]
    if config.model.backbone_source == "stub":
        report.notes.append(STUB_NOTE)
    if not is_cuda(device):
        report.notes.append(CPU_NOTE)

    if not is_cuda(device) and not config.allow_cpu:
        report.add(
            CheckResult(
                "run.execution",
                STATUS_SKIP,
                "no CUDA device and allow_cpu=false: smoke, benchmark and determinism were "
                "not run, so nothing here validates a GPU",
                {"resolved_device": device},
            )
        )
        report.duration_seconds = time.perf_counter() - started
        return report

    benchmark_rows: List[Dict[str, Any]] = []
    for backbone, architecture in config.model.variants:
        model = build_detector(backbone, architecture, config.model)
        try:
            report.parameters.append(
                parameter_row(model, config, backbone=backbone, architecture=architecture)
            )
            if include_smoke:
                for result in smoke_variant(
                    backbone, architecture, config, device=device, model=model
                ):
                    report.add(result)
            if include_benchmarks:
                benchmark_rows.extend(
                    benchmark_variant(
                        model,
                        config,
                        backbone=backbone,
                        architecture=architecture,
                        device=device,
                    )
                )
        finally:
            # Release before the next variant; a pretrained I-JEPA plus a
            # fusion branch is too large to keep several alive at once.
            model.to("cpu")
            del model

    report.add(_parameter_check(report.parameters, config))

    if include_benchmarks:
        report.benchmarks = benchmark_rows
        report.recommendations = budget_recommendations(benchmark_rows, config)
        for result in benchmark_checks(benchmark_rows, config, device=device):
            report.add(result)
    else:
        report.add(
            CheckResult("benchmark.completed", STATUS_SKIP, "benchmarks not requested", {})
        )

    if include_determinism:
        payload, results = determinism_report(config, device=device)
        report.determinism = payload
        for result in results:
            report.add(result)
    else:
        report.add(
            CheckResult("determinism.repeat_run", STATUS_SKIP, "determinism not requested", {})
        )

    report.duration_seconds = time.perf_counter() - started
    report.add(_wall_clock_check(report.duration_seconds, config))
    return report


def run_benchmark_report(config: GpuCheckConfig) -> GpuReport:
    """Benchmark-only variant used by ``scripts/benchmark_gpu.py``."""
    return run_gpu_checks(
        config, include_smoke=False, include_benchmarks=True, include_determinism=False
    )


__all__ = ["run_gpu_checks", "run_benchmark_report", "STUB_NOTE", "CPU_NOTE"]
