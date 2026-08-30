"""Throughput, VRAM and time-to-first-batch measurement.

Every timing brackets itself with a device synchronisation, because CUDA work
is queued asynchronously and an unsynchronised timer measures Python, not the
GPU. Memory figures come from ``torch.cuda`` allocator statistics, which are
what actually determines whether a configuration fits.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .builders import (
    build_detector,
    build_objective,
    build_step_optimizer,
    synthetic_batch,
)
from .config import GpuCheckConfig
from .errors import GpuCheckError, is_out_of_memory
from .fallbacks import budget_recommendations, vram_budget_check
from .probes import (
    AMP_DTYPES,
    create_scaler,
    inference_step,
    is_cuda,
    synchronize,
    training_step,
)
from .report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN, CheckResult


def _memory_snapshot(device: str) -> Dict[str, Optional[int]]:
    """Allocator statistics, or ``None`` everywhere when this is not CUDA."""
    if not is_cuda(device):
        return {
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "steady_memory_allocated_bytes": None,
            "device_total_memory_bytes": None,
        }
    index = torch.device(device)
    return {
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "steady_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "device_total_memory_bytes": int(
            torch.cuda.get_device_properties(index).total_memory
        ),
    }


def _reset_memory(device: str) -> None:
    if is_cuda(device):
        index = torch.device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)


def planning_budget_bytes(config: GpuCheckConfig, device_total: Optional[int]) -> Optional[int]:
    """The VRAM ceiling a configuration must fit inside.

    An explicit ``budget.vram_budget_mb`` always wins, even when the measuring
    device is larger: the point is to plan for the GPU the team actually has,
    not for whatever happened to run the sweep. With no explicit budget, the
    measured device's own capacity is used, and the smaller of the two is taken
    when both are known.
    """
    configured = config.budget.vram_budget_bytes
    if configured is None:
        return device_total
    if device_total is None:
        return configured
    return min(configured, device_total)


def measure_config(
    model: nn.Module,
    config: GpuCheckConfig,
    *,
    backbone: str,
    architecture: str,
    device: str,
    batch_size: int,
    precision: str,
    mode: str,
) -> Dict[str, Any]:
    """Benchmark one (variant, batch size, precision, mode) point.

    Out-of-memory is a *result*, not a crash: the row comes back with
    ``status='oom'`` and the full allocator message, because flagging configs
    that will not fit is one of this subsystem's jobs. Any other CUDA error
    propagates as a :class:`GpuCheckError` carrying the same context.
    """
    benchmark = config.benchmark
    amp = precision == "amp"
    amp_dtype = AMP_DTYPES[benchmark.amp_dtype]
    row: Dict[str, Any] = {
        "backbone": backbone,
        "architecture": architecture,
        "mode": mode,
        "precision": precision,
        "batch_size": int(batch_size),
        "image_size": config.model.image_size,
        "device": device,
        "backbone_source": config.model.backbone_source,
        "warmup_steps": benchmark.warmup_steps,
        "measure_steps": benchmark.measure_steps,
        "status": "ok",
    }
    if amp and not is_cuda(device):
        row["status"] = "skipped"
        row["reason"] = "AMP requires CUDA"
        return row

    model.to(device)
    objective = build_objective(paired=False).to(device)
    optimizer = build_step_optimizer(model, config) if mode == "train" else None
    scaler = create_scaler(device, enabled=amp) if mode == "train" else None
    generator = torch.Generator().manual_seed(config.seed)

    def make_batch() -> Dict[str, torch.Tensor]:
        return synthetic_batch(
            batch_size,
            config.model.image_size,
            device=device,
            generator=generator,
            paired=False,
        )

    def run_step(check_finite: bool) -> None:
        batch = make_batch()
        if mode == "train":
            training_step(
                model,
                objective,
                optimizer,
                batch,
                device=device,
                amp=amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                check_finite=check_finite,
            )
        else:
            inference_step(
                model,
                batch,
                device=device,
                amp=amp,
                amp_dtype=amp_dtype,
                check_finite=check_finite,
            )

    try:
        _reset_memory(device)
        # Time to first batch deliberately includes CUDA context setup, kernel
        # autotuning and the host-to-device copy: that is the latency a user
        # sees before a run appears to start.
        synchronize(device)
        first_started = time.perf_counter()
        run_step(check_finite=True)
        synchronize(device)
        row["time_to_first_batch_seconds"] = time.perf_counter() - first_started

        for _ in range(benchmark.warmup_steps):
            run_step(check_finite=False)
        synchronize(device)
        _reset_memory(device)

        started = time.perf_counter()
        for _ in range(benchmark.measure_steps):
            run_step(check_finite=False)
        synchronize(device)
        elapsed = time.perf_counter() - started
    except GpuCheckError as exc:
        cause = exc.__cause__ or exc
        if not is_out_of_memory(cause):
            raise
        _reset_memory(device)
        row["status"] = "oom"
        row["error"] = str(exc)
        row.update(_memory_snapshot(device))
        row["vram_budget_bytes"] = planning_budget_bytes(
            config, row.get("device_total_memory_bytes")
        )
        row["fits_vram_budget"] = False
        return row

    row["total_seconds"] = elapsed
    row["seconds_per_step"] = elapsed / benchmark.measure_steps
    row["images_per_second"] = (batch_size * benchmark.measure_steps) / elapsed if elapsed else None
    memory = _memory_snapshot(device)
    row.update(memory)
    total = memory["device_total_memory_bytes"]
    peak = memory["peak_memory_reserved_bytes"]
    row["peak_reserved_fraction"] = (peak / total) if (total and peak is not None) else None
    budget = planning_budget_bytes(config, total)
    row["vram_budget_bytes"] = budget
    row["budget_fraction"] = (peak / budget) if (budget and peak is not None) else None
    row["fits_vram_budget"] = (
        None
        if row["budget_fraction"] is None
        else bool(row["budget_fraction"] <= config.budget.vram_headroom_fraction)
    )
    return row


def benchmark_variant(
    model: nn.Module,
    config: GpuCheckConfig,
    *,
    backbone: str,
    architecture: str,
    device: str,
) -> List[Dict[str, Any]]:
    """Every batch size x precision x mode point for one already-built model."""
    rows: List[Dict[str, Any]] = []
    for mode in config.benchmark.modes:
        for precision in config.benchmark.precisions:
            for batch_size in config.benchmark.batch_sizes:
                rows.append(
                    measure_config(
                        model,
                        config,
                        backbone=backbone,
                        architecture=architecture,
                        device=device,
                        batch_size=batch_size,
                        precision=precision,
                        mode=mode,
                    )
                )
    return rows


def run_benchmarks(
    config: GpuCheckConfig, *, device: str
) -> Tuple[List[Dict[str, Any]], List[CheckResult]]:
    """Sweep every configured variant x batch size x precision x mode.

    One model is alive at a time: a pretrained I-JEPA plus a fusion branch is
    large enough that holding every variant would exhaust host memory before
    the GPU ever complained.
    """
    rows: List[Dict[str, Any]] = []
    for backbone, architecture in config.model.variants:
        model = build_detector(backbone, architecture, config.model)
        try:
            rows.extend(
                benchmark_variant(
                    model,
                    config,
                    backbone=backbone,
                    architecture=architecture,
                    device=device,
                )
            )
        finally:
            model.to("cpu")
            del model
            _reset_memory(device)
    return rows, benchmark_checks(rows, config, device=device)


def _row_label(row: Dict[str, Any]) -> str:
    return "%s/%s %s %s b%s" % (
        row["backbone"],
        row["architecture"],
        row["mode"],
        row["precision"],
        row["batch_size"],
    )


def benchmark_checks(
    rows: List[Dict[str, Any]], config: GpuCheckConfig, *, device: str
) -> List[CheckResult]:
    """Turn measurements into budget verdicts."""
    budget = config.budget
    completed = [row for row in rows if row["status"] == "ok"]
    oom = [row for row in rows if row["status"] == "oom"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    results: List[CheckResult] = []

    results.append(
        CheckResult(
            "benchmark.completed",
            STATUS_PASS if completed else STATUS_FAIL,
            "%d of %d configurations measured (%d OOM, %d skipped)"
            % (len(completed), len(rows), len(oom), len(skipped)),
            {"measured": len(completed), "total": len(rows), "device": device},
        )
    )

    results.append(
        CheckResult(
            "benchmark.oom",
            STATUS_PASS if not oom else STATUS_WARN,
            (
                "no configuration ran out of memory"
                if not oom
                else "%d configuration(s) OOM on this device: %s"
                % (len(oom), ", ".join(_row_label(row) for row in oom))
            ),
            {"oom": [{"config": _row_label(row), "error": row.get("error")} for row in oom]},
        )
    )

    over_headroom = [
        row
        for row in completed
        if row.get("peak_reserved_fraction") is not None
        and row["peak_reserved_fraction"] > budget.vram_headroom_fraction
    ]
    if not is_cuda(device):
        results.append(
            CheckResult(
                "benchmark.vram_headroom",
                STATUS_SKIP,
                "VRAM accounting requires CUDA; measured on %s" % device,
                {},
            )
        )
    else:
        results.append(
            CheckResult(
                "benchmark.vram_headroom",
                STATUS_PASS if not over_headroom else STATUS_WARN,
                (
                    "every measured configuration stayed under %.0f%% of VRAM"
                    % (budget.vram_headroom_fraction * 100)
                    if not over_headroom
                    else "%d configuration(s) exceeded %.0f%% of VRAM: %s"
                    % (
                        len(over_headroom),
                        budget.vram_headroom_fraction * 100,
                        ", ".join(
                            "%s (%.0f%%)" % (_row_label(row), row["peak_reserved_fraction"] * 100)
                            for row in over_headroom
                        ),
                    )
                ),
                {
                    "headroom_fraction": budget.vram_headroom_fraction,
                    "over": [_row_label(row) for row in over_headroom],
                },
            )
        )

    recommendations = budget_recommendations(rows, config)
    results.append(vram_budget_check(rows, recommendations, config))

    results.append(
        _throughput_check(
            completed, "train", budget.min_train_images_per_second, "benchmark.throughput.train"
        )
    )
    results.append(
        _throughput_check(
            completed,
            "inference",
            budget.min_inference_images_per_second,
            "benchmark.throughput.inference",
        )
    )
    results.append(_first_batch_check(completed, budget.max_time_to_first_batch_seconds))
    return results


def _throughput_check(
    rows: List[Dict[str, Any]], mode: str, floor: Optional[float], name: str
) -> CheckResult:
    subset = [row for row in rows if row["mode"] == mode and row.get("images_per_second")]
    if not subset:
        return CheckResult(name, STATUS_SKIP, "no %s measurements" % mode, {})
    best = max(row["images_per_second"] for row in subset)
    fastest = max(subset, key=lambda row: row["images_per_second"])
    details = {
        "best_images_per_second": best,
        "best_config": _row_label(fastest),
        "floor": floor,
        "measurements": {_row_label(row): row["images_per_second"] for row in subset},
    }
    if floor is None:
        return CheckResult(
            name,
            STATUS_PASS,
            "best %s throughput %.1f img/s (%s); no floor configured"
            % (mode, best, _row_label(fastest)),
            details,
        )
    below = [row for row in subset if row["images_per_second"] < floor]
    if len(below) == len(subset):
        return CheckResult(
            name,
            STATUS_FAIL,
            "every %s configuration is below the %.1f img/s floor (best %.1f)"
            % (mode, floor, best),
            details,
        )
    return CheckResult(
        name,
        STATUS_PASS,
        "best %s throughput %.1f img/s (%s) clears the %.1f img/s floor"
        % (mode, best, _row_label(fastest), floor),
        details,
    )


def _first_batch_check(rows: List[Dict[str, Any]], limit: Optional[float]) -> CheckResult:
    subset = [row for row in rows if row.get("time_to_first_batch_seconds") is not None]
    name = "benchmark.time_to_first_batch"
    if not subset:
        return CheckResult(name, STATUS_SKIP, "no timed configurations", {})
    slowest = max(subset, key=lambda row: row["time_to_first_batch_seconds"])
    worst = slowest["time_to_first_batch_seconds"]
    details = {
        "worst_seconds": worst,
        "worst_config": _row_label(slowest),
        "limit_seconds": limit,
    }
    if limit is None:
        return CheckResult(
            name, STATUS_PASS, "slowest first batch %.2fs (%s)" % (worst, _row_label(slowest)), details
        )
    if worst > limit:
        return CheckResult(
            name,
            STATUS_WARN,
            "slowest first batch %.2fs (%s) exceeds the %.1fs budget"
            % (worst, _row_label(slowest), limit),
            details,
        )
    return CheckResult(
        name,
        STATUS_PASS,
        "slowest first batch %.2fs (%s) within the %.1fs budget"
        % (worst, _row_label(slowest), limit),
        details,
    )


__all__ = ["measure_config", "benchmark_variant", "run_benchmarks", "benchmark_checks"]
