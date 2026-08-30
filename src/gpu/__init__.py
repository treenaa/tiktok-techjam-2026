"""GPU environment validation, smoke tests, and performance benchmarking.

Run this before committing a rented GPU instance to a full training run. It
detects the environment rather than assuming one, proves the pipeline really
executes on the GPU, measures throughput and VRAM at realistic batch sizes,
and states its reproducibility caveats instead of hiding them.

    python scripts/gpu_check.py --config configs/gpu_check.yaml
    python scripts/benchmark_gpu.py --config configs/gpu_check.yaml

The subsystem owns no training data, model architecture, or training loop; it
only exercises them.
"""

from .benchmark import (
    benchmark_checks,
    benchmark_variant,
    measure_config,
    planning_budget_bytes,
    run_benchmarks,
)
from .builders import (
    StubVisionBackbone,
    build_detector,
    build_objective,
    build_step_optimizer,
    parameter_row,
    synthetic_batch,
    variant_label,
)
from .config import (
    BenchmarkConfig,
    BudgetConfig,
    DeterminismConfig,
    GpuCheckConfig,
    GpuConfigError,
    ModelConfig,
    SmokeConfig,
    StubConfig,
    config_from_mapping,
    load_gpu_config,
)
from .determinism import CUDNN_CAVEATS, determinism_controls, determinism_report
from .environment import (
    collect_environment,
    detect_display_adapters,
    environment_checks,
    has_nvidia_adapter,
    resolve_device,
)
from .errors import GpuCheckError, gpu_error_context, is_out_of_memory
from .fallbacks import (
    EXTERNAL_LEVERS,
    budget_recommendations,
    largest_fitting_batch,
    suggest_fallbacks,
    vram_budget_check,
)
from .probes import (
    StepOutcome,
    assert_on_device,
    autocast_context,
    create_scaler,
    inference_step,
    is_cuda,
    smoke_variant,
    synchronize,
    training_step,
)
from .report import (
    GPU_REPORT_SCHEMA_VERSION,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_WARN,
    CheckResult,
    GpuReport,
    overall_status,
    read_report,
    render_text,
    write_report,
)
from .runner import CPU_NOTE, STUB_NOTE, run_benchmark_report, run_gpu_checks

__all__ = [
    "GpuConfigError",
    "GpuCheckConfig",
    "ModelConfig",
    "SmokeConfig",
    "BenchmarkConfig",
    "DeterminismConfig",
    "BudgetConfig",
    "StubConfig",
    "config_from_mapping",
    "load_gpu_config",
    "GpuCheckError",
    "gpu_error_context",
    "is_out_of_memory",
    "collect_environment",
    "detect_display_adapters",
    "has_nvidia_adapter",
    "environment_checks",
    "resolve_device",
    "StubVisionBackbone",
    "build_detector",
    "build_objective",
    "build_step_optimizer",
    "parameter_row",
    "synthetic_batch",
    "variant_label",
    "StepOutcome",
    "assert_on_device",
    "autocast_context",
    "create_scaler",
    "inference_step",
    "is_cuda",
    "smoke_variant",
    "synchronize",
    "training_step",
    "measure_config",
    "benchmark_variant",
    "planning_budget_bytes",
    "run_benchmarks",
    "benchmark_checks",
    "EXTERNAL_LEVERS",
    "budget_recommendations",
    "largest_fitting_batch",
    "suggest_fallbacks",
    "vram_budget_check",
    "CUDNN_CAVEATS",
    "determinism_controls",
    "determinism_report",
    "GPU_REPORT_SCHEMA_VERSION",
    "STATUS_PASS",
    "STATUS_WARN",
    "STATUS_FAIL",
    "STATUS_SKIP",
    "CheckResult",
    "GpuReport",
    "overall_status",
    "render_text",
    "write_report",
    "read_report",
    "run_gpu_checks",
    "run_benchmark_report",
    "STUB_NOTE",
    "CPU_NOTE",
]
