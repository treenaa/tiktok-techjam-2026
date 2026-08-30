"""Same-seed reproducibility on the GPU, with the caveats stated out loud.

Two runs of the same seed are executed in one process and compared. Bitwise
equality is not claimed: the sources of residual nondeterminism are enumerated
in :data:`CUDNN_CAVEATS` and the controls that were actually active are
recorded in the report, so nobody has to guess whether a difference was
expected.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.data import seed_everything

from .builders import (
    build_detector,
    build_objective,
    build_step_optimizer,
    synthetic_batch,
    variant_label,
)
from .config import GpuCheckConfig
from .errors import gpu_error_context
from .probes import inference_step, is_cuda, training_step
from .report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN, CheckResult

CUDNN_CAVEATS: Tuple[str, ...] = (
    "cuDNN selects convolution algorithms by benchmarking; with "
    "torch.backends.cudnn.benchmark=True the winning algorithm -- and therefore the "
    "last bits of the result -- can differ between processes and even between shapes.",
    "Several CUDA kernels (scatter-style reductions, some backward passes) accumulate with "
    "atomics, whose summation order is not fixed. torch.use_deterministic_algorithms replaces "
    "them where a deterministic kernel exists and raises where none does.",
    "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) must be exported BEFORE the CUDA context is "
    "created. Setting it after the first CUDA call has no effect, so a run that sets it in "
    "Python may still be nondeterministic in cuBLAS reductions.",
    "TF32 matmuls on Ampere and later trade mantissa bits for speed. That is a precision "
    "choice rather than a determinism bug, but it makes GPU results differ from CPU results "
    "by far more than float32 rounding would suggest.",
    "These two runs share one process. A separate process can still differ if the autotuner "
    "cache, driver, or library versions differ; treat cross-process reproducibility as a "
    "separate claim that this check does not make.",
    "Multi-worker DataLoader ordering is out of scope here: the check feeds fixed synthetic "
    "tensors so that only kernel-level nondeterminism can move the numbers.",
)


def determinism_controls() -> Dict[str, Any]:
    """Snapshot every knob that decides how reproducible this run can be."""
    deterministic_algorithms = None
    try:
        deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    except AttributeError:  # pragma: no cover - very old torch
        pass
    return {
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "use_deterministic_algorithms": deterministic_algorithms,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _parameter_signature(model: nn.Module) -> Tensor:
    """Flatten initial weights so two runs' initialisations can be compared."""
    return torch.cat(
        [parameter.detach().float().reshape(-1).cpu() for parameter in model.parameters()]
    )


def _single_run(
    backbone: str,
    architecture: str,
    config: GpuCheckConfig,
    *,
    device: str,
) -> Dict[str, Any]:
    """Seed, build, forward, train one step, forward again."""
    seed_everything(config.seed, deterministic_torch=config.deterministic)
    model = build_detector(backbone, architecture, config.model).to(device)
    signature = _parameter_signature(model)

    batch = synthetic_batch(
        config.determinism.batch_size,
        config.model.image_size,
        device=device,
        generator=torch.Generator().manual_seed(config.seed),
        paired=True,
    )
    before = inference_step(model, batch, device=device, amp=False, capture_logits=True)
    objective = build_objective(paired=True).to(device)
    optimizer = build_step_optimizer(model, config)
    step = training_step(
        model,
        objective,
        optimizer,
        batch,
        device=device,
        amp=False,
        capture_logits=True,
    )
    after = inference_step(model, batch, device=device, amp=False, capture_logits=True)
    return {
        "init_signature": signature,
        "logits_before": before.logits,
        "logits_after": after.logits,
        "loss": step.loss,
    }


def _max_abs_diff(first: Optional[Tensor], second: Optional[Tensor]) -> Optional[float]:
    if first is None or second is None:
        return None
    if first.shape != second.shape:
        return float("inf")
    return float((first - second).abs().max())


def determinism_report(
    config: GpuCheckConfig, *, device: str
) -> Tuple[Dict[str, Any], List[CheckResult]]:
    """Run each variant twice from the same seed and compare the results."""
    determinism = config.determinism
    if not determinism.enabled:
        return (
            {"enabled": False, "caveats": list(CUDNN_CAVEATS)},
            [CheckResult("determinism.repeat_run", STATUS_SKIP, "disabled in config", {})],
        )

    previous_deterministic = None
    try:
        previous_deterministic = bool(torch.are_deterministic_algorithms_enabled())
    except AttributeError:  # pragma: no cover - very old torch
        pass
    # seed_everything(deterministic_torch=True) mutates process-global cuDNN
    # flags; restore them so this check cannot change how a later one behaves.
    previous_cudnn = (
        getattr(torch.backends.cudnn, "deterministic", None),
        getattr(torch.backends.cudnn, "benchmark", None),
    )

    variants: List[Dict[str, Any]] = []
    try:
        if config.deterministic:
            # warn_only keeps a missing deterministic kernel from aborting the
            # whole report; the warning still surfaces in stderr.
            torch.use_deterministic_algorithms(True, warn_only=True)
        for backbone, architecture in config.model.variants:
            label = variant_label(backbone, architecture)
            with gpu_error_context(backbone=backbone, architecture=architecture, device=device):
                first = _single_run(backbone, architecture, config, device=device)
                second = _single_run(backbone, architecture, config, device=device)
            init_diff = _max_abs_diff(first["init_signature"], second["init_signature"])
            logit_diff = _max_abs_diff(first["logits_before"], second["logits_before"])
            post_diff = _max_abs_diff(first["logits_after"], second["logits_after"])
            loss_diff = (
                abs(first["loss"] - second["loss"])
                if first["loss"] is not None and second["loss"] is not None
                else None
            )
            variants.append(
                {
                    "variant": label,
                    "backbone": backbone,
                    "architecture": architecture,
                    "max_abs_init_diff": init_diff,
                    "max_abs_logit_diff": logit_diff,
                    "max_abs_logit_diff_after_step": post_diff,
                    "abs_loss_diff": loss_diff,
                    "logit_tolerance": determinism.logit_tolerance,
                    "loss_tolerance": determinism.loss_tolerance,
                    "within_tolerance": _within(
                        logit_diff, post_diff, loss_diff, determinism
                    ),
                }
            )
    finally:
        controls_during_run = determinism_controls()
        if config.deterministic and previous_deterministic is not None:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)
        if previous_cudnn[0] is not None:
            torch.backends.cudnn.deterministic = previous_cudnn[0]
            torch.backends.cudnn.benchmark = previous_cudnn[1]

    payload = {
        "enabled": True,
        "runs": 2,
        "device": device,
        "seed": config.seed,
        "deterministic_requested": bool(config.deterministic),
        # Snapshotted while the runs were happening, not after the restore.
        "controls": controls_during_run,
        "variants": variants,
        "caveats": list(CUDNN_CAVEATS),
    }
    return payload, _determinism_checks(payload, config, device=device)


def _within(
    logit_diff: Optional[float],
    post_diff: Optional[float],
    loss_diff: Optional[float],
    determinism: Any,
) -> bool:
    values = [
        (logit_diff, determinism.logit_tolerance),
        (post_diff, determinism.logit_tolerance),
        (loss_diff, determinism.loss_tolerance),
    ]
    return all(value is not None and value <= tolerance for value, tolerance in values)


def _determinism_checks(
    payload: Dict[str, Any], config: GpuCheckConfig, *, device: str
) -> List[CheckResult]:
    variants = payload["variants"]
    failures = [row for row in variants if not row["within_tolerance"]]
    worst = max(
        (row["max_abs_logit_diff"] or 0.0 for row in variants),
        default=0.0,
    )
    results = [
        CheckResult(
            "determinism.repeat_run",
            STATUS_PASS if not failures else STATUS_FAIL,
            (
                "%d variant(s) reproduced within tolerance (worst logit diff %.3g)"
                % (len(variants), worst)
                if not failures
                else "%d variant(s) exceeded tolerance: %s"
                % (
                    len(failures),
                    ", ".join(
                        "%s (logits %.3g, loss %s)"
                        % (
                            row["variant"],
                            row["max_abs_logit_diff"] or float("nan"),
                            "%.3g" % row["abs_loss_diff"]
                            if row["abs_loss_diff"] is not None
                            else "n/a",
                        )
                        for row in failures
                    ),
                )
            ),
            {"variants": variants, "seed": config.seed},
        )
    ]

    init_failures = [
        row
        for row in variants
        if row["max_abs_init_diff"] is None or row["max_abs_init_diff"] > 0
    ]
    results.append(
        CheckResult(
            "determinism.seeded_init",
            STATUS_PASS if not init_failures else STATUS_FAIL,
            (
                "seed %d reproduces identical model initialisation" % config.seed
                if not init_failures
                else "model initialisation differs between runs for %s"
                % ", ".join(row["variant"] for row in init_failures)
            ),
            {"seed": config.seed},
        )
    )

    controls = payload["controls"]
    if not is_cuda(device):
        results.append(
            CheckResult(
                "determinism.controls",
                STATUS_PASS,
                "run executed on %s; the cuDNN/cuBLAS/TF32 caveats below apply only to CUDA "
                "and were not exercised here" % device,
                {"controls": controls, "relaxed": [], "caveats": list(CUDNN_CAVEATS)},
            )
        )
        return results

    relaxed = []
    if controls["cudnn_benchmark"]:
        relaxed.append("cudnn.benchmark=True (algorithm choice may vary)")
    if not controls["cudnn_deterministic"]:
        relaxed.append("cudnn.deterministic=False")
    if controls["matmul_tf32"] or controls["cudnn_tf32"]:
        relaxed.append("TF32 enabled (reduced matmul precision)")
    if not controls["cublas_workspace_config"]:
        relaxed.append("CUBLAS_WORKSPACE_CONFIG unset")
    results.append(
        CheckResult(
            "determinism.controls",
            STATUS_PASS if not relaxed else STATUS_WARN,
            (
                "all nondeterminism controls are tightened"
                if not relaxed
                else "nondeterminism remains possible: %s -- see the report's caveats"
                % "; ".join(relaxed)
            ),
            {"controls": controls, "relaxed": relaxed, "caveats": list(CUDNN_CAVEATS)},
        )
    )
    return results


__all__ = ["CUDNN_CAVEATS", "determinism_controls", "determinism_report"]
