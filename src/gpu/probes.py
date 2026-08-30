"""Device-placement, forward/backward, and mixed-precision probes.

These functions are the primitives the smoke tests, the benchmark sweep and
the determinism check all share. They deliberately assert rather than warn: a
model that has quietly stayed on the CPU still trains, just fifty times slower,
and that is exactly the failure this subsystem exists to catch before someone
pays for a long run.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .builders import (
    build_detector,
    build_objective,
    build_step_optimizer,
    synthetic_batch,
    variant_label,
)
from .config import GpuCheckConfig
from .errors import GpuCheckError, gpu_error_context
from .report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN, CheckResult

AMP_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def device_type(device: str) -> str:
    return torch.device(device).type


def is_cuda(device: str) -> bool:
    return device_type(device) == "cuda"


def synchronize(device: str) -> None:
    """Wait for queued CUDA work; a no-op elsewhere.

    Every timing measurement in this package brackets itself with this, because
    CUDA launches are asynchronous and unsynchronised timings measure nothing
    but Python.
    """
    if is_cuda(device):
        torch.cuda.synchronize(torch.device(device))


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    """``cuda`` matches ``cuda:0``; an explicit index must match exactly."""
    if actual.type != expected.type:
        return False
    if expected.index is None:
        return True
    return (actual.index or 0) == expected.index


def assert_on_device(model: nn.Module, device: str) -> Dict[str, Any]:
    """Verify every parameter and buffer really lives on ``device``.

    Raises :class:`GpuCheckError` naming the offending tensors -- a module that
    was built after ``.to(device)``, or a buffer registered on the CPU, is the
    usual cause of a "GPU" run that is secretly half on the host.
    """
    expected = torch.device(device)
    misplaced: List[Dict[str, str]] = []
    parameters = 0
    buffers = 0
    for name, tensor in model.named_parameters():
        parameters += 1
        if not _same_device(tensor.device, expected):
            misplaced.append({"kind": "parameter", "name": name, "device": str(tensor.device)})
    for name, tensor in model.named_buffers():
        buffers += 1
        if not _same_device(tensor.device, expected):
            misplaced.append({"kind": "buffer", "name": name, "device": str(tensor.device)})
    summary = {
        "expected_device": str(expected),
        "parameters": parameters,
        "buffers": buffers,
        "misplaced": misplaced,
    }
    if misplaced:
        raise GpuCheckError(
            "%d tensor(s) are not on %s: %s"
            % (
                len(misplaced),
                device,
                ", ".join("%s(%s)" % (item["name"], item["device"]) for item in misplaced[:5]),
            ),
            {"device": device, "misplaced": len(misplaced)},
        )
    return summary


def autocast_context(device: str, *, enabled: bool, dtype: torch.dtype):
    """Autocast on CUDA only.

    ``src.training`` refuses ``amp=True`` off CUDA, so mirroring that here keeps
    the benchmark honest about what training will actually do.
    """
    if not enabled or not is_cuda(device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def create_scaler(device: str, *, enabled: bool):
    """A ``GradScaler`` across torch versions; disabled off CUDA."""
    enabled = bool(enabled and is_cuda(device))
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _validate_logits(logits: Any, batch_size: int, context: Dict[str, Any]) -> Tensor:
    """The detector contract: raw logits shaped ``(B,)``, finite, on-device."""
    if not torch.is_tensor(logits):
        raise GpuCheckError("model did not return a tensor of logits", context)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or logits.shape[0] != batch_size:
        raise GpuCheckError(
            "model must return logits shaped (B,), got %s" % (tuple(logits.shape),), context
        )
    return logits


@dataclass
class StepOutcome:
    """What one training or inference step produced."""

    loss: Optional[float]
    logits_finite: bool
    logits_device: str
    logits_dtype: str
    loss_device: Optional[str]
    grads_finite: Optional[bool]
    grad_norm: Optional[float]
    step_skipped: bool = False
    scaler_scale: Optional[float] = None
    logits: Optional[Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def training_step(
    model: nn.Module,
    objective: Any,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
    *,
    device: str,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Any = None,
    max_grad_norm: Optional[float] = 1.0,
    capture_logits: bool = False,
    check_finite: bool = True,
) -> StepOutcome:
    """One real forward/backward/step, mirroring ``train_one_epoch``.

    ``check_finite`` costs a device synchronisation, so the benchmark turns it
    off inside the timed loop and the smoke test leaves it on.
    """
    context = {"device": device, "amp": amp, "batch_size": int(batch["label"].shape[0])}
    model.train()
    labels = batch["label"]
    batch_size = int(labels.shape[0])
    augmented = batch.get("augmented")
    scaler = scaler if scaler is not None else create_scaler(device, enabled=amp)

    with gpu_error_context(**context):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=amp, dtype=amp_dtype):
            clean_logits = _validate_logits(model(batch["clean"]), batch_size, context)
            augmented_logits = (
                _validate_logits(model(augmented), batch_size, context)
                if augmented is not None
                else None
            )
            losses = objective(clean_logits.float(), labels.float(),
                               None if augmented_logits is None else augmented_logits.float())
        scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
        scaler.scale(losses.total).backward()
        grads_finite: Optional[bool] = None
        grad_norm: Optional[float] = None
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            # clip_grad_norm_ returns the total norm *before* clipping, which is
            # both the number worth reporting and a sufficient NaN/inf detector:
            # one non-finite gradient makes the whole norm non-finite.
            total_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
            if check_finite:
                grad_norm = float(total_norm)
                grads_finite = math.isfinite(grad_norm)
        elif check_finite:
            grads_finite, grad_norm = _grad_health(model)
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale()) if scaler.is_enabled() else None

    loss_value = float(losses.total.detach()) if check_finite else None
    return StepOutcome(
        loss=loss_value,
        logits_finite=(
            bool(torch.isfinite(clean_logits).all()) if check_finite else True
        ),
        logits_device=str(clean_logits.device),
        logits_dtype=str(clean_logits.dtype).replace("torch.", ""),
        loss_device=str(losses.total.device),
        grads_finite=grads_finite,
        grad_norm=grad_norm,
        # A skipped step means the scaler found inf/NaN gradients and backed
        # off. That is normal AMP behaviour on the first steps, not a defect.
        step_skipped=bool(
            scale_before is not None and scale_after is not None and scale_after < scale_before
        ),
        scaler_scale=scale_after,
        logits=clean_logits.detach().float().cpu() if capture_logits else None,
        extra={"scale_before": scale_before, "scale_after": scale_after},
    )


def _grad_health(model: nn.Module) -> Tuple[bool, Optional[float]]:
    total = 0.0
    finite = True
    seen = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        seen = True
        if not bool(torch.isfinite(parameter.grad).all()):
            finite = False
        total += float(parameter.grad.detach().float().pow(2).sum())
    return finite, (total ** 0.5 if seen else None)


def inference_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    *,
    device: str,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    capture_logits: bool = False,
    check_finite: bool = True,
) -> StepOutcome:
    """One ``torch.inference_mode`` forward, mirroring ``validate_one_epoch``."""
    context = {"device": device, "amp": amp, "batch_size": int(batch["label"].shape[0])}
    model.eval()
    batch_size = int(batch["label"].shape[0])
    with gpu_error_context(**context):
        with torch.inference_mode():
            with autocast_context(device, enabled=amp, dtype=amp_dtype):
                logits = _validate_logits(model(batch["clean"]), batch_size, context)
            captured = logits.detach().float().cpu() if capture_logits else None
            finite = bool(torch.isfinite(logits).all()) if check_finite else True
    return StepOutcome(
        loss=None,
        logits_finite=finite,
        logits_device=str(logits.device),
        logits_dtype=str(logits.dtype).replace("torch.", ""),
        loss_device=None,
        grads_finite=None,
        grad_norm=None,
        logits=captured,
    )


def placement_check(
    model: nn.Module, outcome: StepOutcome, device: str, label: str
) -> CheckResult:
    """No silent CPU fallback: weights, logits and loss must share the device."""
    expected = torch.device(device)
    placements = {
        "logits": outcome.logits_device,
        "loss": outcome.loss_device,
    }
    wrong = {
        name: value
        for name, value in placements.items()
        if value is not None and not _same_device(torch.device(value), expected)
    }
    if wrong:
        return CheckResult(
            "placement.%s" % label,
            STATUS_FAIL,
            "computation fell back off %s: %s"
            % (device, ", ".join("%s on %s" % item for item in sorted(wrong.items()))),
            {"expected_device": str(expected), **placements},
        )
    return CheckResult(
        "placement.%s" % label,
        STATUS_PASS,
        "model, logits and loss all on %s" % device,
        {"expected_device": str(expected), **placements},
    )


def smoke_variant(
    backbone: str,
    architecture: str,
    config: GpuCheckConfig,
    *,
    device: str,
    model: Optional[nn.Module] = None,
) -> List[CheckResult]:
    """Forward/backward one variant, then repeat under AMP if configured.

    Any CUDA error is re-raised with the backbone, architecture, batch size and
    precision attached; nothing is swallowed.
    """
    label = variant_label(backbone, architecture)
    smoke = config.smoke
    context = {
        "backbone": backbone,
        "architecture": architecture,
        "device": device,
        "batch_size": smoke.batch_size,
    }
    results: List[CheckResult] = []

    with gpu_error_context(**context):
        if model is None:
            model = build_detector(backbone, architecture, config.model)
        model.to(device)
        placement = assert_on_device(model, device)
        results.append(
            CheckResult(
                "placement.weights.%s" % label,
                STATUS_PASS,
                "%d parameters and %d buffers on %s"
                % (placement["parameters"], placement["buffers"], device),
                placement,
            )
        )

        objective = build_objective(smoke.paired).to(device)
        optimizer = build_step_optimizer(model, config)
        generator = torch.Generator().manual_seed(config.seed)

        fp32_outcome: Optional[StepOutcome] = None
        for step in range(smoke.steps):
            batch = synthetic_batch(
                smoke.batch_size,
                config.model.image_size,
                device=device,
                generator=generator,
                paired=smoke.paired,
            )
            fp32_outcome = training_step(
                model,
                objective,
                optimizer,
                batch,
                device=device,
                amp=False,
            )
            if not fp32_outcome.logits_finite or fp32_outcome.grads_finite is False:
                raise GpuCheckError(
                    "fp32 step %d produced non-finite logits or gradients" % step,
                    dict(context, precision="fp32"),
                )

    assert fp32_outcome is not None  # smoke.steps >= 1 is enforced by the config
    results.append(placement_check(model, fp32_outcome, device, "compute.%s" % label))
    results.append(
        CheckResult(
            "smoke.fp32.%s" % label,
            STATUS_PASS,
            "%d fp32 train step(s) at batch %d: loss %.4f, grad norm %s"
            % (
                smoke.steps,
                smoke.batch_size,
                fp32_outcome.loss,
                "%.4f" % fp32_outcome.grad_norm if fp32_outcome.grad_norm is not None else "n/a",
            ),
            {
                "loss": fp32_outcome.loss,
                "grad_norm": fp32_outcome.grad_norm,
                "logits_dtype": fp32_outcome.logits_dtype,
                "batch_size": smoke.batch_size,
                "paired": smoke.paired,
                "steps": smoke.steps,
            },
        )
    )
    results.append(_amp_check(model, config, device, label, context))
    return results


def _amp_check(
    model: nn.Module,
    config: GpuCheckConfig,
    device: str,
    label: str,
    context: Dict[str, Any],
) -> CheckResult:
    """Prove mixed precision does not silently produce NaNs.

    Two distinct failures are separated here: a *skipped* optimizer step is
    ``GradScaler`` doing its job after an fp16 overflow, whereas a non-finite
    loss is a genuine numerical failure that would poison training.
    """
    smoke = config.smoke
    name = "amp.%s" % label
    if not smoke.amp:
        return CheckResult(name, STATUS_SKIP, "AMP checks disabled in config", {})
    if not is_cuda(device):
        return CheckResult(
            name,
            STATUS_SKIP,
            "AMP requires CUDA; src.training also refuses amp=true off CUDA",
            {"device": device},
        )
    dtype = AMP_DTYPES[smoke.amp_dtype]
    amp_context = dict(context, precision="amp", amp_dtype=smoke.amp_dtype)
    objective = build_objective(smoke.paired).to(device)
    optimizer = build_step_optimizer(model, config)
    scaler = create_scaler(device, enabled=True)
    generator = torch.Generator().manual_seed(config.seed)
    losses: List[float] = []
    skipped = 0
    amp_outcome: Optional[StepOutcome] = None

    with gpu_error_context(**amp_context):
        for step in range(smoke.steps):
            batch = synthetic_batch(
                smoke.batch_size,
                config.model.image_size,
                device=device,
                generator=generator,
                paired=smoke.paired,
            )
            amp_outcome = training_step(
                model,
                objective,
                optimizer,
                batch,
                device=device,
                amp=True,
                amp_dtype=dtype,
                scaler=scaler,
            )
            losses.append(amp_outcome.loss)
            skipped += int(amp_outcome.step_skipped)
            if amp_outcome.loss is None or not math.isfinite(amp_outcome.loss):
                raise GpuCheckError(
                    "AMP step %d produced a non-finite loss (%s)" % (step, amp_outcome.loss),
                    amp_context,
                )
            if not amp_outcome.logits_finite:
                raise GpuCheckError("AMP step %d produced non-finite logits" % step, amp_context)

        # Apples to apples: the same weights and the same batch, evaluated once
        # in fp32 and once under autocast, with no optimizer step in between.
        comparison_batch = synthetic_batch(
            smoke.batch_size,
            config.model.image_size,
            device=device,
            generator=torch.Generator().manual_seed(config.seed + 1),
            paired=False,
        )
        reference = inference_step(
            model, comparison_batch, device=device, amp=False, capture_logits=True
        )
        under_amp = inference_step(
            model,
            comparison_batch,
            device=device,
            amp=True,
            amp_dtype=dtype,
            capture_logits=True,
        )

    details: Dict[str, Any] = {
        "amp_dtype": smoke.amp_dtype,
        "losses": losses,
        "steps": smoke.steps,
        "skipped_steps": skipped,
        "final_scaler_scale": amp_outcome.scaler_scale if amp_outcome else None,
        "logits_dtype": under_amp.logits_dtype,
    }

    if not under_amp.logits_finite:
        raise GpuCheckError("autocast inference produced non-finite logits", amp_context)
    divergence = float((reference.logits - under_amp.logits).abs().max())
    details["max_abs_logit_divergence_vs_fp32"] = divergence
    details["tolerance"] = config.budget.amp_logit_tolerance

    if skipped == smoke.steps:
        return CheckResult(
            name,
            STATUS_WARN,
            "every AMP step was skipped by GradScaler (%d/%d); loss stayed finite but no "
            "optimizer update landed" % (skipped, smoke.steps),
            details,
        )
    if divergence > config.budget.amp_logit_tolerance:
        return CheckResult(
            name,
            STATUS_WARN,
            "same weights, same batch: autocast logits differ from fp32 by %.4g "
            "(> tolerance %.4g)" % (divergence, config.budget.amp_logit_tolerance),
            details,
        )
    return CheckResult(
        name,
        STATUS_PASS,
        "%d AMP (%s) step(s) NaN-free, %d skipped by GradScaler; logit divergence %.4g"
        % (smoke.steps, smoke.amp_dtype, skipped, divergence),
        details,
    )


__all__ = [
    "AMP_DTYPES",
    "StepOutcome",
    "assert_on_device",
    "autocast_context",
    "create_scaler",
    "device_type",
    "inference_step",
    "is_cuda",
    "placement_check",
    "smoke_variant",
    "synchronize",
    "training_step",
]
