"""Turn "this will not fit" into "here is what to run instead".

A benchmark that only reports an OOM leaves the next person guessing. For every
configuration that overflows the VRAM budget this module derives concrete,
measured alternatives -- the largest batch size that actually fit in the same
sweep, the gradient-accumulation factor that preserves the intended effective
batch, whether mixed precision was measured to help -- plus the standard
memory levers, each labelled with what it costs.

Nothing here assumes a bigger GPU is available. That is the point.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .config import GpuCheckConfig
from .report import STATUS_PASS, STATUS_SKIP, STATUS_WARN, CheckResult

#: Levers that need a change outside this subsystem. They are proposals, not
#: something the GPU checks may implement on another agent's behalf.
EXTERNAL_LEVERS = (
    "gradient checkpointing on the encoder (a Hugging Face backbone exposes "
    "gradient_checkpointing_enable(); it trades roughly 30-40% step time for a large "
    "activation saving) -- this needs a hook in src/models, so raise it with the model owner "
    "rather than patching it locally",
    "keep freeze_backbone: true (the phase-1 linear probe): a frozen encoder runs under "
    "no_grad and stores no activations for backward, which is the single largest saving "
    "available here",
)


def _variant_key(row: Dict[str, Any]) -> tuple:
    return (row["backbone"], row["architecture"], row["mode"], row["precision"])


def _fits(row: Dict[str, Any]) -> bool:
    return row.get("status") == "ok" and bool(row.get("fits_vram_budget"))


def largest_fitting_batch(
    rows: Sequence[Dict[str, Any]], reference: Dict[str, Any]
) -> Optional[int]:
    """Largest measured batch size that fit, for the same variant/mode/precision."""
    candidates = [
        row["batch_size"]
        for row in rows
        if _variant_key(row) == _variant_key(reference) and _fits(row)
    ]
    return max(candidates) if candidates else None


def _amp_alternative(
    rows: Sequence[Dict[str, Any]], reference: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """An AMP row for the same variant/mode/batch that did fit, if one exists."""
    if reference["precision"] != "fp32":
        return None
    for row in rows:
        if (
            row["backbone"] == reference["backbone"]
            and row["architecture"] == reference["architecture"]
            and row["mode"] == reference["mode"]
            and row["batch_size"] == reference["batch_size"]
            and row["precision"] == "amp"
            and _fits(row)
        ):
            return row
    return None


def suggest_fallbacks(
    row: Dict[str, Any], rows: Sequence[Dict[str, Any]], config: GpuCheckConfig
) -> List[str]:
    """Concrete alternatives for one configuration that does not fit.

    Ordered cheapest-to-adopt first. Every batch-size number quoted here was
    measured in the same sweep, never estimated.
    """
    suggestions: List[str] = []
    fitting = largest_fitting_batch(rows, row)
    target = config.budget.target_train_batch_size or row["batch_size"]

    if fitting is not None and fitting < row["batch_size"]:
        suggestions.append(
            "drop batch_size %d -> %d (the largest size measured to fit in this sweep)"
            % (row["batch_size"], fitting)
        )
        steps = max(1, math.ceil(target / fitting))
        if steps > 1:
            suggestions.append(
                "preserve the effective batch of %d with gradient accumulation: "
                "%d micro-batches of %d (optimizer.step() every %d forward/backward passes)"
                % (target, steps, fitting, steps)
            )
    elif fitting is None:
        suggestions.append(
            "no measured batch size fit for this configuration; re-run the sweep with smaller "
            "--batch-sizes (for example 1 2 4) to find the ceiling before committing to a run"
        )

    amp_row = _amp_alternative(rows, row)
    if amp_row is not None:
        suggestions.append(
            "switch this configuration to mixed precision: amp fp16 at batch %d was measured to "
            "fit (%s peak) where fp32 did not"
            % (amp_row["batch_size"], _mib(amp_row.get("peak_memory_reserved_bytes")))
        )
    elif row["precision"] == "fp32":
        suggestions.append(
            "enable mixed precision (training.amp: true, fp16): activations roughly halve, and "
            "the AMP NaN check in this suite already covers the risk"
        )

    if config.model.image_size > 224:
        suggestions.append(
            "reduce model.image_size from %d to 224, the size the registered backbones were "
            "trained at; attention memory grows with the square of the token count"
            % config.model.image_size
        )
    suggestions.extend(EXTERNAL_LEVERS)
    return suggestions


def _mib(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.0f MiB" % (float(value) / (1024 * 1024))


def budget_recommendations(
    rows: Sequence[Dict[str, Any]], config: GpuCheckConfig
) -> List[Dict[str, Any]]:
    """One recommendation record per configuration that overflows the budget."""
    recommendations = []
    for row in rows:
        if row.get("status") == "skipped":
            continue
        if row.get("fits_vram_budget") is not False:
            continue
        recommendations.append(
            {
                "backbone": row["backbone"],
                "architecture": row["architecture"],
                "mode": row["mode"],
                "precision": row["precision"],
                "batch_size": row["batch_size"],
                "status": row["status"],
                "peak_memory_reserved_bytes": row.get("peak_memory_reserved_bytes"),
                "vram_budget_bytes": row.get("vram_budget_bytes"),
                "budget_fraction": row.get("budget_fraction"),
                "fallbacks": suggest_fallbacks(row, rows, config),
            }
        )
    return recommendations


def vram_budget_check(
    rows: Sequence[Dict[str, Any]],
    recommendations: Sequence[Dict[str, Any]],
    config: GpuCheckConfig,
) -> CheckResult:
    """Flag every configuration that would not fit the target GPU."""
    budgets = {
        row.get("vram_budget_bytes") for row in rows if row.get("vram_budget_bytes") is not None
    }
    budget = min(budgets) if budgets else None
    details: Dict[str, Any] = {
        "vram_budget_bytes": budget,
        "vram_budget_mb": config.budget.vram_budget_mb,
        "headroom_fraction": config.budget.vram_headroom_fraction,
        "recommendations": list(recommendations),
    }
    if budget is None:
        return CheckResult(
            "benchmark.vram_budget",
            STATUS_SKIP,
            "no VRAM budget to judge against: set budget.vram_budget_mb to plan for the target "
            "GPU from a machine that does not have one",
            details,
        )
    fitting = [row for row in rows if _fits(row)]
    if not recommendations:
        return CheckResult(
            "benchmark.vram_budget",
            STATUS_PASS,
            "all %d measured configuration(s) fit within %s of the %s budget"
            % (len(fitting), "%.0f%%" % (config.budget.vram_headroom_fraction * 100), _mib(budget)),
            details,
        )
    return CheckResult(
        "benchmark.vram_budget",
        STATUS_WARN,
        "%d configuration(s) do not fit the %s budget: %s -- see the fallback recommendations"
        % (
            len(recommendations),
            _mib(budget),
            ", ".join(
                "%s/%s %s %s b%s"
                % (
                    item["backbone"],
                    item["architecture"],
                    item["mode"],
                    item["precision"],
                    item["batch_size"],
                )
                for item in recommendations
            ),
        ),
        details,
    )


__all__ = [
    "EXTERNAL_LEVERS",
    "budget_recommendations",
    "largest_fitting_batch",
    "suggest_fallbacks",
    "vram_budget_check",
]
