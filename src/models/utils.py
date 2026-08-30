"""Parameter accounting and trainability checks."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn


PARAMETER_LIMIT = 2_000_000_000


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count unique parameters, guarding against shared-module double counts."""
    seen = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen or (trainable_only and not parameter.requires_grad):
            continue
        seen.add(identity)
        total += parameter.numel()
    return int(total)


def parameter_report(model: nn.Module, limit: int = PARAMETER_LIMIT) -> Dict[str, Any]:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    components = {
        name: {
            "total": count_parameters(module),
            "trainable": count_parameters(module, trainable_only=True),
        }
        for name, module in model.named_children()
    }
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": float(trainable / total) if total else 0.0,
        "limit": int(limit),
        "within_limit": bool(total < int(limit)),
        "components": components,
    }
