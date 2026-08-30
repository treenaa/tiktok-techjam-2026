"""Validated training configuration with backwards-compatible experiment keys."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


class TrainingError(ValueError):
    """Raised when a training run would be invalid or irreproducible."""


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9
    scheduler: str = "none"
    min_lr: float = 0.0
    clean_loss_weight: float = 1.0
    augmented_loss_weight: float = 1.0
    consistency_weight: float = 0.0
    positive_class_weight: Optional[float] = None
    augment: str = "none"
    augment_families: Optional[Tuple[str, ...]] = None
    augment_weights: Optional[Tuple[float, ...]] = None
    augment_identity_probability: float = 0.0
    augment_operations: Any = 1
    max_grad_norm: Optional[float] = 1.0
    amp: bool = False
    threshold_metric: str = "f1"
    early_stopping_monitor: str = "val_auroc"
    early_stopping_mode: str = "max"
    early_stopping_patience: Optional[int] = 3
    early_stopping_min_delta: float = 0.0
    deterministic_torch: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise TrainingError("epochs and batch_size must be positive")
        if self.num_workers < 0:
            raise TrainingError("num_workers cannot be negative")
        if self.optimizer not in {"adamw", "sgd"}:
            raise TrainingError("optimizer must be 'adamw' or 'sgd'")
        if self.scheduler not in {"none", "cosine"}:
            raise TrainingError("scheduler must be 'none' or 'cosine'")
        if self.lr <= 0 or self.weight_decay < 0 or self.min_lr < 0:
            raise TrainingError("lr must be positive; weight_decay/min_lr cannot be negative")
        if not 0 <= self.momentum < 1:
            raise TrainingError("momentum must be in [0, 1)")
        for name in ("clean_loss_weight", "augmented_loss_weight", "consistency_weight"):
            if float(getattr(self, name)) < 0:
                raise TrainingError("%s cannot be negative" % name)
        if self.clean_loss_weight == 0 and self.augmented_loss_weight == 0:
            raise TrainingError("at least one classification loss weight must be positive")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise TrainingError("positive_class_weight must be positive")
        if self.augment not in {"none", "competition"}:
            raise TrainingError("augment must be 'none' or 'competition'")
        if self.augment == "none" and self.consistency_weight > 0:
            raise TrainingError("consistency_weight requires augment='competition'")
        if self.augment == "none" and self.clean_loss_weight == 0:
            raise TrainingError("single-view training requires clean_loss_weight > 0")
        if not 0.0 <= self.augment_identity_probability <= 1.0:
            raise TrainingError("augment_identity_probability must be in [0, 1]")
        if self.augment_weights is not None and self.augment_families is None:
            raise TrainingError("augment_weights requires augment_families")
        if self.augment_weights is not None and len(self.augment_weights) != len(self.augment_families):
            raise TrainingError("augment_weights must match augment_families")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise TrainingError("max_grad_norm must be positive or None")
        if self.threshold_metric not in {"f1", "balanced_accuracy", "accuracy"}:
            raise TrainingError("threshold_metric must be f1, balanced_accuracy, or accuracy")
        if self.early_stopping_mode not in {"max", "min"}:
            raise TrainingError("early_stopping_mode must be 'max' or 'min'")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise TrainingError("early_stopping_patience cannot be negative")
        if self.early_stopping_min_delta < 0:
            raise TrainingError("early_stopping_min_delta cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "TrainingConfig":
        raw = dict(values or {})
        loss_name = raw.pop("loss", "bce_with_logits")
        if loss_name not in {"bce_with_logits", "BCEWithLogitsLoss"}:
            raise TrainingError("only BCEWithLogitsLoss is supported, got %r" % loss_name)
        # Model construction consumes this legacy shared-config key.
        raw.pop("freeze_backbone", None)
        early = raw.pop("early_stopping", {})
        if early in (None, False):
            raw["early_stopping_patience"] = None
        elif isinstance(early, Mapping):
            aliases = {
                "monitor": "early_stopping_monitor",
                "mode": "early_stopping_mode",
                "patience": "early_stopping_patience",
                "min_delta": "early_stopping_min_delta",
            }
            unknown_early = set(early) - set(aliases)
            if unknown_early:
                raise TrainingError("unknown early_stopping key(s) %s" % sorted(unknown_early))
            for old, new in aliases.items():
                if old in early:
                    raw[new] = early[old]
        else:
            raise TrainingError("early_stopping must be a mapping, false, or null")

        scheduler = raw.get("scheduler")
        if isinstance(scheduler, Mapping):
            scheduler = dict(scheduler)
            raw["scheduler"] = scheduler.pop("name", "none")
            if "min_lr" in scheduler:
                raw["min_lr"] = scheduler.pop("min_lr")
            if scheduler:
                raise TrainingError("unknown scheduler key(s) %s" % sorted(scheduler))
        if "augment" in raw:
            aliases = {"official": "competition", "robustness": "competition"}
            raw["augment"] = aliases.get(str(raw["augment"]).lower(), str(raw["augment"]).lower())
        for key in ("augment_families", "augment_weights"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise TrainingError("unknown training config key(s) %s" % sorted(unknown))
        return cls(**raw)
