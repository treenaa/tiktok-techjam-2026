"""Binary classification and clean/transformed consistency objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .config import TrainingError


@dataclass
class LossOutput:
    total: Tensor
    clean_classification: Tensor
    augmented_classification: Optional[Tensor]
    consistency: Optional[Tensor]

    def detached(self) -> Dict[str, Optional[float]]:
        return {
            "loss": float(self.total.detach()),
            "clean_classification_loss": float(self.clean_classification.detach()),
            "augmented_classification_loss": (
                float(self.augmented_classification.detach())
                if self.augmented_classification is not None
                else None
            ),
            "consistency_loss": (
                float(self.consistency.detach()) if self.consistency is not None else None
            ),
        }


class RobustBinaryObjective(nn.Module):
    """BCE(clean) + BCE(T(x)) + lambda*MSE(P(clean), P(T(x)))."""

    def __init__(
        self,
        *,
        clean_weight: float = 1.0,
        augmented_weight: float = 1.0,
        consistency_weight: float = 0.0,
        positive_class_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        for name, value in {
            "clean_weight": clean_weight,
            "augmented_weight": augmented_weight,
            "consistency_weight": consistency_weight,
        }.items():
            if float(value) < 0:
                raise TrainingError("%s cannot be negative" % name)
        if clean_weight == 0 and augmented_weight == 0:
            raise TrainingError("at least one classification weight must be positive")
        if positive_class_weight is not None and float(positive_class_weight) <= 0:
            raise TrainingError("positive_class_weight must be positive")
        self.clean_weight = float(clean_weight)
        self.augmented_weight = float(augmented_weight)
        self.consistency_weight = float(consistency_weight)
        weight = None if positive_class_weight is None else torch.tensor(float(positive_class_weight))
        self.register_buffer("positive_class_weight", weight)

    def classification_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        labels = labels.float()
        self._validate(logits, labels, "classification")
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.positive_class_weight,
        )

    @staticmethod
    def _validate(logits: Tensor, labels: Tensor, name: str) -> None:
        if not torch.is_tensor(logits) or logits.ndim != 1:
            raise TrainingError("%s logits must have shape (B,)" % name)
        if labels.ndim != 1 or labels.shape != logits.shape:
            raise TrainingError(
                "%s labels/logits shape mismatch: %s vs %s"
                % (name, tuple(labels.shape), tuple(logits.shape))
            )
        if not bool(torch.isfinite(logits).all()):
            raise TrainingError("%s logits contain NaN or infinity" % name)

    def forward(
        self,
        clean_logits: Tensor,
        labels: Tensor,
        augmented_logits: Optional[Tensor] = None,
    ) -> LossOutput:
        labels = labels.float()
        self._validate(clean_logits, labels, "clean")
        clean = self.classification_loss(clean_logits, labels)
        total = self.clean_weight * clean
        augmented = None
        consistency = None
        if augmented_logits is not None:
            self._validate(augmented_logits, labels, "augmented")
            augmented = self.classification_loss(augmented_logits, labels)
            # Consistency follows the project hypothesis in probability space;
            # BCE still receives raw logits.
            consistency = nn.functional.mse_loss(
                torch.sigmoid(clean_logits), torch.sigmoid(augmented_logits)
            )
            total = (
                total
                + self.augmented_weight * augmented
                + self.consistency_weight * consistency
            )
        elif self.consistency_weight > 0:
            raise TrainingError("consistency_weight > 0 requires augmented logits")
        return LossOutput(total, clean, augmented, consistency)
