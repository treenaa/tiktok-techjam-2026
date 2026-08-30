"""Visual baseline and dual-domain fusion detectors."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .encoders import ModelError, VisionEncoder
from .forensic import ForensicBranch
from .heads import BinaryClassificationHead


class VisualDetector(nn.Module):
    """Pretrained visual encoder plus the shared binary classifier head."""

    architecture = "visual"

    def __init__(
        self,
        encoder: VisionEncoder,
        *,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = BinaryClassificationHead(
            encoder.feature_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.feature_dim = encoder.feature_dim

    def forward_features(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def forward(self, images: Tensor) -> Tensor:
        # Raw logits only. Sigmoid belongs in training metrics / inference.
        return self.classifier(self.forward_features(images))


class FusionDetector(nn.Module):
    """Concatenate visual structure with low-level frequency evidence."""

    architecture = "fusion"

    def __init__(
        self,
        encoder: VisionEncoder,
        forensic: ForensicBranch,
        *,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.forensic = forensic
        self.feature_dim = encoder.feature_dim + forensic.output_dim
        self.classifier = BinaryClassificationHead(
            self.feature_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward_features(self, images: Tensor) -> Dict[str, Tensor]:
        visual = self.encoder(images)
        forensic = self.forensic(images)
        if visual.shape[0] != forensic.shape[0]:
            raise ModelError("visual and forensic branches returned different batch sizes")
        return {
            "visual": visual,
            "forensic": forensic,
            "fused": torch.cat((visual, forensic), dim=1),
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.forward_features(images)["fused"])
