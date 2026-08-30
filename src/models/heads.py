"""Small, comparable binary classification heads."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .encoders import ModelError


class BinaryClassificationHead(nn.Module):
    """Layer-normalised MLP that returns one raw AIGC logit per sample."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = None if hidden_dim in (None, 0) else int(hidden_dim)
        if self.hidden_dim is not None and self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive, zero, or None")
        if self.hidden_dim is None:
            self.layers = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Dropout(float(dropout)),
                nn.Linear(self.input_dim, 1),
            )
        else:
            self.layers = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

    def forward(self, features: Tensor) -> Tensor:
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ModelError("classification head expects features shaped (B, D)")
        if features.shape[1] != self.input_dim:
            raise ModelError(
                "classification head expected width %d, got %d"
                % (self.input_dim, features.shape[1])
            )
        return self.layers(features).squeeze(-1)
