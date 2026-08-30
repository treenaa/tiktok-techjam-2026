"""Construct the things under test: detectors, optimizers, synthetic batches.

This module owns no architecture. It calls ``src.models.create_model`` and
``src.training.build_optimizer`` so the GPU checks exercise the real code
paths, and it supplies a small download-free stand-in encoder so the subsystem
still runs when pretrained weights are unavailable or another agent's code is
mid-flight.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from src.models import create_model, parameter_report
from src.training import RobustBinaryObjective, TrainingConfig, build_optimizer

from .config import GpuCheckConfig, ModelConfig
from .errors import gpu_error_context


class StubVisionBackbone(nn.Module):
    """A tiny patch-embedding transformer shaped like a Hugging Face encoder.

    It returns ``last_hidden_state`` and ``pooler_output`` and exposes
    ``config.hidden_size``, which is the whole interface
    ``src.models.VisionEncoder`` depends on, so every registered pooling mode
    (``mean`` for I-JEPA, ``cls`` for DINOv2, ``pooler`` for CLIP) is genuinely
    exercised.

    There is no positional embedding, so the stub accepts any input size that
    is divisible by ``patch_size``. It is a plumbing and performance fixture:
    it carries no pretrained knowledge and must never be used for accuracy
    claims.
    """

    def __init__(
        self,
        hidden_size: int = 192,
        layers: int = 2,
        heads: int = 3,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size %d must be divisible by heads %d" % (hidden_size, heads))
        self.config = SimpleNamespace(hidden_size=int(hidden_size))
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is pointless here (norm_first=True disables it) and only
        # emits a warning that would clutter every report.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(layers), enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, pixel_values: Tensor) -> SimpleNamespace:
        if pixel_values.ndim != 4:
            raise ValueError("stub backbone expects a BCHW tensor, got %s" % (tuple(pixel_values.shape),))
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "stub backbone needs spatial dims divisible by patch_size %d, got %dx%d"
                % (self.patch_size, height, width)
            )
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat(
            (self.cls_token.to(patches.dtype).expand(patches.shape[0], -1, -1), patches), dim=1
        )
        hidden = self.layer_norm(self.encoder(tokens))
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=torch.tanh(self.pooler(hidden[:, 0])),
        )


def build_stub_backbone(model_config: ModelConfig) -> StubVisionBackbone:
    stub = model_config.stub
    return StubVisionBackbone(
        hidden_size=stub.hidden_size,
        layers=stub.layers,
        heads=stub.heads,
        patch_size=stub.patch_size,
    )


def build_detector(
    backbone: str,
    architecture: str,
    model_config: ModelConfig,
) -> nn.Module:
    """Build one detector variant through the real model factory.

    ``backbone_source='stub'`` injects :class:`StubVisionBackbone`;
    ``'pretrained'`` lets ``create_model`` download or restore the registered
    Hugging Face checkpoint, which is what a real GPU box should run.
    """
    context = {
        "backbone": backbone,
        "architecture": architecture,
        "backbone_source": model_config.backbone_source,
    }
    with gpu_error_context(**context):
        if model_config.backbone_source == "stub":
            stub = build_stub_backbone(model_config)
            return create_model(
                backbone=backbone,
                architecture=architecture,
                freeze_backbone=model_config.freeze_backbone,
                head_hidden_dim=model_config.head_hidden_dim,
                backbone_model=stub,
                backbone_feature_dim=model_config.stub.hidden_size,
            )
        return create_model(
            backbone=backbone,
            architecture=architecture,
            freeze_backbone=model_config.freeze_backbone,
            head_hidden_dim=model_config.head_hidden_dim,
            local_files_only=model_config.local_files_only,
        )


def build_objective(paired: bool) -> RobustBinaryObjective:
    """The real training objective; consistency is only valid when paired."""
    return RobustBinaryObjective(
        clean_weight=1.0,
        augmented_weight=1.0 if paired else 0.0,
        consistency_weight=0.5 if paired else 0.0,
    )


def build_step_optimizer(model: nn.Module, config: GpuCheckConfig) -> torch.optim.Optimizer:
    """Use the project's own optimizer builder so the step under test is real."""
    training_config = TrainingConfig(
        epochs=1,
        batch_size=config.smoke.batch_size,
        num_workers=0,
        amp=False,
    )
    with gpu_error_context(component="optimizer"):
        return build_optimizer(model, training_config)


def synthetic_batch(
    batch_size: int,
    image_size: int,
    *,
    device: str,
    generator: Optional[torch.Generator] = None,
    paired: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """A device-resident stand-in for one ``PairedViewDataset`` batch.

    Tensors are drawn on the CPU with an explicit generator and then moved, so
    two runs see byte-identical inputs regardless of the CUDA RNG state. This
    isolates kernel nondeterminism from data nondeterminism.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    shape = (batch_size, 3, image_size, image_size)
    clean = torch.randn(shape, generator=generator, dtype=torch.float32)
    labels = (
        torch.rand((batch_size,), generator=generator) < 0.5
    ).to(torch.float32)
    batch = {
        "clean": clean.to(device=device, dtype=dtype),
        "label": labels.to(device=device),
    }
    if paired:
        # A mild perturbation stands in for a competition transform; the point
        # is to exercise the paired code path, not to model any corruption.
        noise = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.05
        batch["augmented"] = (clean + noise).to(device=device, dtype=dtype)
    return batch


def parameter_row(
    model: nn.Module,
    config: GpuCheckConfig,
    *,
    backbone: str,
    architecture: str,
) -> Dict[str, Any]:
    """Parameter accounting for one variant, against the competition limit."""
    report = parameter_report(model, limit=config.budget.parameter_limit)
    return {
        "backbone": backbone,
        "architecture": architecture,
        "backbone_source": config.model.backbone_source,
        "backbone_model_id": getattr(model, "backbone_model_id", None),
        **report,
    }


def variant_label(backbone: str, architecture: str) -> str:
    return "%s/%s" % (backbone, architecture)


def describe_dtype(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


__all__ = [
    "StubVisionBackbone",
    "build_stub_backbone",
    "build_detector",
    "build_objective",
    "build_step_optimizer",
    "synthetic_batch",
    "parameter_row",
    "variant_label",
    "describe_dtype",
]
