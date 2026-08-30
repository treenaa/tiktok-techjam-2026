"""Config-driven construction for comparable backbone experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch import nn

from src.data import ImagePreprocessing

from .detectors import FusionDetector, VisualDetector
from .encoders import ModelError, create_backbone
from .forensic import ForensicBranch
from .specs import canonical_backbone_name, get_backbone_spec


ARCHITECTURES = ("visual", "fusion")


def create_preprocess(
    backbone: str = "dinov2",
    *,
    image_size: Optional[int] = None,
    resize_mode: Optional[str] = None,
) -> ImagePreprocessing:
    """Build preprocessing that exactly matches a registered backbone family."""
    spec = get_backbone_spec(backbone)
    return ImagePreprocessing(
        image_size=spec.image_size if image_size is None else image_size,
        resize_mode=resize_mode or spec.resize_mode,
        normalization=spec.normalization,
    )


def create_model(
    backbone: str = "dinov2",
    architecture: str = "visual",
    *,
    model_id: Optional[str] = None,
    pooling: Optional[str] = None,
    freeze_backbone: bool = True,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    head_hidden_dim: Optional[int] = 256,
    head_dropout: float = 0.2,
    forensic_dim: int = 128,
    forensic_width: int = 32,
    forensic_dropout: float = 0.1,
    # Injection points keep tests/offline development download-free.
    backbone_model: Optional[nn.Module] = None,
    backbone_feature_dim: Optional[int] = None,
    call_with_pixel_values: bool = True,
) -> nn.Module:
    """Create a baseline or fused detector from JSON-friendly arguments.

    All three candidate backbones use the same head options. ``backbone_model``
    is intended for tests and locally constructed modules; production normally
    leaves it unset and restores a downloaded/cached pretrained backbone.
    """
    name = canonical_backbone_name(backbone)
    architecture = str(architecture).lower()
    if architecture not in ARCHITECTURES:
        raise ModelError("architecture must be one of %s" % (ARCHITECTURES,))
    spec = get_backbone_spec(name)
    encoder = create_backbone(
        name,
        model_id=model_id,
        pooling=pooling,
        frozen=freeze_backbone,
        revision=revision,
        local_files_only=local_files_only,
        model=backbone_model,
        feature_dim=backbone_feature_dim,
        call_with_pixel_values=call_with_pixel_values,
    )
    if architecture == "visual":
        detector: nn.Module = VisualDetector(
            encoder,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
    else:
        forensic = ForensicBranch(
            output_dim=forensic_dim,
            width=forensic_width,
            normalization=spec.normalization,
            dropout=forensic_dropout,
        )
        detector = FusionDetector(
            encoder,
            forensic,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
    detector.backbone_name = name
    detector.backbone_model_id = model_id or spec.model_id
    detector.backbone_revision = revision
    detector.preprocessing_spec = spec
    return detector


def comparison_configs(
    *,
    architecture: str = "visual",
    freeze_backbone: bool = True,
    head_hidden_dim: Optional[int] = 256,
    head_dropout: float = 0.2,
) -> List[Dict[str, Any]]:
    """Three directly comparable configs; data/training settings stay external."""
    if architecture not in ARCHITECTURES:
        raise ModelError("architecture must be one of %s" % (ARCHITECTURES,))
    return [
        {
            "backbone": name,
            "architecture": architecture,
            "freeze_backbone": bool(freeze_backbone),
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": float(head_dropout),
        }
        for name in ("clip", "dinov2", "ijepa")
    ]
