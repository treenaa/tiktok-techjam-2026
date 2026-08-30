"""Model architectures for robust binary AIGC image detection.

Canonical convention: every detector returns raw logits shaped ``(B,)`` for
label 1 = AIGC. No model applies sigmoid internally.
"""

from .detectors import FusionDetector, VisualDetector
from .encoders import (
    ModelError,
    OptionalDependencyError,
    VisionEncoder,
    create_backbone,
    create_clip_encoder,
    create_dinov2_encoder,
    create_ijepa_encoder,
    infer_feature_dim,
    pool_vision_output,
)
from .factory import ARCHITECTURES, comparison_configs, create_model, create_preprocess
from .forensic import ForensicBranch, InputDenormalizer, LogMagnitudeFFT
from .heads import BinaryClassificationHead
from .specs import (
    BACKBONE_ALIASES,
    BACKBONE_SPECS,
    CLIP_STATS,
    IMAGENET_STATS,
    BackboneSpec,
    canonical_backbone_name,
    get_backbone_spec,
)
from .utils import PARAMETER_LIMIT, count_parameters, parameter_report

__all__ = [
    "ModelError",
    "OptionalDependencyError",
    "BackboneSpec",
    "BACKBONE_SPECS",
    "BACKBONE_ALIASES",
    "IMAGENET_STATS",
    "CLIP_STATS",
    "canonical_backbone_name",
    "get_backbone_spec",
    "VisionEncoder",
    "pool_vision_output",
    "infer_feature_dim",
    "create_backbone",
    "create_ijepa_encoder",
    "create_dinov2_encoder",
    "create_clip_encoder",
    "BinaryClassificationHead",
    "InputDenormalizer",
    "LogMagnitudeFFT",
    "ForensicBranch",
    "VisualDetector",
    "FusionDetector",
    "ARCHITECTURES",
    "create_model",
    "create_preprocess",
    "comparison_configs",
    "PARAMETER_LIMIT",
    "count_parameters",
    "parameter_report",
]
