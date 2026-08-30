"""Backbone and preprocessing specifications used across model variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


RGBStats = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

IMAGENET_STATS: RGBStats = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225),
)
CLIP_STATS: RGBStats = (
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


@dataclass(frozen=True)
class BackboneSpec:
    """Everything needed to construct and preprocess one backbone family."""

    name: str
    model_id: str
    model_class: str
    pooling: str
    normalization: RGBStats
    image_size: int = 224
    resize_mode: str = "shortest"


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "ijepa": BackboneSpec(
        name="ijepa",
        model_id="facebook/ijepa_vith14_1k",
        model_class="IJepaModel",
        pooling="mean",
        normalization=IMAGENET_STATS,
    ),
    "dinov2": BackboneSpec(
        name="dinov2",
        model_id="facebook/dinov2-base",
        model_class="Dinov2Model",
        pooling="cls",
        normalization=IMAGENET_STATS,
    ),
    "clip": BackboneSpec(
        name="clip",
        model_id="openai/clip-vit-base-patch32",
        model_class="CLIPVisionModel",
        pooling="pooler",
        normalization=CLIP_STATS,
    ),
}

BACKBONE_ALIASES = {
    "i-jepa": "ijepa",
    "jepa": "ijepa",
    "dino": "dinov2",
    "dino-v2": "dinov2",
    "clip-vit": "clip",
}


def canonical_backbone_name(name: str) -> str:
    key = str(name).strip().lower()
    key = BACKBONE_ALIASES.get(key, key)
    if key not in BACKBONE_SPECS:
        raise KeyError(
            "unknown backbone %r; available: %s" % (name, sorted(BACKBONE_SPECS))
        )
    return key


def get_backbone_spec(name: str) -> BackboneSpec:
    return BACKBONE_SPECS[canonical_backbone_name(name)]
