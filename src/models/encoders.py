"""Uniform feature adapters for I-JEPA, DINOv2, and CLIP vision encoders."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from .specs import BackboneSpec, get_backbone_spec


class ModelError(ValueError):
    """Raised when a model violates the repository's detector contract."""


class OptionalDependencyError(ImportError):
    """Raised when a real pretrained adapter needs an uninstalled package."""


def _field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def pool_vision_output(output: Any, pooling: str) -> Tensor:
    """Convert common Hugging Face vision outputs into ``(B, D)`` features."""
    pooling = str(pooling).lower()
    if torch.is_tensor(output):
        if output.ndim == 2:
            return output
        hidden = output
    else:
        if pooling == "pooler":
            pooled = _field(output, "pooler_output")
            if pooled is None:
                raise ModelError("pooling='pooler' requested but model returned no pooler_output")
            if pooled.ndim != 2:
                raise ModelError("pooler_output must have shape (B, D), got %s" % (tuple(pooled.shape),))
            return pooled
        hidden = _field(output, "last_hidden_state")
    if hidden is None or not torch.is_tensor(hidden):
        raise ModelError("encoder output has no tensor last_hidden_state")
    if hidden.ndim != 3:
        raise ModelError("last_hidden_state must have shape (B, tokens, D), got %s" % (tuple(hidden.shape),))
    if pooling == "cls":
        if hidden.shape[1] < 1:
            raise ModelError("cannot CLS-pool an empty token sequence")
        return hidden[:, 0]
    if pooling == "mean":
        return hidden.mean(dim=1)
    if pooling == "mean_patches":
        if hidden.shape[1] < 2:
            raise ModelError("mean_patches requires a CLS token plus at least one patch token")
        return hidden[:, 1:].mean(dim=1)
    raise ModelError("pooling must be one of cls/mean/mean_patches/pooler, got %r" % pooling)


def infer_feature_dim(model: nn.Module) -> int:
    """Read a hidden width without running a dummy image through a huge model."""
    config = getattr(model, "config", None)
    for owner in (config, getattr(config, "vision_config", None), model):
        if owner is None:
            continue
        for attribute in ("hidden_size", "embed_dim", "num_features", "feature_dim"):
            value = getattr(owner, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ModelError(
        "could not infer encoder feature dimension; pass feature_dim explicitly"
    )


class VisionEncoder(nn.Module):
    """Wrap a vision module behind a strict embedding and freezing contract."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: Optional[int] = None,
        pooling: str = "cls",
        frozen: bool = True,
        spec: Optional[BackboneSpec] = None,
        call_with_pixel_values: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(
            infer_feature_dim(backbone) if feature_dim is None else feature_dim
        )
        if self.feature_dim < 1:
            raise ModelError("feature_dim must be positive")
        self.pooling = str(pooling)
        self.spec = spec
        self.call_with_pixel_values = bool(call_with_pixel_values)
        self._frozen = False
        self.set_frozen(frozen)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def set_frozen(self, frozen: bool = True) -> "VisionEncoder":
        self._frozen = bool(frozen)
        self.backbone.requires_grad_(not self._frozen)
        if self._frozen:
            self.backbone.eval()
        else:
            self.backbone.train(self.training)
        return self

    def train(self, mode: bool = True) -> "VisionEncoder":
        super().train(mode)
        if self._frozen:
            # Dropout/stochastic depth in a frozen encoder would otherwise make
            # features change while its weights cannot adapt.
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ModelError("encoder input must be a BCHW tensor")
        context = torch.no_grad() if self._frozen else nullcontext()
        with context:
            if self.call_with_pixel_values:
                output = self.backbone(pixel_values=images)
            else:
                output = self.backbone(images)
            features = pool_vision_output(output, self.pooling)
        if features.ndim != 2 or features.shape[0] != images.shape[0]:
            raise ModelError(
                "pooled features must have shape (B, D), got %s" % (tuple(features.shape),)
            )
        if features.shape[1] != self.feature_dim:
            raise ModelError(
                "pooled width %d does not match declared feature_dim %d"
                % (features.shape[1], self.feature_dim)
            )
        return features


def _load_transformers_backbone(
    spec: BackboneSpec,
    model_id: str,
    *,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> nn.Module:
    try:
        import transformers
    except ImportError as exc:
        raise OptionalDependencyError(
            "pretrained %s requires the optional 'transformers' package; "
            "install it before constructing real backbones" % spec.name
        ) from exc
    model_class = getattr(transformers, spec.model_class, None)
    if model_class is None:
        raise OptionalDependencyError(
            "installed transformers does not provide %s; install a version with %s support"
            % (spec.model_class, spec.name)
        )
    kwargs = {"local_files_only": bool(local_files_only)}
    if revision is not None:
        kwargs["revision"] = revision
    return model_class.from_pretrained(model_id, **kwargs)


def create_backbone(
    name: str,
    *,
    model_id: Optional[str] = None,
    pooling: Optional[str] = None,
    frozen: bool = True,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    model: Optional[nn.Module] = None,
    feature_dim: Optional[int] = None,
    call_with_pixel_values: bool = True,
) -> VisionEncoder:
    """Create a real pretrained adapter or wrap an injected test/local model."""
    spec = get_backbone_spec(name)
    backbone = (
        model
        if model is not None
        else _load_transformers_backbone(
            spec,
            model_id or spec.model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    )
    return VisionEncoder(
        backbone,
        feature_dim=feature_dim,
        pooling=pooling or spec.pooling,
        frozen=frozen,
        spec=spec,
        call_with_pixel_values=call_with_pixel_values,
    )


def create_ijepa_encoder(**kwargs: Any) -> VisionEncoder:
    return create_backbone("ijepa", **kwargs)


def create_dinov2_encoder(**kwargs: Any) -> VisionEncoder:
    return create_backbone("dinov2", **kwargs)


def create_clip_encoder(**kwargs: Any) -> VisionEncoder:
    return create_backbone("clip", **kwargs)
