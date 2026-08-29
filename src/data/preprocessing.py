"""Model-aware preprocessing (resize / crop / tensor / normalise).

Deliberately kept *out* of the raw dataset: :class:`ManifestDataset` yields PIL
images unless a ``preprocess`` callable is supplied.  The model owner passes
whichever preprocessing their backbone needs (I-JEPA, CLIP, ImageNet-normalised
CNN, ...) and the data layer stays model-agnostic.

Ordering matters: competition transforms are applied to the *raw* PIL image, and
preprocessing runs afterwards -- corruptions must happen at native resolution,
before any resize to the model's input size.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "NORMALIZATION_PRESETS",
    "ImagePreprocessing",
    "build_preprocess",
    "to_tensor",
    "normalize",
]

_RESAMPLE = getattr(Image, "Resampling", Image)


class _Default:
    """Sentinel: distinguishes "left at the default" from "explicitly asked for"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<default>"


_DEFAULT = _Default()

#: ``name -> (mean, std)`` in [0, 1] units.  I-JEPA/DINOv2/ViT checkpoints use
#: ImageNet statistics; ``"none"`` skips normalisation entirely.
NORMALIZATION_PRESETS: Dict[str, Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "ijepa": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "none": None,
}


def to_tensor(img: "Image.Image"):
    """PIL -> float32 ``CHW`` tensor in ``[0, 1]`` (torch imported lazily)."""
    import torch

    arr = np.asarray(img.convert("RGB") if img.mode != "RGB" else img, dtype=np.uint8)
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
    return tensor.float().div_(255.0)


def normalize(tensor, mean: Sequence[float], std: Sequence[float]):
    """In-place-safe channel normalisation of a ``CHW`` float tensor."""
    import torch

    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    if float(std_t.min()) <= 0:
        raise ValueError("normalisation std must be > 0, got %s" % (list(std),))
    return (tensor - mean_t) / std_t


class ImagePreprocessing:
    """Resize -> optional center crop -> tensor -> optional normalise.

    Parameters
    ----------
    image_size:
        ``int`` (square) or ``(h, w)``.  ``None`` keeps the native resolution --
        only safe with equally sized images or ``batch_size=1``.
    resize_mode:
        ``"squash"`` distorts to the exact size; ``"shortest"`` preserves aspect
        ratio by resizing the shorter side then center-cropping.
    normalization:
        A key of :data:`NORMALIZATION_PRESETS`, an explicit ``(mean, std)``
        pair, or ``None`` for no normalisation.  Left unset it follows
        ``to_tensor``: ImageNet statistics for tensors, none for PIL output.
    to_tensor:
        ``False`` returns a PIL image (lets a model owner plug in their own
        tensor conversion / feature extractor downstream).

    This is a plain callable, so ``torchvision.transforms.Compose([...])``,
    a HuggingFace ``image_processor`` or any ``PIL.Image -> Any`` function can
    be substituted for it wherever a ``preprocess`` argument is accepted.
    """

    def __init__(
        self,
        image_size: Optional[Any] = 224,
        resize_mode: str = "squash",
        normalization: Any = _DEFAULT,
        to_tensor: bool = True,
        interpolation=_RESAMPLE.BICUBIC,
    ):
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            h, w = image_size
            self.image_size = (int(h), int(w))
        if resize_mode not in ("squash", "shortest"):
            raise ValueError("resize_mode must be 'squash' or 'shortest', got %r" % resize_mode)
        self.resize_mode = resize_mode
        self.to_tensor = bool(to_tensor)
        self.interpolation = interpolation

        explicit_normalization = normalization is not _DEFAULT
        if not explicit_normalization:
            # PIL output cannot carry normalisation, so the default adapts.
            normalization = "imagenet" if to_tensor else None
        if isinstance(normalization, str):
            if normalization not in NORMALIZATION_PRESETS:
                raise KeyError(
                    "unknown normalization preset %r; available: %s"
                    % (normalization, sorted(NORMALIZATION_PRESETS))
                )
            self.normalization = NORMALIZATION_PRESETS[normalization]
            self.normalization_name = normalization
        elif normalization is None:
            self.normalization = None
            self.normalization_name = "none"
        else:
            mean, std = normalization
            self.normalization = (tuple(mean), tuple(std))
            self.normalization_name = "custom"
        if self.normalization and not self.to_tensor:
            raise ValueError(
                "normalization=%r requires to_tensor=True (normalisation cannot be "
                "applied to a PIL image)" % (self.normalization_name,)
            )

    def resize(self, img: "Image.Image") -> "Image.Image":
        if self.image_size is None:
            return img
        th, tw = self.image_size
        if self.resize_mode == "squash":
            return img.resize((tw, th), self.interpolation) if img.size != (tw, th) else img
        w, h = img.size
        scale = max(tw / w, th / h)
        rw, rh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
        img = img.resize((rw, rh), self.interpolation)
        left, top = (rw - tw) // 2, (rh - th) // 2
        return img.crop((left, top, left + tw, top + th))

    def __call__(self, img: "Image.Image"):
        if not isinstance(img, Image.Image):
            raise TypeError("ImagePreprocessing expects a PIL.Image, got %r" % type(img))
        img = img.convert("RGB") if img.mode != "RGB" else img
        img = self.resize(img)
        if not self.to_tensor:
            return img
        tensor = to_tensor(img)
        if self.normalization:
            tensor = normalize(tensor, *self.normalization)
        return tensor

    def __repr__(self) -> str:
        return "ImagePreprocessing(image_size=%s, resize_mode=%r, normalization=%r, to_tensor=%s)" % (
            self.image_size,
            self.resize_mode,
            self.normalization_name,
            self.to_tensor,
        )


def build_preprocess(
    model: str = "ijepa",
    image_size: Optional[Any] = 224,
    **kwargs: Any,
) -> ImagePreprocessing:
    """Convenience factory keyed by model family.

    Unknown names fall back to ImageNet statistics -- the safe default for
    ViT-style backbones.  The model owner should override explicitly rather than
    rely on this if their checkpoint differs.
    """
    key = str(model).lower()
    if key in NORMALIZATION_PRESETS:
        normalization = key
    elif "clip" in key:
        normalization = "clip"
    else:
        normalization = "imagenet"
    return ImagePreprocessing(image_size=image_size, normalization=normalization, **kwargs)
