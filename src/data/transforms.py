"""Competition transformation suite.

Every transform is a callable ``PIL.Image -> PIL.Image`` with a stable ``name``,
so it composes with ``torchvision.transforms.Compose`` but does not require
torchvision to be installed.

Two entry points:

``A. deterministic named transforms`` -- :data:`TRANSFORM_REGISTRY` /
:func:`get_transform`, for evaluation.  Calling the same named transform twice
on the same image yields byte-identical output (noise is seeded).

``B. stochastic sampling`` -- :class:`RandomCompetitionTransform`, for training.
It samples a family and continuous parameters inside the competition ranges and
returns a concrete deterministic transform, so the applied operation is always
inspectable/loggable via ``.name``.
"""

from __future__ import annotations

import io
import random
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

__all__ = [
    "Transform",
    "Identity",
    "Compose",
    "JPEGCompression",
    "GaussianBlur",
    "ResizeRoundTrip",
    "GaussianNoise",
    "ColorJitter",
    "CenterCropResize",
    "TRANSFORM_REGISTRY",
    "TRANSFORM_FAMILIES",
    "TRANSFORM_ALIASES",
    "EVAL_TRANSFORM_NAMES",
    "OFFICIAL_TRANSFORM_NAMES",
    "canonical_transform_name",
    "get_eval_transform",
    "list_eval_transforms",
    "describe_eval_transforms",
    "get_transform",
    "list_transforms",
    "build_eval_suite",
    "RandomCompetitionTransform",
]

#: Resampling filters (Pillow >= 9.1 moved these under ``Image.Resampling``).
_RESAMPLE = getattr(Image, "Resampling", Image)
BILINEAR = _RESAMPLE.BILINEAR
BICUBIC = _RESAMPLE.BICUBIC


def _to_rgb(img: "Image.Image") -> "Image.Image":
    return img if img.mode == "RGB" else img.convert("RGB")


class Transform:
    """Base class: a named, deterministic, PIL-in/PIL-out operation."""

    name = "transform"
    family = "identity"

    def __call__(self, img: "Image.Image") -> "Image.Image":
        if not isinstance(img, Image.Image):
            raise TypeError("%s expects a PIL.Image, got %r" % (type(self).__name__, type(img)))
        return self.apply(img)

    def apply(self, img: "Image.Image") -> "Image.Image":  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%r" % kv for kv in self.params.items()),
        )


class Identity(Transform):
    """The clean view."""

    name = "clean"
    family = "identity"

    def apply(self, img):
        return img.copy()


class Compose(Transform):
    """Sequential application; ``name`` is the joined child names."""

    family = "compose"

    def __init__(self, transforms: Sequence[Callable[["Image.Image"], "Image.Image"]]):
        self.transforms = list(transforms)
        self.name = "+".join(getattr(t, "name", type(t).__name__) for t in self.transforms) or "clean"

    def apply(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    @property
    def params(self):
        return {"transforms": self.transforms}


# --------------------------------------------------------------------------
# JPEG compression -- quality 90 / 70 / 50 / 30
# --------------------------------------------------------------------------
class JPEGCompression(Transform):
    """Re-encode through JPEG at ``quality`` (1-100, lower = more artifacts)."""

    family = "jpeg"

    def __init__(self, quality: int, subsampling: int = -1):
        if not 1 <= int(quality) <= 100:
            raise ValueError("jpeg quality must be in [1, 100], got %r" % (quality,))
        self.quality = int(quality)
        self.subsampling = subsampling
        self.name = "jpeg_%d" % self.quality

    def apply(self, img):
        mode = img.mode
        enc = img if mode in ("RGB", "L") else _to_rgb(img)
        buf = io.BytesIO()
        enc.save(buf, format="JPEG", quality=self.quality, subsampling=self.subsampling)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out if out.mode == mode else out.convert(mode if mode != "P" else "RGB")

    @property
    def params(self):
        return {"quality": self.quality}


# --------------------------------------------------------------------------
# Gaussian blur -- sigma 0.5 / 1.0 / 2.0
# --------------------------------------------------------------------------
class GaussianBlur(Transform):
    """Gaussian blur; ``sigma`` is the standard deviation in pixels.

    Pillow's ``ImageFilter.GaussianBlur(radius=...)`` takes the standard
    deviation, so sigma maps onto it directly.
    """

    family = "blur"

    def __init__(self, sigma: float, decimals: Optional[int] = None):
        if float(sigma) < 0:
            raise ValueError("blur sigma must be >= 0, got %r" % (sigma,))
        self.sigma = float(sigma)
        self.name = "blur_%s" % _fmt(self.sigma, decimals)

    def apply(self, img):
        if self.sigma == 0:
            return img.copy()
        # Pillow cannot filter palette/bilevel images; RGB is the only sane
        # target since blurring produces colours outside the palette anyway.
        if img.mode in ("P", "1"):
            img = _to_rgb(img)
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

    @property
    def params(self):
        return {"sigma": self.sigma}


# --------------------------------------------------------------------------
# Resize round-trip -- 0.5x and 0.25x down then back up
# --------------------------------------------------------------------------
class ResizeRoundTrip(Transform):
    """Downscale by ``scale`` then upscale back to the original resolution."""

    family = "resize"

    def __init__(
        self,
        scale: float,
        down_filter=BILINEAR,
        up_filter=BICUBIC,
        decimals: Optional[int] = None,
    ):
        if not 0 < float(scale) <= 1:
            raise ValueError("resize scale must be in (0, 1], got %r" % (scale,))
        self.scale = float(scale)
        self.down_filter = down_filter
        self.up_filter = up_filter
        self.name = "resize_%s" % _fmt(self.scale, decimals)

    def apply(self, img):
        w, h = img.size
        # Never collapse a dimension to zero on tiny images (CIFAKE is 32x32).
        dw, dh = max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))
        small = img.resize((dw, dh), self.down_filter)
        return small.resize((w, h), self.up_filter)

    @property
    def params(self):
        return {"scale": self.scale}


# --------------------------------------------------------------------------
# Gaussian noise -- sigma 0.02 / 0.05 / 0.10 (in normalised [0, 1] units)
# --------------------------------------------------------------------------
class GaussianNoise(Transform):
    """Additive i.i.d. Gaussian noise.

    ``sigma`` is expressed on the normalised ``[0, 1]`` intensity scale and
    scaled by 255 internally.  With ``seed`` set the transform is deterministic
    (required for the evaluation suite); with ``seed=None`` each call draws
    fresh noise (training).
    """

    family = "noise"

    def __init__(
        self,
        sigma: float,
        seed: Optional[int] = 0,
        clip: bool = True,
        decimals: Optional[int] = None,
    ):
        if float(sigma) < 0:
            raise ValueError("noise sigma must be >= 0, got %r" % (sigma,))
        self.sigma = float(sigma)
        self.seed = seed
        self.clip = clip
        self.name = "noise_%s" % _fmt(self.sigma, decimals)
        self._rng = np.random.default_rng(seed) if seed is None else None

    def _generator(self):
        if self.seed is None:
            if self._rng is None:
                self._rng = np.random.default_rng()
            return self._rng
        return np.random.default_rng(self.seed)

    def apply(self, img):
        if self.sigma == 0:
            return img.copy()
        mode = img.mode
        arr = np.asarray(_to_rgb(img), dtype=np.float32)
        noise = self._generator().normal(0.0, self.sigma * 255.0, size=arr.shape)
        arr = arr + noise.astype(np.float32)
        if self.clip:
            arr = np.clip(arr, 0.0, 255.0)
        out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return out if mode == "RGB" else out.convert(mode)

    @property
    def params(self):
        return {"sigma": self.sigma, "seed": self.seed}


# --------------------------------------------------------------------------
# Colour jitter -- brightness / contrast / saturation +-20%
# --------------------------------------------------------------------------
_ENHANCERS = {
    "brightness": ImageEnhance.Brightness,
    "contrast": ImageEnhance.Contrast,
    "saturation": ImageEnhance.Color,
}


class ColorJitter(Transform):
    """Deterministic colour jitter with explicit multiplicative factors.

    ``1.0`` is a no-op, ``1.2`` is +20%, ``0.8`` is -20%.  Unlike
    ``torchvision.transforms.ColorJitter`` this takes concrete factors rather
    than sampling ranges -- sampling lives in
    :class:`RandomCompetitionTransform`.
    """

    family = "jitter"

    def __init__(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        order: Sequence[str] = ("brightness", "contrast", "saturation"),
    ):
        self.factors = {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "saturation": float(saturation),
        }
        for k, v in self.factors.items():
            if v < 0:
                raise ValueError("%s factor must be >= 0, got %r" % (k, v))
        self.order = tuple(order)
        active = [(k, self.factors[k]) for k in self.order if self.factors[k] != 1.0]
        self.name = (
            "jitter_" + "_".join("%s%s" % (k[:4], _fmt(v)) for k, v in active)
            if active
            else "jitter_none"
        )

    def apply(self, img):
        out = _to_rgb(img)
        for key in self.order:
            factor = self.factors[key]
            if factor != 1.0:
                out = _ENHANCERS[key](out).enhance(factor)
        return out

    @property
    def params(self):
        return dict(self.factors)


# --------------------------------------------------------------------------
# Center crop -- crop to 80% then resize back
# --------------------------------------------------------------------------
class CenterCropResize(Transform):
    """Center-crop to ``ratio`` of each side, then resize back to the input size.

    ``ratio`` is applied per *linear dimension* (0.8 -> 64% of the area), which
    is the usual reading of "crop to 80%".  Set ``resize_back=False`` to keep the
    cropped resolution.
    """

    family = "crop"

    def __init__(
        self,
        ratio: float = 0.8,
        resize_back: bool = True,
        up_filter=BICUBIC,
        decimals: Optional[int] = None,
    ):
        if not 0 < float(ratio) <= 1:
            raise ValueError("crop ratio must be in (0, 1], got %r" % (ratio,))
        self.ratio = float(ratio)
        self.resize_back = bool(resize_back)
        self.up_filter = up_filter
        self.name = "crop_%s" % _fmt(self.ratio, decimals)

    def apply(self, img):
        w, h = img.size
        cw, ch = max(1, int(round(w * self.ratio))), max(1, int(round(h * self.ratio)))
        left, top = (w - cw) // 2, (h - ch) // 2
        out = img.crop((left, top, left + cw, top + ch))
        if self.resize_back and out.size != (w, h):
            out = out.resize((w, h), self.up_filter)
        return out

    @property
    def params(self):
        return {"ratio": self.ratio, "resize_back": self.resize_back}


def _fmt(value: float, decimals: Optional[int] = None) -> str:
    """Format a parameter for use inside a transform ``name``.

    ``decimals`` pins the precision so registry names match the competition
    spec verbatim (``noise_sigma0.10``, not ``noise_sigma0.1``).  Without it,
    floats keep at least one decimal and drop trailing zeros beyond that --
    which keeps sampled (continuous) parameters readable.
    """
    value = float(value)
    if decimals is not None:
        return "%.*f" % (decimals, value)
    text = ("%.4f" % value).rstrip("0")
    return text + "0" if text.endswith(".") else text


# ==========================================================================
# A. Deterministic named transforms (evaluation)
# ==========================================================================
JPEG_QUALITIES: Tuple[int, ...] = (90, 70, 50, 30)
BLUR_SIGMAS: Tuple[float, ...] = (0.5, 1.0, 2.0)
RESIZE_SCALES: Tuple[float, ...] = (0.5, 0.25)
NOISE_SIGMAS: Tuple[float, ...] = (0.02, 0.05, 0.10)
JITTER_DELTA: float = 0.20
CROP_RATIO: float = 0.80

#: ``name -> zero-arg factory``.  Factories (not instances) so each caller gets
#: an independent object -- relevant for the noise RNG.
TRANSFORM_REGISTRY: Dict[str, Callable[[], Transform]] = {"clean": Identity}


def _register(factory: Callable[[], Transform]) -> None:
    TRANSFORM_REGISTRY[factory().name] = factory


for _q in JPEG_QUALITIES:
    _register(lambda q=_q: JPEGCompression(q))
for _s in BLUR_SIGMAS:
    _register(lambda s=_s: GaussianBlur(s, decimals=1))
for _s in RESIZE_SCALES:
    _register(lambda s=_s: ResizeRoundTrip(s, decimals=2 if s < 0.5 else 1))
for _s in NOISE_SIGMAS:
    _register(lambda s=_s: GaussianNoise(s, seed=0, decimals=2))
for _ch in ("brightness", "contrast", "saturation"):
    for _sign, _tag in ((1.0 + JITTER_DELTA, "up"), (1.0 - JITTER_DELTA, "down")):
        def _factory(ch=_ch, f=_sign, tag=_tag):
            t = ColorJitter(**{ch: f})
            t.name = "jitter_%s_%s" % (ch, tag)
            return t

        _register(_factory)
_register(lambda: CenterCropResize(CROP_RATIO, decimals=2))

#: ``family -> [names]``
TRANSFORM_FAMILIES: Dict[str, List[str]] = {}
for _name, _fac in TRANSFORM_REGISTRY.items():
    TRANSFORM_FAMILIES.setdefault(_fac().family, []).append(_name)

#: Full evaluation suite, ``clean`` first.
EVAL_TRANSFORM_NAMES: Tuple[str, ...] = ("clean",) + tuple(
    n for n in TRANSFORM_REGISTRY if n != "clean"
)


#: Deprecated spellings kept resolvable so older manifests / result files do not
#: break.  Canonical names are the keys of :data:`TRANSFORM_REGISTRY`.
TRANSFORM_ALIASES: Dict[str, str] = {}
for _q in JPEG_QUALITIES:
    TRANSFORM_ALIASES["jpeg_q%d" % _q] = "jpeg_%d" % _q
    TRANSFORM_ALIASES["jpeg%d" % _q] = "jpeg_%d" % _q
for _s, _canon in zip(BLUR_SIGMAS, ("blur_0.5", "blur_1.0", "blur_2.0")):
    TRANSFORM_ALIASES["blur_sigma%s" % _fmt(_s, 1)] = _canon
    TRANSFORM_ALIASES["blur_%s" % _fmt(_s)] = _canon
for _s, _canon in zip(RESIZE_SCALES, ("resize_0.5", "resize_0.25")):
    TRANSFORM_ALIASES["resize_%sx" % _fmt(_s)] = _canon
    TRANSFORM_ALIASES["resize_%s" % _fmt(_s)] = _canon
for _s, _canon in zip(NOISE_SIGMAS, ("noise_0.02", "noise_0.05", "noise_0.10")):
    TRANSFORM_ALIASES["noise_sigma%s" % _fmt(_s, 2)] = _canon
    TRANSFORM_ALIASES["noise_%s" % _fmt(_s)] = _canon
TRANSFORM_ALIASES.update({
    "crop_0.8": "crop_0.80",
    "identity": "clean",
    "none": "clean",
    "original": "clean",
})
TRANSFORM_ALIASES = {k: v for k, v in TRANSFORM_ALIASES.items() if k not in TRANSFORM_REGISTRY}


def canonical_transform_name(name: str) -> str:
    """Resolve an alias to its canonical registry name.

    Accepts the canonical name unchanged, a known deprecated spelling, or a
    case/whitespace variant.  Raises ``KeyError`` otherwise -- never guesses.
    """
    key = str(name).strip()
    if key in TRANSFORM_REGISTRY:
        return key
    if key in TRANSFORM_ALIASES:
        return TRANSFORM_ALIASES[key]
    lowered = key.lower()
    if lowered in TRANSFORM_REGISTRY:
        return lowered
    if lowered in TRANSFORM_ALIASES:
        return TRANSFORM_ALIASES[lowered]
    raise KeyError(
        "unknown transform %r; official names: %s"
        % (name, list(EVAL_TRANSFORM_NAMES))
    )


#: Explicit alias emphasising that these are the *official* competition
#: transforms, ``clean`` included as the uncorrupted reference.
OFFICIAL_TRANSFORM_NAMES = EVAL_TRANSFORM_NAMES


def get_eval_transform(name: str) -> Transform:
    """Instantiate an official benchmark transform by name.

    The single entry point for evaluation code::

        get_eval_transform("jpeg_30")
        get_eval_transform("blur_2.0")
        get_eval_transform("resize_0.25")
        get_eval_transform("noise_0.10")
        get_eval_transform("crop_0.80")

    Returned transforms are deterministic: the same name applied to the same
    image always yields identical pixels.  A fresh instance is returned per
    call, so callers cannot share mutable RNG state by accident.
    """
    return TRANSFORM_REGISTRY[canonical_transform_name(name)]()


#: Backwards-compatible alias of :func:`get_eval_transform`.
get_transform = get_eval_transform


def list_eval_transforms(
    family: Optional[str] = None, include_clean: bool = True
) -> List[str]:
    """Enumerate the official benchmark transform names.

    ``family`` restricts to one of ``jpeg``/``blur``/``resize``/``noise``/
    ``jitter``/``crop``/``identity``.  Order is stable across processes:
    ``clean`` first, then families in competition order.
    """
    if family is not None:
        if family not in TRANSFORM_FAMILIES:
            raise KeyError(
                "unknown family %r; available: %s" % (family, sorted(TRANSFORM_FAMILIES))
            )
        return list(TRANSFORM_FAMILIES[family])
    names = list(EVAL_TRANSFORM_NAMES)
    return names if include_clean else [n for n in names if n != "clean"]


#: Backwards-compatible alias of :func:`list_eval_transforms`.
list_transforms = list_eval_transforms


def describe_eval_transforms() -> "List[Dict[str, Any]]":
    """Machine-readable spec of the benchmark: name, family and parameters.

    Intended for evaluation code that reports a robustness grid and for
    serialising exactly which corruption produced a number.
    """
    out = []
    for name in EVAL_TRANSFORM_NAMES:
        transform = TRANSFORM_REGISTRY[name]()
        out.append(
            {
                "name": name,
                "family": transform.family,
                "params": transform.params,
                "severity": _SEVERITY.get(name),
            }
        )
    return out


#: Ordinal severity within a family (0 = mildest).  ``None`` where the family
#: has no natural ordering (jitter directions are symmetric).
_SEVERITY: Dict[str, Optional[int]] = {"clean": 0}
for _i, _q in enumerate(JPEG_QUALITIES):
    _SEVERITY["jpeg_%d" % _q] = _i  # 90 -> 0 (mildest) ... 30 -> 3
for _i, _s in enumerate(BLUR_SIGMAS):
    _SEVERITY["blur_%s" % _fmt(_s, 1)] = _i
for _i, _n in enumerate(("resize_0.5", "resize_0.25")):
    _SEVERITY[_n] = _i
for _i, _s in enumerate(NOISE_SIGMAS):
    _SEVERITY["noise_%s" % _fmt(_s, 2)] = _i
for _n in TRANSFORM_FAMILIES["jitter"]:
    _SEVERITY[_n] = None
_SEVERITY["crop_0.80"] = 0


def build_eval_suite(names: Optional[Iterable[str]] = None) -> Dict[str, Transform]:
    """``{name: transform}`` for robustness evaluation (defaults to all).

    Keys are canonical names even when aliases are passed in.
    """
    names = list(EVAL_TRANSFORM_NAMES) if names is None else [
        canonical_transform_name(n) for n in names
    ]
    return OrderedDict((n, get_eval_transform(n)) for n in names)


# ==========================================================================
# B. Stochastic transform sampling (training)
# ==========================================================================
#: Continuous sampling ranges, one per family.
DEFAULT_SAMPLING_SPACE: Dict[str, Dict[str, Any]] = {
    "jpeg": {"quality": (30, 95)},
    "blur": {"sigma": (0.3, 2.0)},
    "resize": {"scale": (0.25, 0.75)},
    "noise": {"sigma": (0.01, 0.10)},
    "jitter": {"delta": (0.0, JITTER_DELTA)},
    "crop": {"ratio": (0.75, 0.95)},
}
DEFAULT_FAMILIES: Tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop")


class RandomCompetitionTransform:
    """Sample a competition-style corruption per call.

    Parameters
    ----------
    families:
        Families to draw from.  ``None`` uses all six.
    weights:
        Optional per-family sampling weights (same length as ``families``).
    p_identity:
        Probability of returning the clean image untouched.
    n_ops:
        ``int`` or ``(low, high)`` inclusive range -- how many corruptions to
        chain.  Chaining draws distinct families.
    seed:
        Seeds an internal ``random.Random`` for reproducible training streams.
        ``None`` uses fresh randomness.
    space:
        Override :data:`DEFAULT_SAMPLING_SPACE`.

    Notes
    -----
    ``sample()`` returns a concrete :class:`Transform`, so callers can log
    ``.name`` to know exactly what was applied.  ``__call__`` samples and applies
    in one step.
    """

    name = "random_competition"
    family = "random"

    def __init__(
        self,
        families: Optional[Sequence[str]] = None,
        weights: Optional[Sequence[float]] = None,
        p_identity: float = 0.0,
        n_ops: Any = 1,
        seed: Optional[int] = None,
        space: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.families = tuple(families) if families else DEFAULT_FAMILIES
        unknown = [f for f in self.families if f not in DEFAULT_SAMPLING_SPACE]
        if unknown:
            raise ValueError(
                "unknown transform families %s; available: %s"
                % (unknown, sorted(DEFAULT_SAMPLING_SPACE))
            )
        if weights is not None and len(weights) != len(self.families):
            raise ValueError("weights must match families in length")
        self.weights = list(weights) if weights is not None else None
        if not 0.0 <= float(p_identity) <= 1.0:
            raise ValueError("p_identity must be in [0, 1], got %r" % (p_identity,))
        self.p_identity = float(p_identity)
        self.n_ops = (n_ops, n_ops) if isinstance(n_ops, int) else tuple(n_ops)
        if self.n_ops[0] < 0 or self.n_ops[0] > self.n_ops[1]:
            raise ValueError("invalid n_ops %r" % (n_ops,))
        self.seed = seed
        self.rng = random.Random(seed)
        self.space = dict(DEFAULT_SAMPLING_SPACE)
        if space:
            self.space.update(space)

    # -- sampling ---------------------------------------------------------
    def _sample_family(self, family: str) -> Transform:
        spec = self.space[family]
        rng = self.rng
        if family == "jpeg":
            lo, hi = spec["quality"]
            return JPEGCompression(int(round(rng.uniform(lo, hi))))
        if family == "blur":
            lo, hi = spec["sigma"]
            return GaussianBlur(rng.uniform(lo, hi))
        if family == "resize":
            lo, hi = spec["scale"]
            return ResizeRoundTrip(rng.uniform(lo, hi))
        if family == "noise":
            lo, hi = spec["sigma"]
            # Seed from our RNG: fresh noise per call, reproducible per stream.
            return GaussianNoise(rng.uniform(lo, hi), seed=rng.randrange(2 ** 31))
        if family == "jitter":
            lo, hi = spec["delta"]
            factors = {}
            for channel in ("brightness", "contrast", "saturation"):
                delta = rng.uniform(lo, hi) * rng.choice((-1.0, 1.0))
                factors[channel] = 1.0 + delta
            return ColorJitter(**factors)
        if family == "crop":
            lo, hi = spec["ratio"]
            return CenterCropResize(rng.uniform(lo, hi))
        raise ValueError("unhandled family %r" % family)  # pragma: no cover

    def sample(self) -> Transform:
        """Draw a concrete transform (may be :class:`Identity`)."""
        if self.p_identity and self.rng.random() < self.p_identity:
            return Identity()
        k = self.rng.randint(*self.n_ops)
        if k <= 0:
            return Identity()
        k = min(k, len(self.families))
        if self.weights is None:
            chosen = self.rng.sample(list(self.families), k)
        else:
            chosen, pool, weights = [], list(self.families), list(self.weights)
            for _ in range(k):
                pick = self.rng.choices(range(len(pool)), weights=weights, k=1)[0]
                chosen.append(pool.pop(pick))
                weights.pop(pick)
        ops = [self._sample_family(f) for f in chosen]
        return ops[0] if len(ops) == 1 else Compose(ops)

    def __call__(self, img: "Image.Image") -> "Image.Image":
        return self.sample()(img)

    def set_seed(self, seed: Optional[int]) -> None:
        """Re-seed the stream (e.g. per epoch / per DataLoader worker)."""
        self.seed = seed
        self.rng = random.Random(seed)

    def __repr__(self) -> str:
        return "RandomCompetitionTransform(families=%s, n_ops=%s, p_identity=%s, seed=%s)" % (
            list(self.families),
            self.n_ops,
            self.p_identity,
            self.seed,
        )
