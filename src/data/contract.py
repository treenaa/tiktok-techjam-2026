"""The data <-> model interface contract.

Defines, in one place, the exact keys a sample/batch carries in each mode and a
validator other owners can call to fail loudly on a mismatch.  ``DATA_CONTRACT.md``
is the prose version of this module; this module is the executable one.

Two modes
---------
``standard`` -- single view, for ordinary training and evaluation::

    {"image", "label", "source_id", "image_path"[, "dataset", "generator",
     "index", "transform_name"]}

``paired`` -- clean + corrupted view of the *same* image, for robustness
training::

    {"clean", "augmented", "label", "source_id", "image_path"[, "dataset",
     "generator", "index", "transform_name"]}

Both modes always carry ``label``, ``source_id`` and ``image_path``, and in
paired mode the two views always share them by construction.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MODE_STANDARD",
    "MODE_PAIRED",
    "MODES",
    "STANDARD_REQUIRED_KEYS",
    "STANDARD_OPTIONAL_KEYS",
    "PAIRED_REQUIRED_KEYS",
    "PAIRED_OPTIONAL_KEYS",
    "IMAGE_KEYS",
    "SchemaError",
    "required_keys",
    "optional_keys",
    "all_keys",
    "validate_sample",
    "validate_batch",
    "describe_contract",
]

MODE_STANDARD = "standard"
MODE_PAIRED = "paired"
MODES: Tuple[str, ...] = (MODE_STANDARD, MODE_PAIRED)

#: Keys every ``standard`` sample must have.
STANDARD_REQUIRED_KEYS: Tuple[str, ...] = ("image", "label", "source_id", "image_path")
#: Documented extras a ``standard`` sample may add.  Nothing else is allowed.
STANDARD_OPTIONAL_KEYS: Tuple[str, ...] = ("dataset", "generator", "index", "transform_name")

#: Keys every ``paired`` sample must have.
PAIRED_REQUIRED_KEYS: Tuple[str, ...] = (
    "clean",
    "augmented",
    "label",
    "source_id",
    "image_path",
)
#: Documented extras a ``paired`` sample may add.
PAIRED_OPTIONAL_KEYS: Tuple[str, ...] = ("transform_name", "dataset", "generator", "index")

#: Which keys hold image data, per mode -- these are what ``preprocess`` turns
#: into tensors.
IMAGE_KEYS: Dict[str, Tuple[str, ...]] = {
    MODE_STANDARD: ("image",),
    MODE_PAIRED: ("clean", "augmented"),
}

_REQUIRED = {MODE_STANDARD: STANDARD_REQUIRED_KEYS, MODE_PAIRED: PAIRED_REQUIRED_KEYS}
_OPTIONAL = {MODE_STANDARD: STANDARD_OPTIONAL_KEYS, MODE_PAIRED: PAIRED_OPTIONAL_KEYS}


class SchemaError(AssertionError):
    """Raised when a sample or batch violates the documented contract."""


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (list(MODES), mode))
    return mode


def required_keys(mode: str = MODE_STANDARD) -> Tuple[str, ...]:
    return _REQUIRED[_check_mode(mode)]


def optional_keys(mode: str = MODE_STANDARD) -> Tuple[str, ...]:
    return _OPTIONAL[_check_mode(mode)]


def all_keys(mode: str = MODE_STANDARD) -> Tuple[str, ...]:
    return required_keys(mode) + optional_keys(mode)


def _is_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch") and hasattr(value, "shape")


def validate_sample(
    sample: Mapping[str, Any],
    mode: str = MODE_STANDARD,
    require_tensor_images: bool = False,
    allow_extra_keys: bool = False,
) -> Mapping[str, Any]:
    """Validate one ``__getitem__`` result.  Returns it, or raises loudly.

    Parameters
    ----------
    require_tensor_images:
        Assert the image keys hold ``torch.Tensor`` -- i.e. that a ``preprocess``
        was supplied.  Off by default because the raw dataset legitimately
        yields PIL images.
    allow_extra_keys:
        Permit keys outside the documented set (off by default so typos and
        drifting key names surface immediately).
    """
    _check_mode(mode)
    if not isinstance(sample, Mapping):
        raise SchemaError(
            "expected a mapping sample in %r mode, got %r" % (mode, type(sample).__name__)
        )

    missing = [k for k in required_keys(mode) if k not in sample]
    if missing:
        raise SchemaError(
            "sample is missing required key(s) %s in %r mode; present: %s"
            % (missing, mode, sorted(sample))
        )
    if not allow_extra_keys:
        unexpected = [k for k in sample if k not in all_keys(mode)]
        if unexpected:
            raise SchemaError(
                "sample has undocumented key(s) %s in %r mode; allowed: %s"
                % (sorted(unexpected), mode, list(all_keys(mode)))
            )

    label = sample["label"]
    if _is_tensor(label):
        values = [int(v) for v in label.reshape(-1).tolist()]
    elif isinstance(label, (list, tuple)):
        values = [int(v) for v in label]
    else:
        values = [int(label)]
    bad = sorted({v for v in values if v not in (0, 1)})
    if bad:
        raise SchemaError("label(s) %s outside the binary domain {0, 1}" % bad)

    for key in ("source_id", "image_path"):
        value = sample[key]
        if isinstance(value, (list, tuple)):
            if not all(isinstance(v, str) and v for v in value):
                raise SchemaError("%r must contain non-empty strings" % key)
        elif not isinstance(value, str) or not value:
            raise SchemaError(
                "%r must be a non-empty string (or list of them), got %r" % (key, value)
            )

    if require_tensor_images:
        for key in IMAGE_KEYS[mode]:
            if not _is_tensor(sample[key]):
                raise SchemaError(
                    "%r must be a torch.Tensor when require_tensor_images=True, got %r; "
                    "supply a `preprocess` callable to the dataset" % (key, type(sample[key]).__name__)
                )
    return sample


def validate_batch(
    batch: Mapping[str, Any],
    mode: str = MODE_STANDARD,
    batch_size: Optional[int] = None,
    require_tensor_images: bool = True,
    allow_extra_keys: bool = False,
) -> Mapping[str, Any]:
    """Validate a collated batch (the DataLoader's output).

    Also checks that every key agrees on the batch dimension, and -- in paired
    mode -- that ``clean`` and ``augmented`` have identical shapes so they can
    be stacked or fed through the backbone together.
    """
    validate_sample(
        batch, mode=mode, require_tensor_images=require_tensor_images,
        allow_extra_keys=allow_extra_keys,
    )

    def _len_of(value: Any) -> Optional[int]:
        if _is_tensor(value):
            return int(value.shape[0]) if value.dim() > 0 else None
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    sizes = {k: _len_of(v) for k, v in batch.items()}
    observed = {n for n in sizes.values() if n is not None}
    if len(observed) > 1:
        raise SchemaError(
            "inconsistent batch dimension across keys: %s"
            % {k: v for k, v in sizes.items() if v is not None}
        )
    if batch_size is not None and observed and batch_size not in observed:
        raise SchemaError(
            "expected batch size %d, observed %s" % (batch_size, sorted(observed))
        )

    if mode == MODE_PAIRED:
        clean, augmented = batch["clean"], batch["augmented"]
        if _is_tensor(clean) and _is_tensor(augmented) and clean.shape != augmented.shape:
            raise SchemaError(
                "paired views must have identical shapes, got clean=%s augmented=%s"
                % (tuple(clean.shape), tuple(augmented.shape))
            )
    return batch


def describe_contract() -> Dict[str, Any]:
    """Machine-readable form of the contract, for docs and cross-team checks."""
    return {
        "modes": list(MODES),
        "label_domain": {0: "real", 1: "aigc"},
        "standard": {
            "required": list(STANDARD_REQUIRED_KEYS),
            "optional": list(STANDARD_OPTIONAL_KEYS),
            "image_keys": list(IMAGE_KEYS[MODE_STANDARD]),
        },
        "paired": {
            "required": list(PAIRED_REQUIRED_KEYS),
            "optional": list(PAIRED_OPTIONAL_KEYS),
            "image_keys": list(IMAGE_KEYS[MODE_PAIRED]),
        },
        "raw_image": {
            "type": "PIL.Image.Image",
            "mode": "RGB",
            "dtype": "uint8",
            "range": [0, 255],
        },
        "preprocessed_image": {
            "type": "torch.Tensor",
            "layout": "CHW",
            "dtype": "float32",
            "range": "[0, 1] before normalization; normalized if a preset is set",
            "owner": "model (passed in as `preprocess`)",
        },
    }
