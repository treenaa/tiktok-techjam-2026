"""Image loading: the single decode path for the whole repository.

Training, evaluation and inference must read pixels the same way -- if
``predict.py`` grows its own loader, the corrupt-file policy and colour handling
drift apart and the served model stops matching the evaluated one.

Project rule 20.6 says corrupt files must not be skipped silently.  The policy
is therefore explicit and chosen by the caller:

``"raise"``
    Default.  A bad file aborts with a clear, path-carrying error.  Correct for
    training and evaluation, where a silently dropped image changes the metric
    denominator.
``"skip"``
    Return ``None``.  The caller must handle it and is expected to report the
    count.
``"placeholder"``
    Return a mid-grey image so a batch keeps its shape.  For inference only,
    where a run over a user directory should not die on one bad file -- the
    result must still be reported as unreadable, never as a real score.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile

__all__ = [
    "ImageLoadError",
    "SUPPORTED_EXTENSIONS",
    "ON_ERROR_POLICIES",
    "load_image",
    "default_loader",
    "make_loader",
    "list_images",
    "verify_images",
]

#: Formats the competition CLI must accept (rule 17), lower-case with dot.
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
)

ON_ERROR_POLICIES: Tuple[str, ...] = ("raise", "skip", "placeholder")

#: Mid-grey; only used by the ``placeholder`` policy.
PLACEHOLDER_COLOR = (128, 128, 128)


class ImageLoadError(OSError):
    """Raised when an image cannot be read.  Always names the path."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__("cannot read image %r: %s" % (path, reason))


def load_image(
    path: str,
    on_error: str = "raise",
    placeholder_size: Tuple[int, int] = (224, 224),
    allow_truncated: bool = False,
) -> Optional["Image.Image"]:
    """Load one image as fully-decoded RGB.

    Always returns mode ``RGB`` so downstream code never has to branch on
    palette/greyscale/alpha inputs.  The file is decoded eagerly (``img.load()``)
    rather than lazily, which matters under DataLoader workers: a lazy handle can
    outlive the worker and fail far from the cause.

    Parameters
    ----------
    on_error:
        One of :data:`ON_ERROR_POLICIES` -- see the module docstring.
    allow_truncated:
        Permit PIL to decode truncated JPEGs instead of erroring.  Off by
        default: a truncated file is usually a real data problem worth seeing,
        and half an image produces a meaningless prediction.

    Returns
    -------
    A PIL image, or ``None`` when ``on_error="skip"`` and the file is unreadable.
    """
    if on_error not in ON_ERROR_POLICIES:
        raise ValueError(
            "on_error must be one of %s, got %r" % (list(ON_ERROR_POLICIES), on_error)
        )

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        if allow_truncated:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with open(path, "rb") as handle:
                img = Image.open(handle)
                img.load()
            return img if img.mode == "RGB" else img.convert("RGB")
        except Exception as exc:
            # Deliberately broad: PIL raises OSError, ValueError, SyntaxError,
            # struct.error and others for malformed files. Never swallowed --
            # re-raised with the path, or handled by an explicit policy.
            reason = "%s: %s" % (type(exc).__name__, exc)
            if on_error == "raise":
                raise ImageLoadError(path, reason) from exc
            if on_error == "skip":
                return None
            return Image.new("RGB", placeholder_size, PLACEHOLDER_COLOR)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def default_loader(path: str) -> "Image.Image":
    """Strict loader used by the datasets: RGB or a loud :class:`ImageLoadError`."""
    return load_image(path, on_error="raise")


def make_loader(
    on_error: str = "raise",
    placeholder_size: Tuple[int, int] = (224, 224),
    allow_truncated: bool = False,
) -> Callable[[str], Optional["Image.Image"]]:
    """Build a ``path -> image`` loader with a fixed policy.

    Inference wants ``make_loader("placeholder")`` so one unreadable file cannot
    abort a run over a user-supplied directory; training should keep the strict
    default.
    """

    def _loader(path: str) -> Optional["Image.Image"]:
        return load_image(
            path,
            on_error=on_error,
            placeholder_size=placeholder_size,
            allow_truncated=allow_truncated,
        )

    return _loader


def list_images(
    directory: str,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
    follow_links: bool = False,
) -> List[str]:
    """Every image under ``directory``, in deterministic sorted order.

    Rule 17 requires deterministic file ordering in the prediction JSON, so this
    sorts rather than relying on filesystem order.  Hidden files (dotfiles) and
    macOS ``__MACOSX`` resource forks are ignored.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError("not a directory: %s" % directory)
    wanted = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}

    found: List[str] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(directory, followlinks=follow_links):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
            for name in filenames:
                if name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in wanted:
                    found.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(directory):
            if name.startswith("."):
                continue
            full = os.path.join(directory, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in wanted:
                found.append(full)
    return sorted(found)


def verify_images(
    paths: Sequence[str], allow_truncated: bool = False
) -> Dict[str, Any]:
    """Check which paths are readable, without keeping the pixels.

    A cheap pre-flight for a manifest or an inference directory::

        report = verify_images(list_images("./images"))
        if report["unreadable"]:
            ...

    Returns ``{"n_checked", "n_readable", "readable", "unreadable"}`` where
    ``unreadable`` is a list of ``(path, reason)``.
    """
    readable: List[str] = []
    unreadable: List[Tuple[str, str]] = []
    for path in paths:
        try:
            load_image(path, on_error="raise", allow_truncated=allow_truncated)
            readable.append(path)
        except ImageLoadError as exc:
            unreadable.append((path, exc.reason))
        except OSError as exc:  # missing file, permissions
            unreadable.append((path, "%s: %s" % (type(exc).__name__, exc)))
    return {
        "n_checked": len(paths),
        "n_readable": len(readable),
        "readable": readable,
        "unreadable": unreadable,
    }
