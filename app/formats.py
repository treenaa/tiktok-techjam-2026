"""Upload formats the demos accept.

The default photo format on iPhone and macOS Photos is HEIC. Pillow cannot
decode it without an extra opener, and Streamlit's `file_uploader` rejects a
disallowed extension in the browser before any of our code runs -- so an
unlisted format looks to the user like the app refusing their picture with no
explanation.

`pillow-heif` is treated as optional: if it is missing the demos still run, they
just decline HEIC with a message that says how to fix it, rather than failing
somewhere less obvious.
"""

from __future__ import annotations

from typing import Tuple


def enable_extra_formats() -> bool:
    """Register the HEIF/HEIC opener with Pillow. True if HEIC is decodable.

    Safe to call repeatedly; registering twice is a no-op in Pillow.
    """
    try:
        import pillow_heif
    except ImportError:
        return False
    try:
        pillow_heif.register_heif_opener()
    except Exception:
        return False
    return True


HEIC_AVAILABLE = enable_extra_formats()

#: Formats Pillow decodes natively once the module above has run. GIF and AVIF
#: need no extra dependency on Pillow 12.
_BASE: Tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "avif")

#: Extensions offered by the uploader.
UPLOAD_TYPES: Tuple[str, ...] = _BASE + (("heic", "heif") if HEIC_AVAILABLE else ())


def unsupported_note() -> str:
    """One line for the UI describing what is accepted, and what is not."""
    if HEIC_AVAILABLE:
        return "JPG, PNG, WEBP, HEIC, AVIF, GIF, BMP, TIFF"
    return (
        "JPG, PNG, WEBP, AVIF, GIF, BMP, TIFF — HEIC needs `pip install pillow-heif`"
    )
