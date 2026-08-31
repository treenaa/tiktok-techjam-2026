"""Coverage for the Spectral Evidence demo (`app/studio.py`).

The spectral rendering is tested directly; the page itself is driven through
`streamlit.testing.v1.AppTest` with an injected tiny model, so nothing here
downloads weights or needs a real checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from app.spectral import ambient, pair_strip, spectral_distance, spectrum_image
from app.theme import RAMP, css, ramp_css
from src.data import get_eval_transform

#: AppTest resolves relative paths against the calling file, not the repo root.
STUDIO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "studio.py")


def _textured(size: int = 128, seed: int = 5) -> Image.Image:
    """Structured image: flat noise would make every spectrum look alike."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    field = 120 + 60 * np.sin(xx / 6.0) + 45 * np.cos(yy / 9.0) + rng.normal(0, 10, (size, size))
    return Image.fromarray(
        np.stack([np.clip(field, 0, 255)] * 3, -1).astype("uint8"), mode="RGB"
    )


# -- spectral rendering ----------------------------------------------------
def test_spectrum_is_square_rgb():
    out = spectrum_image(_textured(), size=64)
    assert out.size == (64, 64) and out.mode == "RGB"


def test_spectrum_uses_the_ramp_not_greyscale():
    """A greyscale map would mean the colour ramp was not applied."""
    pixels = np.asarray(spectrum_image(_textured(), size=64)).reshape(-1, 3)
    spread = np.abs(pixels[:, 0].astype(int) - pixels[:, 2].astype(int))
    assert spread.max() > 30


def test_degradation_changes_the_spectrum():
    """The whole premise: corruption is visible in the frequency signature."""
    base = _textured()
    for name in ("jpeg_30", "blur_2.0", "noise_0.10", "resize_0.25"):
        assert spectral_distance(base, get_eval_transform(name)(base)) > 0.05


def test_identical_images_have_zero_spectral_distance():
    base = _textured()
    assert spectral_distance(base, base.copy()) == pytest.approx(0.0, abs=1e-6)


def test_pair_strip_is_two_squares_side_by_side():
    strip = pair_strip(_textured(), size=64, gap=2)
    assert strip.size == (130, 64)


def test_pair_strip_accepts_non_square_input():
    tall = _textured().resize((60, 140))
    assert pair_strip(tall, size=48, gap=2).size == (98, 48)


def test_ambient_renders_without_an_image():
    assert ambient(size=48).size == (48, 48)


# -- upload formats --------------------------------------------------------
# A judge photographing something on a phone hands us HEIC. Streamlit rejects an
# unlisted extension in the browser before any of our code runs, so an omission
# here looks like the app refusing the picture for no reason.
def test_uploader_accepts_the_common_photo_formats():
    from app.formats import UPLOAD_TYPES

    for extension in ("jpg", "jpeg", "png", "webp", "gif", "avif"):
        assert extension in UPLOAD_TYPES


def test_heic_is_offered_when_the_optional_opener_is_installed():
    from app.formats import HEIC_AVAILABLE, UPLOAD_TYPES

    if HEIC_AVAILABLE:
        assert "heic" in UPLOAD_TYPES and "heif" in UPLOAD_TYPES
    else:
        assert "heic" not in UPLOAD_TYPES


@pytest.mark.parametrize("fmt,extension", [("GIF", ".gif"), ("AVIF", ".avif")])
def test_natively_supported_formats_decode_through_the_shared_loader(tmp_path, fmt, extension):
    from src.data import load_image

    path = tmp_path / ("photo" + extension)
    _textured(96).save(path, format=fmt)
    assert load_image(str(path), on_error="raise").mode == "RGB"


def test_heic_decodes_when_available(tmp_path):
    from app.formats import HEIC_AVAILABLE

    if not HEIC_AVAILABLE:
        pytest.skip("pillow-heif not installed")
    from src.data import load_image

    path = tmp_path / "photo.heic"
    _textured(96).save(path, format="HEIF")
    decoded = load_image(str(path), on_error="raise")
    assert decoded.mode == "RGB" and decoded.size == (96, 96)


# -- theme -----------------------------------------------------------------
def test_css_defines_every_token_the_markup_uses():
    block = css()
    for token in ("--void", "--panel", "--rule", "--ink", "--stable", "--flipped", "--ramp"):
        assert token in block
    for klass in ("sx-card", "sx-verdict", "sx-prob", "sx-img", "sx-flag", "sx-tbl"):
        assert klass in block


def test_ramp_css_is_a_gradient_over_the_ramp():
    gradient = ramp_css()
    assert gradient.startswith("linear-gradient(")
    assert gradient.count("rgb(") == len(RAMP)


def test_css_percentages_survived_interpolation():
    """theme.css() is built with %-formatting; literal % must be escaped."""
    block = css()
    assert "width:100%;" in block and "height:100%;" in block
    assert "%%" not in block


# -- the page --------------------------------------------------------------
def test_page_renders_onboarding_without_a_checkpoint():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=60)
    app.run()
    assert not app.exception
    assert any("No detector loaded" in str(m.value) for m in app.markdown)


def test_page_surfaces_a_bad_checkpoint_as_an_error_not_a_traceback(tmp_path):
    from streamlit.testing.v1 import AppTest

    broken = tmp_path / "not-a-checkpoint.pt"
    broken.write_bytes(b"definitely not a torch checkpoint")

    app = AppTest.from_file(STUDIO, default_timeout=60)
    app.run()
    app.text_input[0].set_value(str(broken)).run()
    assert not app.exception
    assert any("Could not load that checkpoint" in str(e.value) for e in app.error)


# -- the full ladder, with a tiny injected model ---------------------------
# The real checkpoint is 330 MB and not in the repository, so these drive the
# page against a self-describing checkpoint built here. Same code path, no
# download, seconds to run.
torch = pytest.importorskip("torch")


class _TinyDetector(torch.nn.Module):
    """Mean-intensity detector: enough to exercise the whole page."""

    def __init__(self, scale: float = 6.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, images):
        return (images.mean(dim=(1, 2, 3)) - 0.45) * self.scale


def tiny_model(scale: float = 6.0):
    return _TinyDetector(scale)


def tiny_preprocess(image_size: int = 32):
    from src.data import build_preprocess

    return build_preprocess("none", image_size=image_size)


def _tiny_checkpoint(tmp_path) -> str:
    payload = {
        "model_state_dict": tiny_model().state_dict(),
        "best_threshold": 0.5,
        "threshold_source": "validation",
        "run_metadata": {
            "model_factory": "test_studio_app:tiny_model",
            "model_kwargs": {"scale": 6.0},
            "preprocess_factory": "test_studio_app:tiny_preprocess",
            "preprocess_kwargs": {"image_size": 32},
            "backbone": "tiny",
        },
    }
    path = tmp_path / "tiny.pt"
    torch.save(payload, path)
    return str(path)


def _upload(app, image: Image.Image):
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    app.file_uploader[0].set_value(("sample.png", buffer.getvalue(), "image/png"))
    return app.run()


def test_upload_renders_the_core_ladder(tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=300)
    app.run()
    app.text_input[0].set_value(_tiny_checkpoint(tmp_path)).run()
    _upload(app, _textured(96))

    assert not app.exception
    assert not app.error

    blocks = [str(m.value) for m in app.markdown]
    cards = [b for b in blocks if 'class="sx-card' in b]
    assert len(cards) == 6, "core ladder should render one card per core view"
    # Every card carries its own embedded image + spectrum pair.
    assert sum(b.count("data:image/jpeg;base64,") for b in blocks) == 6
    assert any("sx-verdict" in b for b in blocks)
    assert any("sx-prov" in b for b in blocks), "provenance strip should render"


def test_full_suite_renders_all_twenty_views(tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=300)
    app.run()
    app.text_input[0].set_value(_tiny_checkpoint(tmp_path)).run()
    _upload(app, _textured(96))
    app.radio[0].set_value("All twenty").run()

    assert not app.exception
    cards = [str(m.value) for m in app.markdown if 'class="sx-card' in str(m.value)]
    assert len(cards) == 20


def test_drift_chart_is_rendered(tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=300)
    app.run()
    app.text_input[0].set_value(_tiny_checkpoint(tmp_path)).run()
    _upload(app, _textured(96))

    types = set()

    def walk(node):
        kind = getattr(node, "type", None)
        if kind:
            types.add(kind)
        for child in getattr(node, "children", {}).values():
            walk(child)

    walk(app._tree)
    assert "vega_lite_chart" in types


def test_ladder_card_names_match_the_official_transforms(tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=300)
    app.run()
    app.text_input[0].set_value(_tiny_checkpoint(tmp_path)).run()
    _upload(app, _textured(96))

    blocks = "".join(str(m.value) for m in app.markdown)
    for name in ("clean", "jpeg_30", "blur_2.0", "resize_0.25", "noise_0.10", "crop_0.80"):
        assert '>%s<' % name in blocks
