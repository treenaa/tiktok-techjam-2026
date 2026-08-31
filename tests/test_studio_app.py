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


# -- transform lab ---------------------------------------------------------
def _bundle(image, names=None, fmt="PNG"):
    import app.studio as studio

    views = studio._ordered_views(True)
    if names is not None:
        views = [v for v in views if v["name"] in names]
    return studio._transform_bundle(image, views, "photo", fmt=fmt), views


def test_bundle_contains_one_file_per_transform_plus_a_manifest():
    import io
    import zipfile

    data, views = _bundle(_textured(128))
    archive = zipfile.ZipFile(io.BytesIO(data))
    assert "manifest.json" in archive.namelist()
    images = [n for n in archive.namelist() if n != "manifest.json"]
    assert len(images) == len(views) == 20


def test_manifest_records_the_official_parameters():
    import io
    import json
    import zipfile

    data, _ = _bundle(_textured(128))
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("manifest.json"))
    assert manifest["transform_source"] == "src.data.get_eval_transform"
    by_name = {v["transform"]: v for v in manifest["views"]}
    assert by_name["jpeg_30"]["params"]["quality"] == 30
    assert by_name["blur_2.0"]["params"]["sigma"] == 2.0
    assert by_name["resize_0.25"]["params"]["scale"] == 0.25
    assert by_name["noise_0.10"]["params"]["sigma"] == 0.1
    assert by_name["crop_0.80"]["params"]["ratio"] == 0.8


def test_exported_pixels_match_the_official_transform_exactly():
    """A PNG export must not re-compress a view that already carries artifacts."""
    import io
    import zipfile

    import numpy as np

    from src.data import get_eval_transform

    base = _textured(128)
    data, _ = _bundle(base, names={"jpeg_30", "blur_2.0"})
    archive = zipfile.ZipFile(io.BytesIO(data))
    for name in ("jpeg_30", "blur_2.0"):
        exported = Image.open(io.BytesIO(archive.read("photo__%s.png" % name))).convert("RGB")
        direct = get_eval_transform(name)(base.convert("RGB"))
        assert np.array_equal(np.asarray(exported), np.asarray(direct))


def test_bundle_is_reproducible_pixel_for_pixel():
    """Seeded noise means the same image must always export the same bytes."""
    import io
    import zipfile

    import numpy as np

    base = _textured(128)
    first = zipfile.ZipFile(io.BytesIO(_bundle(base)[0]))
    second = zipfile.ZipFile(io.BytesIO(_bundle(base)[0]))
    for name in [n for n in first.namelist() if n.endswith(".png")]:
        a = np.asarray(Image.open(io.BytesIO(first.read(name))))
        b = np.asarray(Image.open(io.BytesIO(second.read(name))))
        assert np.array_equal(a, b), name


def test_jpeg_export_is_offered_as_an_alternative():
    import io
    import zipfile

    data, _ = _bundle(_textured(128), names={"clean"}, fmt="JPEG")
    assert "photo__clean.jpg" in zipfile.ZipFile(io.BytesIO(data)).namelist()


def test_chaining_returns_a_single_image_not_a_set():
    """The chain mode's whole point: many transforms, one file out."""
    import app.studio as studio

    base = _textured(128)
    out = studio._chained(base, ["jpeg_30", "blur_2.0", "resize_0.25"])
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == base.size


def test_chaining_is_order_dependent_and_harsher_than_one_transform():
    import numpy as np

    import app.studio as studio
    from src.data import get_eval_transform

    base = _textured(160)
    single = get_eval_transform("blur_2.0")(base.convert("RGB"))
    chained = studio._chained(base, ["jpeg_30", "blur_2.0", "noise_0.10"])
    assert not np.array_equal(np.asarray(single), np.asarray(chained))

    # Order matters, so the composition is genuinely sequential.
    forward = studio._chained(base, ["blur_2.0", "noise_0.10"])
    reverse = studio._chained(base, ["noise_0.10", "blur_2.0"])
    assert not np.array_equal(np.asarray(forward), np.asarray(reverse))


def test_chain_encodes_to_both_formats():
    import io

    import app.studio as studio

    chained = studio._chained(_textured(96), ["jpeg_50", "crop_0.80"])
    png = studio._encode(chained, "PNG")
    jpg = studio._encode(chained, "JPEG")
    assert Image.open(io.BytesIO(png)).format == "PNG"
    assert Image.open(io.BytesIO(jpg)).format == "JPEG"


def test_every_tab_renders_before_anything_is_uploaded(tmp_path):
    """Regression: early returns in the Detector tab used to abort main().

    The Detector body was inline in `main()`, so `return` when no image was
    uploaded skipped the `with lab:` and `with evidence:` blocks entirely and
    both of those tabs rendered empty.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=120)
    app.run()
    app.text_input[0].set_value(_tiny_checkpoint(tmp_path)).run()

    assert not app.exception
    body = "".join(str(m.value) for m in app.markdown)
    assert "Transform lab" in body, "lab tab did not render with no image uploaded"
    assert "Published results" in body, "results tab did not render with no image uploaded"
    # The lab has its own uploader, so both must be present.
    assert len(app.file_uploader) == 2


def test_lab_is_reachable_without_a_checkpoint():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(STUDIO, default_timeout=60)
    app.run()
    assert not app.exception
    assert any("Transform lab" in str(m.value) for m in app.markdown)


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
