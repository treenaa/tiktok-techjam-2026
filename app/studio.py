"""Spectral Evidence -- interactive AIGC detection with visible robustness.

Every view is shown as a pair: the image as a person sees it, beside the log-FFT
magnitude the detector's forensic branch actually consumes. Walking the official
degradation ladder shows the picture and its frequency signature decay together,
next to the probability -- so robustness is something you watch rather than read
off a table.

Scoring goes through the same `load_artifact` / `Predictor` path as `predict.py`,
and decoding through the same `src.data.load_image`, so this demo cannot disagree
with the submission it demonstrates.

    streamlit run app/studio.py -- --checkpoint runs/<run>/best.pt --device auto
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import altair as alt
import pandas as pd
import streamlit as st

# `streamlit run app/studio.py` puts app/ on sys.path, not the repository root,
# so `app.*` and `src.*` are otherwise unimportable. Same bootstrap predict.py uses.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.spectral import ambient, pair_strip, spectral_distance  # noqa: E402
from app.theme import ACCENT, ACCENT_WARM, INK, MUTED, RULE, STABLE, css  # noqa: E402
from src.data import describe_eval_transforms, get_eval_transform, load_image  # noqa: E402
from src.inference import InferenceError, Predictor, load_artifact  # noqa: E402

#: Family order for the ladder: identity first, then increasing structural damage.
FAMILY_ORDER = ("identity", "jpeg", "blur", "resize", "noise", "jitter", "crop")

#: The five views that carry the story in a short demo.
CORE_VIEWS = ("clean", "jpeg_30", "blur_2.0", "resize_0.25", "noise_0.10", "crop_0.80")

RESULTS_DIR = "results"


# --------------------------------------------------------------------------
# arguments and model loading
# --------------------------------------------------------------------------
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=os.environ.get("AIGC_CHECKPOINT", ""))
    parser.add_argument("--device", default=os.environ.get("AIGC_DEVICE", "auto"))
    parsed, _ = parser.parse_known_args(sys.argv[1:])
    return parsed


@st.cache_resource(show_spinner="Reconstructing detector from checkpoint…")
def _artifact(checkpoint: str, device: str):
    """Cached because the checkpoint is ~330 MB and the script reruns on every click."""
    return load_artifact(checkpoint, device=device)


def _ordered_views(all_transforms: bool) -> List[Dict[str, Any]]:
    described = {item["name"]: item for item in describe_eval_transforms()}
    names = list(described) if all_transforms else [n for n in CORE_VIEWS if n in described]

    def key(name: str):
        item = described[name]
        family = item.get("family", "")
        rank = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
        severity = item.get("severity")
        return (rank, severity if isinstance(severity, int) else 99, name)

    return [described[name] for name in sorted(names, key=key)]


def _param_summary(metadata: Dict[str, Any]) -> str:
    count = metadata.get("parameter_count")
    limit = 2_000_000_000
    if not isinstance(count, int):
        return "parameters unknown"
    return "%.1fM params (%.2f%% of the 2B limit)" % (count / 1e6, 100.0 * count / limit)


# --------------------------------------------------------------------------
# page sections
# --------------------------------------------------------------------------
def _masthead(artifact: Optional[Any]) -> None:
    st.markdown(
        """
<div class="sx-mast">
  <p class="sx-eyebrow">TikTok TechJam 2026 · Robust AIGC detection</p>
  <h1 class="sx-word">Spectral <em>Evidence</em></h1>
  <p class="sx-thesis">A detector is only useful if its answer survives the trip through
  compression, blur and resizing. This shows both halves of that: the image as you see it,
  and the frequency signature the model reads — degrading together, side by side.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if artifact is None:
        return
    meta = artifact.metadata or {}
    tags = [
        ("backbone", str(meta.get("backbone") or meta.get("model_kwargs", {}).get("backbone", "—"))),
        ("weights", str(meta.get("backbone_model_id") or meta.get("model_kwargs", {}).get("model_id", "—"))),
        ("size", _param_summary(meta)),
        ("threshold", "%.4f · %s" % (artifact.threshold, artifact.threshold_source)),
        ("device", artifact.device),
    ]
    st.markdown(
        '<div class="sx-prov">%s</div>'
        % "".join('<span class="sx-tag">%s <b>%s</b></span>' % (k, v) for k, v in tags),
        unsafe_allow_html=True,
    )


def _verdict(result: Dict[str, Any]) -> None:
    probability = result["probability_aigc"]
    threshold = result["threshold"]
    is_aigc = probability >= threshold
    flips = sum(
        1
        for name, score in result["scores"].items()
        if name != "clean" and (score >= threshold) != is_aigc
    )
    verdict_colour = ACCENT_WARM if is_aigc else STABLE

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            """
<div class="sx-verdict">
  <p class="sx-eyebrow">Clean image</p>
  <div class="sx-prob">%.1f<span style="font-size:.45em;color:%s">%%</span></div>
  <div class="sx-label" style="color:%s">%s</div>
  <div class="sx-bar"><i style="width:%.2f%%"></i><span class="sx-thresh" style="left:%.2f%%"></span></div>
  <p class="sx-note">white marker = decision threshold %.4f (%s)</p>
</div>
"""
            % (
                probability * 100,
                MUTED,
                verdict_colour,
                "AI-generated" if is_aigc else "Authentic",
                probability * 100,
                threshold * 100,
                threshold,
                result["threshold_source"],
            ),
            unsafe_allow_html=True,
        )
    with right:
        a, b = st.columns(2)
        stable = result["class_stable"]
        a.markdown(
            '<div class="sx-stat"><span class="k">Verdict held</span>'
            '<span class="v %s">%s</span></div>'
            % ("ok" if stable else "bad", "YES" if stable else "NO"),
            unsafe_allow_html=True,
        )
        b.markdown(
            '<div class="sx-stat"><span class="k">Views that flipped</span>'
            '<span class="v %s">%d</span></div>' % ("ok" if flips == 0 else "bad", flips),
            unsafe_allow_html=True,
        )
        c, d = st.columns(2)
        c.markdown(
            '<div class="sx-stat"><span class="k">Mean drift</span>'
            '<span class="v">%.3f</span></div>' % result["mean_absolute_drift"],
            unsafe_allow_html=True,
        )
        d.markdown(
            '<div class="sx-stat"><span class="k">Worst drift</span>'
            '<span class="v">%.3f</span></div>' % result["max_absolute_drift"],
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sx-note" style="margin-top:.9rem">%s</p>' % result["stability_note"],
            unsafe_allow_html=True,
        )


def _ladder(image, result: Dict[str, Any], views: Sequence[Dict[str, Any]], show_distance: bool) -> None:
    scores = result["scores"]
    threshold = result["threshold"]
    clean_score = scores.get("clean", result["probability_aigc"])
    clean_class = clean_score >= threshold
    rgb = image.convert("RGB")

    columns_per_row = 4
    for start in range(0, len(views), columns_per_row):
        row = views[start : start + columns_per_row]
        cells = st.columns(len(row), gap="small")
        for cell, view in zip(cells, row):
            name = view["name"]
            score = scores.get(name)
            if score is None:
                continue
            degraded = get_eval_transform(name)(rgb)
            flipped = (score >= threshold) != clean_class
            delta = score - clean_score
            is_clean = name == "clean"

            detail = _describe_params(view)
            if show_distance and not is_clean:
                detail += " · Δspectrum %.2f" % spectral_distance(rgb, degraded)

            # One markdown block per card: Streamlit sanitizes each call
            # separately, so a card opened in one call would not enclose an
            # element emitted by the next.
            cell.markdown(
                '<div class="sx-card %(state)s">'
                '  <div class="sx-cap"><span class="sx-name">%(name)s</span>'
                '    <span class="sx-delta %(dcls)s">%(delta)s</span></div>'
                '  %(img)s'
                '  <div class="sx-pair"><span>image</span><span>spectrum</span></div>'
                '  <div class="sx-p">%(pct).1f%%</div>'
                '  <div class="sx-meta">%(detail)s</div>'
                '  %(flag)s'
                "</div>"
                % {
                    "state": "clean" if is_clean else ("flip" if flipped else ""),
                    "name": name,
                    "dcls": "" if is_clean else ("bad" if abs(delta) >= 0.15 else "ok"),
                    "delta": "reference" if is_clean else "%+.3f" % delta,
                    "img": _img_tag(pair_strip(degraded)),
                    "pct": score * 100,
                    "detail": detail,
                    "flag": '<div class="sx-flag">class flipped</div>' if flipped else "",
                },
                unsafe_allow_html=True,
            )


def _img_tag(image, quality: int = 90) -> str:
    """Embed a PIL image directly in the card markup as a data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return '<img class="sx-img" src="data:image/jpeg;base64,%s" alt="" />' % encoded


def _describe_params(view: Dict[str, Any]) -> str:
    params = view.get("params") or {}
    readable = {k: v for k, v in params.items() if k != "seed"}
    if not readable:
        return "unmodified"
    return ", ".join("%s %s" % (k, v) for k, v in readable.items())


def _drift_chart(result: Dict[str, Any], views: Sequence[Dict[str, Any]]) -> None:
    threshold = result["threshold"]
    scores = result["scores"]
    frame = pd.DataFrame(
        [
            {"view": v["name"], "probability": scores[v["name"]], "order": i}
            for i, v in enumerate(views)
            if v["name"] in scores
        ]
    )
    order = frame.sort_values("order")["view"].tolist()

    line = (
        alt.Chart(frame)
        .mark_line(color=ACCENT, strokeWidth=2, point=alt.OverlayMarkDef(color=ACCENT_WARM, size=55))
        .encode(
            x=alt.X("view:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45, labelColor=MUTED, domainColor=RULE, tickColor=RULE)),
            y=alt.Y(
                "probability:Q",
                title="P(AI-generated)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor=RULE, domainColor=RULE),
            ),
            tooltip=["view", alt.Tooltip("probability:Q", format=".4f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color=INK, strokeDash=[4, 4], strokeWidth=1)
        .encode(y="t:Q")
    )
    st.altair_chart(
        (line + rule).properties(height=260, background="#141C2B"),
        width="stretch",
    )
    st.markdown(
        '<p class="sx-note">Dashed line is the decision threshold. Every crossing is an image '
        'whose verdict changed because of nothing but a repost-grade transformation.</p>',
        unsafe_allow_html=True,
    )


def _evidence() -> None:
    reports = sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "report.json")))
    if not reports:
        st.info("No evaluation reports found under `results/`.")
        return

    rows = []
    for path in reports:
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        summary = report.get("robustness_summary", {})
        stability = report.get("stability_summary", {})
        rows.append(
            {
                "run": os.path.basename(os.path.dirname(path)),
                "n": report.get("n_samples"),
                "clean": summary.get("clean_auroc"),
                "transformed": summary.get("mean_transformed_auroc"),
                "worst_name": summary.get("worst_case_transform"),
                "worst": summary.get("worst_case_transformed_auroc"),
                "flip": stability.get("mean_class_flip_rate"),
            }
        )

    body = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (
            r["run"],
            "{:,}".format(r["n"]) if isinstance(r["n"], int) else "—",
            _fmt(r["clean"]),
            _fmt(r["transformed"]),
            r["worst_name"] or "—",
            _fmt(r["worst"]),
            _fmt(r["flip"], 3),
        )
        for r in rows
    )
    st.markdown(
        '<table class="sx-tbl"><thead><tr>'
        "<th>Run</th><th>Test n</th><th>Clean AUROC</th><th>Mean transformed</th>"
        "<th>Worst transform</th><th>Worst AUROC</th><th>Mean class-flip</th>"
        "</tr></thead><tbody>%s</tbody></table>" % body,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sx-note" style="margin-top:.9rem">Read straight from '
        "<code>results/&lt;run&gt;/report.json</code> at page load — nothing here is typed by hand.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="sx-foot" style="margin-top:1.4rem">
<b>How to read these honestly.</b> The SID_Set and WildFake test sets are small (900 and 600
images), so their robustness is encouraging rather than conclusive. The robustness-aware run
was trained on the same corruption families it is evaluated against, so it demonstrates
robustness on trained corruptions and is not evidence of transfer to unseen ones. Paired
augmentation, the consistency loss and the forensic branch were enabled together, so no
individual component is credited.
</div>
""",
        unsafe_allow_html=True,
    )


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return ("%." + str(places) + "f") % float(value)
    except (TypeError, ValueError):
        return "—"


def _onboarding() -> None:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown(
            """
<div class="sx-verdict">
  <p class="sx-eyebrow">No detector loaded</p>
  <p style="color:%s;line-height:1.6;margin:0">Point this at a trained checkpoint to begin.
  Set it in the sidebar, or pass it on the command line:</p>
</div>
"""
            % MUTED,
            unsafe_allow_html=True,
        )
        st.code(
            "streamlit run app/studio.py -- \\\n"
            "  --checkpoint runs/baseline_dino_wildfake_v3/best.pt \\\n"
            "  --device auto",
            language="bash",
        )
        st.markdown(
            '<p class="sx-note">Checkpoints are published on the repository\'s GitHub '
            "Releases page; see the README for the download command and checksums.</p>",
            unsafe_allow_html=True,
        )
    with right:
        st.image(ambient(), width="stretch")
        st.markdown(
            '<div class="sx-pair"><span>a frequency signature</span></div>',
            unsafe_allow_html=True,
        )


def _decode(uploaded) -> Any:
    """Decode through the same loader `predict.py` uses, via a temporary file."""
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".img"
    descriptor, path = tempfile.mkstemp(prefix="aigc-studio-", suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(uploaded.getvalue())
        return load_image(path, on_error="raise")
    finally:
        if os.path.exists(path):
            os.unlink(path)


# --------------------------------------------------------------------------
def main() -> None:
    args = _arguments()
    st.set_page_config(
        page_title="Spectral Evidence — Robust AIGC Detection",
        page_icon="◐",
        layout="wide",
    )
    st.markdown(css(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="sx-eyebrow">Detector</p>', unsafe_allow_html=True)
        checkpoint = st.text_input("Checkpoint", value=args.checkpoint, label_visibility="collapsed")
        device = st.selectbox("Device", ("auto", "cpu", "mps", "cuda"), index=0)
        st.markdown('<hr class="sx-rule" style="margin:1.2rem 0">', unsafe_allow_html=True)
        st.markdown('<p class="sx-eyebrow">Ladder</p>', unsafe_allow_html=True)
        depth = st.radio(
            "Views",
            ("Core six", "All twenty"),
            index=0,
            help="The full official suite is 20 named transforms; the core six carry the story.",
            label_visibility="collapsed",
        )
        show_distance = st.toggle("Show spectral distance", value=True)

    artifact = None
    error: Optional[str] = None
    if checkpoint:
        try:
            artifact = _artifact(checkpoint, device)
        except Exception as exc:  # surfaced in the page, never as a traceback
            error = str(exc)

    _masthead(artifact)

    if error:
        st.error("Could not load that checkpoint: %s" % error)
        return
    if artifact is None:
        _onboarding()
        st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)
        st.markdown('<h2 class="sx-h">Published results</h2>', unsafe_allow_html=True)
        _evidence()
        return

    detector, evidence = st.tabs(["Detector", "Published results"])

    with detector:
        uploaded = st.file_uploader(
            "Upload an image",
            type=("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"),
            label_visibility="collapsed",
        )
        if uploaded is None:
            st.markdown(
                '<p class="sx-sub">Drop in an image to score it and watch its frequency '
                "signature survive — or fail to survive — the official transformation suite.</p>",
                unsafe_allow_html=True,
            )
            _onboarding_preview()
            return

        views = _ordered_views(depth == "All twenty")
        try:
            image = _decode(uploaded)
            predictor = Predictor(artifact, batch_size=max(len(views), 1))
            result = predictor.diagnose_image(image, [v["name"] for v in views])
        except Exception as exc:
            st.error("Could not analyze this image: %s" % exc)
            return

        _verdict(result)

        st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="sx-h">The degradation ladder</h2>'
            '<p class="sx-sub">Each card pairs the transformed image with the log-FFT magnitude '
            "the forensic branch consumes. Blur empties the outer field; noise floods it; JPEG "
            "prints its block grid into it. A red border marks a view whose verdict flipped.</p>",
            unsafe_allow_html=True,
        )
        _ladder(image, result, views, show_distance)

        st.markdown('<hr class="sx-rule">', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="sx-h">Score under degradation</h2>'
            '<p class="sx-sub">One line, twenty ways of reposting the same picture.</p>',
            unsafe_allow_html=True,
        )
        _drift_chart(result, views)

        st.markdown(
            """
<div class="sx-foot" style="margin-top:2rem">
<b>Stability is not correctness.</b> A verdict that survives every transform can still be the
wrong verdict. This prototype may fail on unseen generators, edited or composite images,
screenshots, and domains unlike its training data.
</div>
""",
            unsafe_allow_html=True,
        )

    with evidence:
        st.markdown(
            '<h2 class="sx-h">Published results</h2>'
            '<p class="sx-sub">Measured runs recorded under <code>results/</code>. '
            "No number on this page is entered by hand.</p>",
            unsafe_allow_html=True,
        )
        _evidence()


def _onboarding_preview() -> None:
    columns = st.columns(4, gap="small")
    for column, (title, blurb) in zip(
        columns,
        (
            ("image + spectrum", "every view shown as the pair the model sees"),
            ("20 transforms", "the full official suite, scored in under a second"),
            ("flip detection", "views where the verdict changed are marked"),
            ("measured drift", "how far the score moved, not just whether it did"),
        ),
    ):
        with column:
            st.markdown(
                '<div class="sx-card"><div class="sx-name">%s</div>'
                '<div class="sx-meta" style="margin-top:.35rem">%s</div></div>' % (title, blurb),
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
