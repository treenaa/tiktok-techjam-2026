"""Interactive AIGC detector demo with live robustness diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import streamlit as st

from src.data import load_image
from src.inference import InferenceError, Predictor, load_artifact


DIAGNOSTIC_TRANSFORMS = (
    "clean",
    "jpeg_30",
    "blur_2.0",
    "resize_0.25",
    "noise_0.10",
    "crop_0.80",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=os.environ.get("AIGC_CHECKPOINT", ""))
    parser.add_argument("--device", default=os.environ.get("AIGC_DEVICE", "auto"))
    parsed, _ = parser.parse_known_args(sys.argv[1:])
    return parsed


def _json_object(text: str, label: str) -> Dict[str, Any]:
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InferenceError("%s is invalid JSON: %s" % (label, exc)) from exc
    if not isinstance(value, dict):
        raise InferenceError("%s must be a JSON object" % label)
    return value


@st.cache_resource(show_spinner="Loading detector checkpoint…")
def _load_cached(
    checkpoint: str,
    device: str,
    model_factory: str,
    model_kwargs_json: str,
    preprocess_factory: str,
    preprocess_kwargs_json: str,
):
    return load_artifact(
        checkpoint,
        device=device,
        model_factory=model_factory or None,
        model_kwargs=_json_object(model_kwargs_json, "model kwargs"),
        preprocess_factory=preprocess_factory or None,
        preprocess_kwargs=_json_object(preprocess_kwargs_json, "preprocess kwargs"),
    )


def _decode_upload(uploaded) -> Any:
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".img"
    descriptor, path = tempfile.mkstemp(prefix="aigc-upload-", suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(uploaded.getvalue())
        return load_image(path, on_error="raise")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def main() -> None:
    args = _arguments()
    st.set_page_config(page_title="Robust AIGC Detector", page_icon="🔎", layout="wide")
    st.title("Robust AI-Generated Image Detector")
    st.caption(
        "Estimate P(AI-generated) and inspect whether the score stays stable after common reposting transformations."
    )

    with st.sidebar:
        st.header("Detector")
        checkpoint = st.text_input("Checkpoint", value=args.checkpoint)
        device = st.selectbox("Device", ("auto", "cpu", "mps", "cuda"), index=0)
        with st.expander("Factory overrides for plain checkpoints"):
            model_factory = st.text_input("Model factory", value="")
            model_kwargs = st.text_area("Model kwargs (JSON)", value="{}")
            preprocess_factory = st.text_input("Preprocess factory", value="")
            preprocess_kwargs = st.text_area("Preprocess kwargs (JSON)", value="{}")
        st.caption("Self-describing training checkpoints need no factory overrides.")

    uploaded = st.file_uploader(
        "Upload an image",
        type=("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"),
    )
    if uploaded is None:
        st.info("Choose an image to begin.")
        return
    if not checkpoint:
        st.error("Provide a trained checkpoint in the sidebar.")
        return

    try:
        artifact = _load_cached(
            checkpoint,
            device,
            model_factory,
            model_kwargs,
            preprocess_factory,
            preprocess_kwargs,
        )
        image = _decode_upload(uploaded)
        predictor = Predictor(artifact, batch_size=len(DIAGNOSTIC_TRANSFORMS))
        result = predictor.diagnose_image(image, DIAGNOSTIC_TRANSFORMS)
    except Exception as exc:
        st.error("Could not analyze this image: %s" % exc)
        return

    left, right = st.columns((1, 1))
    with left:
        st.image(image, caption=uploaded.name, use_column_width=True)
    with right:
        st.metric("AI-generated probability", "%.1f%%" % (result["probability_aigc"] * 100))
        st.metric("Prediction", result["prediction"])
        st.write(
            "Decision threshold: %.3f (%s)"
            % (result["threshold"], result["threshold_source"])
        )
        st.metric("Mean score drift", "%.3f" % result["mean_absolute_drift"])
        st.metric("Class stable", "YES" if result["class_stable"] else "NO")

    st.subheader("Robustness panel")
    rows = [
        {"View": name, "P(AI-generated)": score, "Percent": "%.1f%%" % (score * 100)}
        for name, score in result["scores"].items()
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.warning(result["stability_note"])
    st.caption(
        "This prototype may fail on unseen generators, edited/composite images, screenshots, and domains unlike its training data."
    )


if __name__ == "__main__":
    main()
