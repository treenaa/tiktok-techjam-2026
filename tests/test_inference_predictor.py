from __future__ import annotations

import math

import pytest
from PIL import Image

from src.data import build_preprocess
from src.inference import InferenceError, LoadedArtifact, Predictor

torch = pytest.importorskip("torch")


class MeanLogit(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(1, 2, 3)) * 2.0


def artifact():
    model = MeanLogit().eval()
    return LoadedArtifact(
        model=model,
        preprocess=build_preprocess("none", image_size=16),
        device="cpu",
        threshold=0.7,
        threshold_source="validation",
        checkpoint_path="best.pt",
        metadata={},
    )


def image(path, value):
    Image.new("RGB", (20, 20), (value, value, value)).save(path)
    return str(path)


def test_prediction_order_and_probability_convention(tmp_path):
    paths = [image(tmp_path / "b.png", 255), image(tmp_path / "a.png", 0)]
    run = Predictor(artifact(), batch_size=1).predict_paths(paths)
    assert [prediction.image_path for prediction in run.predictions] == paths
    assert run.predictions[0].probability_aigc == pytest.approx(torch.sigmoid(torch.tensor(2.0)))
    assert run.predictions[1].probability_aigc == pytest.approx(0.5)
    assert run.samples_per_second > 0


def test_corrupt_policy_is_explicit_and_never_scores_placeholder(tmp_path):
    valid = image(tmp_path / "good.png", 100)
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    predictor = Predictor(artifact())
    with pytest.raises(InferenceError, match="unreadable"):
        predictor.predict_paths([valid, str(corrupt)], on_error="raise")
    run = predictor.predict_paths([valid, str(corrupt)], on_error="skip")
    assert [prediction.image_path for prediction in run.predictions] == [valid]
    assert len(run.unreadable) == 1
    assert run.unreadable[0][0] == str(corrupt)
    with pytest.raises(InferenceError, match="placeholder"):
        predictor.predict_paths([valid], on_error="placeholder")


def test_empty_inputs_fail_loudly():
    with pytest.raises(InferenceError, match="no supported images"):
        Predictor(artifact()).predict_paths([])


def test_live_diagnostics_reuse_official_names_and_warn_about_correctness():
    source = Image.new("RGB", (32, 32), (180, 120, 90))
    result = Predictor(artifact()).diagnose_image(
        source, ["jpeg_30", "blur_2.0", "resize_0.25", "crop_0.80"]
    )
    assert list(result["scores"])[0] == "clean"
    assert set(result["scores"]) == {"clean", "jpeg_30", "blur_2.0", "resize_0.25", "crop_0.80"}
    assert 0 <= result["mean_absolute_drift"] <= 1
    assert isinstance(result["class_stable"], bool)
    assert "not proof" in result["stability_note"]


def test_predict_image_requires_pil():
    with pytest.raises(InferenceError, match="PIL"):
        Predictor(artifact()).predict_image(torch.zeros(3, 8, 8))
