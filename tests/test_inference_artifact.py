from __future__ import annotations

import pytest

from src.data import build_preprocess
from src.inference import InferenceError, load_artifact

torch = pytest.importorskip("torch")


class MeanDetector(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, images):
        return images.mean(dim=(1, 2, 3)) * self.scale


def model_factory(scale=2.0):
    return MeanDetector(scale)


def preprocess_factory(image_size=8):
    return build_preprocess("none", image_size=image_size)


def checkpoint(tmp_path, metadata=True):
    model = model_factory(scale=3.0)
    payload = {
        "model_state_dict": model.state_dict(),
        "best_threshold": 0.37,
        "threshold_source": "validation",
    }
    if metadata:
        payload["run_metadata"] = {
            "model_factory": "fake:model",
            "model_kwargs": {"scale": 3.0},
            "preprocess_factory": "fake:preprocess",
            "preprocess_kwargs": {"image_size": 8},
        }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return str(path)


def test_self_describing_checkpoint_restores_model_preprocess_and_threshold(tmp_path, monkeypatch):
    def importer(value):
        return model_factory if value == "fake:model" else preprocess_factory

    monkeypatch.setattr("src.inference.artifact.import_callable", importer)
    artifact = load_artifact(checkpoint(tmp_path), device="cpu")
    assert isinstance(artifact.model, MeanDetector)
    assert float(artifact.model.scale) == pytest.approx(3.0)
    assert artifact.preprocess.image_size == (8, 8)
    assert artifact.threshold == pytest.approx(0.37)
    assert artifact.threshold_source == "validation"
    assert artifact.device == "cpu"


def test_explicit_kwargs_merge_over_checkpoint_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.inference.artifact.import_callable",
        lambda value: model_factory if value == "override:model" else preprocess_factory,
    )
    artifact = load_artifact(
        checkpoint(tmp_path),
        model_factory="override:model",
        model_kwargs={"scale": 3.0},
        preprocess_factory="override:preprocess",
        preprocess_kwargs={"image_size": 12},
        threshold=0.6,
        device="cpu",
    )
    assert artifact.preprocess.image_size == (12, 12)
    assert artifact.threshold == 0.6
    assert artifact.threshold_source == "explicit_override"


def test_checkpoint_threshold_takes_precedence_over_historical_best_threshold(
    tmp_path, monkeypatch
):
    path = checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["threshold"] = 0.61
    torch.save(payload, path)
    monkeypatch.setattr(
        "src.inference.artifact.import_callable",
        lambda value: model_factory if value == "fake:model" else preprocess_factory,
    )

    artifact = load_artifact(path, device="cpu")

    assert artifact.threshold == pytest.approx(0.61)


def test_plain_checkpoint_requires_explicit_factories(tmp_path):
    with pytest.raises(InferenceError, match="not self-describing"):
        load_artifact(checkpoint(tmp_path, metadata=False), device="cpu")


def test_state_loading_is_strict(tmp_path):
    with pytest.raises(RuntimeError, match="state_dict"):
        load_artifact(
            checkpoint(tmp_path, metadata=False),
            model_factory=lambda: torch.nn.Linear(2, 1),
            preprocess_factory=preprocess_factory,
            device="cpu",
        )


def test_invalid_threshold_is_rejected(tmp_path):
    with pytest.raises(InferenceError, match="threshold"):
        load_artifact(
            checkpoint(tmp_path, metadata=False),
            model_factory=lambda: MeanDetector(3.0),
            preprocess_factory=preprocess_factory,
            threshold=2.0,
            device="cpu",
        )
