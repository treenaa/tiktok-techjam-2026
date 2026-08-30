from __future__ import annotations

import pytest
from PIL import Image

from src.data import ManifestRecord, build_preprocess, load_experiment
from src.training import TrainingError
from train import main, translate_model_config

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "filename,backbone,model_id",
    [
        ("baseline_clip.yaml", "clip", "openai/clip-vit-base-patch16"),
        ("baseline_dino.yaml", "dinov2", "facebook/dinov2-base"),
        ("baseline_ijepa.yaml", "ijepa", "facebook/ijepa_vith14_1k"),
    ],
)
def test_existing_baseline_model_configs_translate_to_factory(filename, backbone, model_id):
    experiment = load_experiment("configs/%s" % filename)
    factory, kwargs, preprocess_factory, preprocess_kwargs = translate_model_config(
        experiment.model, experiment.training
    )
    assert factory == "src.models:create_model"
    assert kwargs["backbone"] == backbone
    assert kwargs["model_id"] == model_id
    assert kwargs["architecture"] == "visual"
    assert kwargs["head_hidden_dim"] is None
    assert kwargs["freeze_backbone"] is True
    assert preprocess_factory == "src.models:create_preprocess"
    assert preprocess_kwargs == {"backbone": backbone, "image_size": 224}


def test_translation_rejects_wrong_normalization():
    with pytest.raises(TrainingError, match="conflicts"):
        translate_model_config(
            {"backbone": "clip", "normalization": "imagenet"},
            {"freeze_backbone": True},
        )


def test_train_cli_never_constructs_the_test_dataset(tmp_path, monkeypatch):
    records = []
    examples = (
        (20, 0), (40, 0), (210, 1), (240, 1),
        (30, 0), (50, 0), (200, 1), (230, 1),
    )
    for index, (value, label) in enumerate(examples):
        path = tmp_path / ("image-%d.png" % index)
        Image.new("RGB", (12, 12), (value, value, value)).save(path)
        records.append(ManifestRecord(str(path), label, "source-%d" % index, dataset="tiny"))

    class Experiment:
        name = "cli-smoke"
        seed = 3
        training = {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "optimizer": "sgd",
            "lr": 0.1,
            "weight_decay": 0.0,
            "early_stopping": False,
        }
        model = {"backbone": "dinov2", "image_size": 8}

        def build_splits(self, validate=True):
            # This missing path would fail immediately if train.py ever built a
            # dataset/loader from the test split.
            forbidden_test = ManifestRecord("/definitely/missing-test.png", 0, "test-only")
            return {"train": records[:4], "val": records[4:], "test": [forbidden_test]}

        def fingerprint(self):
            return "fingerprint"

    class MeanDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, images):
            return self.linear(images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)).squeeze(1)

    def importer(spec):
        if spec.endswith("create_model"):
            return lambda **kwargs: MeanDetector()
        return lambda **kwargs: build_preprocess("none", image_size=8)

    monkeypatch.setattr("train.load_experiment", lambda path: Experiment())
    monkeypatch.setattr("train.save_experiment", lambda experiment, path: path)
    monkeypatch.setattr("train._import_callable", importer)
    output = tmp_path / "run"
    assert main(["--config", "fake.yaml", "--output-dir", str(output), "--device", "cpu"]) == 0
    assert (output / "best.pt").exists()
    assert (output / "training_summary.json").exists()
