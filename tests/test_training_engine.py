from __future__ import annotations

import json

import pytest
from torch.utils.data import DataLoader, Dataset

from src.training import (
    RobustBinaryObjective,
    Trainer,
    TrainingConfig,
    read_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)

torch = pytest.importorskip("torch")


class BinaryTensorDataset(Dataset):
    def __init__(self, paired=False):
        self.paired = paired
        self.values = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        image = torch.tensor([[[value]]], dtype=torch.float32)
        label = int(value > 0)
        common = {
            "label": label,
            "source_id": "source-%d" % index,
            "image_path": "image-%d.png" % index,
        }
        if self.paired:
            return {"clean": image, "augmented": image * 0.8, **common}
        return {"image": image, **common}


class ScalarDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, images):
        return self.linear(images.flatten(1)).squeeze(-1)


def loader(paired=False, shuffle=False, seed=4):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(BinaryTensorDataset(paired), batch_size=4, shuffle=shuffle, generator=generator)


def test_paired_train_epoch_reports_both_losses_and_drift():
    model = ScalarDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = train_one_epoch(
        model,
        loader(paired=True),
        optimizer,
        RobustBinaryObjective(consistency_weight=0.5),
        device="cpu",
        scaler=scaler,
        amp=False,
        max_grad_norm=1.0,
    )
    assert metrics["train_loss"] > 0
    assert metrics["train_augmented_classification_loss"] > 0
    assert metrics["train_consistency_loss"] >= 0
    assert metrics["train_mean_absolute_drift"] >= 0


def test_validation_selects_threshold_and_never_updates_model():
    model = ScalarDetector()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = validate_one_epoch(
        model,
        loader(),
        RobustBinaryObjective(),
        device="cpu",
        threshold_metric="f1",
        amp=False,
    )
    assert metrics["val_threshold_source"] == "validation"
    assert 0 <= metrics["val_threshold"] <= 1
    assert metrics["val_auroc"] is not None
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_trainer_writes_best_last_history_and_resumes(tmp_path):
    first_config = TrainingConfig(
        epochs=2,
        batch_size=4,
        optimizer="sgd",
        lr=0.2,
        weight_decay=0.0,
        max_grad_norm=1.0,
        early_stopping_patience=None,
    )
    first = Trainer(ScalarDetector(), first_config, output_dir=str(tmp_path), device="cpu")
    first_result = first.fit(loader(shuffle=True), loader())
    assert len(first_result.history) == 2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "history.json").exists()
    best = read_checkpoint(str(tmp_path / "best.pt"))
    assert best["threshold"] == best["best_threshold"]
    assert best["threshold_source"] == "validation"

    resumed_config = TrainingConfig(
        epochs=3,
        batch_size=4,
        optimizer="sgd",
        lr=0.2,
        weight_decay=0.0,
        max_grad_norm=1.0,
        early_stopping_patience=None,
    )
    resumed = Trainer(ScalarDetector(), resumed_config, output_dir=str(tmp_path), device="cpu")
    result = resumed.fit(
        loader(shuffle=True),
        loader(),
        resume_from=str(tmp_path / "last.pt"),
    )
    assert len(result.history) == 3
    assert result.history[:2] == first_result.history
    assert read_checkpoint(str(tmp_path / "last.pt"))["epoch"] == 2
    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["best_threshold"] == result.best_threshold


def test_resume_rejects_changed_training_method(tmp_path):
    config = TrainingConfig(epochs=1, batch_size=4, early_stopping_patience=None)
    Trainer(ScalarDetector(), config, output_dir=str(tmp_path), device="cpu").fit(
        loader(shuffle=True), loader()
    )
    changed = TrainingConfig(
        epochs=2,
        batch_size=4,
        lr=0.01,
        early_stopping_patience=None,
    )
    with pytest.raises(ValueError, match="resume config differs"):
        Trainer(ScalarDetector(), changed, output_dir=str(tmp_path), device="cpu").fit(
            loader(shuffle=True), loader(), resume_from=str(tmp_path / "last.pt")
        )
