from __future__ import annotations

import random

import numpy as np
import pytest

from src.training import (
    EarlyStopping,
    TrainingConfig,
    capture_rng_state,
    checkpoint_payload,
    read_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
)

torch = pytest.importorskip("torch")


def model_and_optimizer():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2, 2)).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def test_rng_state_round_trip_covers_python_numpy_and_torch():
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_checkpoint_is_weights_only_readable_and_restores_strictly(tmp_path, monkeypatch):
    model, optimizer = model_and_optimizer()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stopper = EarlyStopping(mode="max", patience=2)
    stopper.step(0.8)
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=3,
        history=[{"epoch": 3, "val_auroc": 0.8}],
        training_config=TrainingConfig().to_dict(),
        threshold=0.42,
        best_metric=0.8,
        best_epoch=3,
        early_stopping_state=stopper.state_dict(),
        loader_state={"generator": torch.Generator().get_state()},
        run_metadata={"fingerprint": "abc"},
    )
    payload["best_threshold"] = 0.42
    path = save_checkpoint(payload, str(tmp_path / "nested" / "best.pt"))
    loaded = read_checkpoint(path)
    assert loaded["threshold_source"] == "validation"
    assert loaded["epoch"] == 3
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restore_training_checkpoint(path, model=model, optimizer=optimizer, restore_rng=False)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])

    # Jamie's evaluator consumes the exact same model_state_dict key strictly.
    import evaluate as evaluation_cli

    monkeypatch.setattr(
        evaluation_cli,
        "_import_callable",
        lambda spec: (lambda **kwargs: torch.nn.Linear(2, 1)),
    )
    evaluation_model = evaluation_cli.load_model("fake:create", {}, path)
    for key, value in evaluation_model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_checkpoint_schema_is_checked(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        read_checkpoint(str(path))
