from __future__ import annotations

import pytest

from src.training import RobustBinaryObjective, TrainingConfig, TrainingError

torch = pytest.importorskip("torch")


def test_existing_phase_one_config_maps_without_silent_changes():
    config = TrainingConfig.from_mapping(
        {
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adamw",
            "lr": 3e-4,
            "weight_decay": 0.01,
            "loss": "bce_with_logits",
            "freeze_backbone": True,
            "augment": "none",
            "consistency_weight": 0.0,
            "early_stopping": {"monitor": "val_auroc", "patience": 3},
            "num_workers": 4,
        }
    )
    assert config.augment == "none"
    assert config.consistency_weight == 0.0
    assert config.early_stopping_monitor == "val_auroc"
    assert config.early_stopping_patience == 3


def test_robust_config_aliases_and_nested_scheduler():
    config = TrainingConfig.from_mapping(
        {
            "augment": "official",
            "augment_families": ["jpeg", "blur"],
            "augment_weights": [0.7, 0.3],
            "scheduler": {"name": "cosine", "min_lr": 1e-6},
            "early_stopping": False,
        }
    )
    assert config.augment == "competition"
    assert config.augment_families == ("jpeg", "blur")
    assert config.scheduler == "cosine"
    assert config.min_lr == 1e-6
    assert config.early_stopping_patience is None


@pytest.mark.parametrize(
    "values,match",
    [
        ({"loss": "cross_entropy"}, "BCEWithLogitsLoss"),
        ({"consistency_weight": -1}, "consistency_weight"),
        ({"mystery": 1}, "unknown training"),
        ({"augment_weights": [1]}, "requires augment_families"),
        ({"amp": True, "max_grad_norm": 0}, "max_grad_norm"),
        ({"augment": "none", "consistency_weight": 1}, "requires augment"),
        ({"augment": "none", "clean_loss_weight": 0}, "clean_loss_weight"),
    ],
)
def test_training_config_rejects_methodological_typos(values, match):
    with pytest.raises(TrainingError, match=match):
        TrainingConfig.from_mapping(values)


def test_robust_objective_matches_explicit_formula_and_backpropagates_both_views():
    clean = torch.tensor([-1.0, 2.0], requires_grad=True)
    augmented = torch.tensor([-0.5, 1.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0])
    objective = RobustBinaryObjective(
        clean_weight=1.0,
        augmented_weight=0.5,
        consistency_weight=2.0,
    )
    output = objective(clean, labels, augmented)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(clean, labels)
        + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(augmented, labels)
        + 2.0 * torch.nn.functional.mse_loss(torch.sigmoid(clean), torch.sigmoid(augmented))
    )
    assert float(output.total.detach()) == pytest.approx(float(expected.detach()))
    output.total.backward()
    assert clean.grad is not None
    assert augmented.grad is not None


def test_consistency_requires_a_paired_view_and_bce_receives_logits():
    objective = RobustBinaryObjective(consistency_weight=1.0)
    logits = torch.tensor([3.0, -3.0])
    labels = torch.tensor([1.0, 0.0])
    with pytest.raises(TrainingError, match="requires augmented"):
        objective(logits, labels)
    no_consistency = RobustBinaryObjective(consistency_weight=0.0)
    result = no_consistency(logits, labels)
    assert result.clean_classification == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    )


def test_objective_rejects_probable_interface_shape_errors():
    objective = RobustBinaryObjective()
    with pytest.raises(TrainingError, match=r"shape \(B,\)"):
        objective(torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(TrainingError, match="NaN"):
        objective(torch.tensor([0.0, float("nan")]), torch.zeros(2))
