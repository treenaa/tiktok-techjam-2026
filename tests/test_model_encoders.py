from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from src.models import (
    ModelError,
    OptionalDependencyError,
    VisionEncoder,
    create_backbone,
    pool_vision_output,
)

torch = pytest.importorskip("torch")


class DummyVisionBackbone(torch.nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.projection = torch.nn.Linear(3, hidden_size)
        self.dropout = torch.nn.Dropout(0.9)

    def forward(self, pixel_values):
        pooled_rgb = pixel_values.mean(dim=(-2, -1))
        token = self.dropout(self.projection(pooled_rgb))
        hidden = torch.stack((token, token + 1.0, token + 2.0), dim=1)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=token + 3.0)


def test_pooling_modes_are_explicit():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    output = SimpleNamespace(last_hidden_state=hidden, pooler_output=torch.tensor([[8.0, 9.0]]))
    assert torch.equal(pool_vision_output(output, "cls"), torch.tensor([[1.0, 2.0]]))
    assert torch.equal(pool_vision_output(output, "mean"), torch.tensor([[3.0, 4.0]]))
    assert torch.equal(pool_vision_output(output, "mean_patches"), torch.tensor([[4.0, 5.0]]))
    assert torch.equal(pool_vision_output(output, "pooler"), torch.tensor([[8.0, 9.0]]))


def test_pooling_rejects_missing_or_ambiguous_outputs():
    with pytest.raises(ModelError, match="pooler_output"):
        pool_vision_output({"last_hidden_state": torch.zeros(2, 3, 4)}, "pooler")
    with pytest.raises(ModelError, match="pooling"):
        pool_vision_output(torch.zeros(2, 3, 4), "mystery")


def test_frozen_encoder_stays_eval_inside_training_detector():
    backbone = DummyVisionBackbone()
    encoder = VisionEncoder(backbone, pooling="cls", frozen=True)
    encoder.train()
    assert encoder.training is True
    assert backbone.training is False
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    images = torch.randn(2, 3, 8, 8)
    assert torch.equal(encoder(images), encoder(images)), "frozen dropout must stay disabled"


def test_unfrozen_encoder_propagates_gradients():
    backbone = DummyVisionBackbone()
    encoder = VisionEncoder(backbone, pooling="mean", frozen=False)
    encoder.eval()
    encoder(torch.randn(2, 3, 8, 8)).sum().backward()
    assert backbone.projection.weight.grad is not None


@pytest.mark.parametrize(
    "name,expected_pooling",
    [("ijepa", "mean"), ("dino", "cls"), ("clip", "pooler")],
)
def test_candidate_adapters_share_one_embedding_contract(name, expected_pooling):
    encoder = create_backbone(name, model=DummyVisionBackbone(), frozen=True)
    features = encoder(torch.randn(4, 3, 16, 16))
    assert features.shape == (4, 6)
    assert encoder.pooling == expected_pooling
    assert encoder.spec.name in {"ijepa", "dinov2", "clip"}


def test_encoder_checks_declared_feature_width():
    encoder = VisionEncoder(DummyVisionBackbone(hidden_size=5), feature_dim=7, pooling="cls")
    with pytest.raises(ModelError, match="declared feature_dim"):
        encoder(torch.randn(2, 3, 8, 8))
    with pytest.raises(ModelError, match="positive"):
        VisionEncoder(DummyVisionBackbone(), feature_dim=0)


def test_real_backbone_dependency_error_is_actionable_when_transformers_missing():
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("environment has transformers; downloading is intentionally not tested")
    with pytest.raises(OptionalDependencyError, match="transformers"):
        create_backbone("dinov2", local_files_only=True)
