from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models import (
    BinaryClassificationHead,
    FusionDetector,
    PARAMETER_LIMIT,
    VisionEncoder,
    VisualDetector,
    comparison_configs,
    count_parameters,
    create_model,
    create_preprocess,
    parameter_report,
)
from src.models.forensic import ForensicBranch

torch = pytest.importorskip("torch")


class TinyBackbone(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.conv = torch.nn.Conv2d(3, width, 1)

    def forward(self, pixel_values):
        tokens = self.conv(pixel_values).flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1, keepdim=True)
        hidden = torch.cat((cls, tokens), dim=1)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=cls[:, 0])


def encoder(frozen=True):
    return VisionEncoder(TinyBackbone(), pooling="cls", frozen=frozen)


def test_binary_head_returns_raw_logits_not_probabilities():
    head = BinaryClassificationHead(4, hidden_dim=None, dropout=0.0)
    with torch.no_grad():
        head.layers[-1].weight.zero_()
        head.layers[-1].bias.fill_(3.0)
    logits = head(torch.randn(2, 4))
    assert logits.shape == (2,)
    assert torch.equal(logits, torch.tensor([3.0, 3.0]))


def test_visual_detector_shape_and_frozen_gradient_boundary():
    model = VisualDetector(encoder(frozen=True), hidden_dim=16, dropout=0.0)
    logits = model(torch.randn(3, 3, 16, 16))
    assert logits.shape == (3,)
    logits.sum().backward()
    assert model.encoder.backbone.conv.weight.grad is None
    assert model.classifier.layers[-1].weight.grad is not None


def test_fusion_detector_exposes_both_domains_and_raw_logits():
    model = FusionDetector(
        encoder(frozen=True),
        ForensicBranch(output_dim=10, width=8, normalization=None, dropout=0.0),
        hidden_dim=16,
        dropout=0.0,
    )
    images = torch.rand(2, 3, 20, 20)
    features = model.forward_features(images)
    assert features["visual"].shape == (2, 8)
    assert features["forensic"].shape == (2, 10)
    assert features["fused"].shape == (2, 18)
    assert model(images).shape == (2,)


@pytest.mark.parametrize("architecture", ["visual", "fusion"])
def test_factory_supports_download_free_injected_backbone(architecture):
    model = create_model(
        backbone="dinov2",
        architecture=architecture,
        backbone_model=TinyBackbone(),
        freeze_backbone=True,
        head_hidden_dim=12,
        forensic_dim=7,
        forensic_width=8,
    )
    assert model(torch.randn(2, 3, 16, 16)).shape == (2,)
    assert model.backbone_name == "dinov2"


def test_strict_state_dict_round_trip():
    kwargs = dict(
        backbone="clip",
        architecture="fusion",
        freeze_backbone=True,
        head_hidden_dim=12,
        forensic_dim=7,
        forensic_width=8,
    )
    first = create_model(backbone_model=TinyBackbone(), **kwargs)
    second = create_model(backbone_model=TinyBackbone(), **kwargs)
    second.load_state_dict(first.state_dict(), strict=True)
    first.eval()
    second.eval()
    images = torch.randn(2, 3, 16, 16)
    assert torch.equal(first(images), second(images))


def test_parameter_report_is_consistent_and_checks_competition_limit():
    model = create_model(
        backbone="ijepa",
        architecture="visual",
        backbone_model=TinyBackbone(),
        freeze_backbone=True,
        head_hidden_dim=12,
    )
    report = parameter_report(model)
    assert report["total"] == count_parameters(model)
    assert report["frozen"] == report["total"] - report["trainable"]
    assert report["limit"] == PARAMETER_LIMIT
    assert report["within_limit"] is True
    assert report["components"]["encoder"]["trainable"] == 0


def test_comparison_configs_hold_head_and_freezing_constant():
    configs = comparison_configs(head_hidden_dim=123, head_dropout=0.15)
    assert [config["backbone"] for config in configs] == ["clip", "dinov2", "ijepa"]
    assert {config["head_hidden_dim"] for config in configs} == {123}
    assert {config["head_dropout"] for config in configs} == {0.15}
    assert {config["freeze_backbone"] for config in configs} == {True}


def test_preprocessing_tracks_backbone_family():
    clip = create_preprocess("clip")
    dino = create_preprocess("dinov2")
    assert clip.resize_mode == dino.resize_mode == "shortest"
    assert clip.normalization != dino.normalization
    assert clip.image_size == dino.image_size == (224, 224)
