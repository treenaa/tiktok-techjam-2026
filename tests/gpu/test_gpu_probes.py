from __future__ import annotations

import pytest

from src.gpu import (
    GpuCheckError,
    StubVisionBackbone,
    assert_on_device,
    build_detector,
    build_objective,
    build_step_optimizer,
    inference_step,
    smoke_variant,
    synthetic_batch,
    training_step,
)
from src.gpu.report import STATUS_PASS, STATUS_SKIP

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("backbone", ["ijepa", "dinov2", "clip"])
@pytest.mark.parametrize("architecture", ["visual", "fusion"])
def test_every_backbone_and_architecture_runs_forward_and_backward(
    backbone, architecture, config_factory
):
    """Each pooling mode -- mean, cls, pooler -- must survive a real step."""
    config = config_factory(
        model={
            "backbones": [backbone],
            "architectures": [architecture],
            "backbone_source": "stub",
            "image_size": 32,
            "head_hidden_dim": 8,
            "stub": {"hidden_size": 16, "layers": 1, "heads": 2, "patch_size": 8},
        }
    )
    model = build_detector(backbone, architecture, config.model)
    objective = build_objective(paired=True)
    optimizer = build_step_optimizer(model, config)
    batch = synthetic_batch(2, 32, device="cpu", generator=torch.Generator().manual_seed(0))
    outcome = training_step(model, objective, optimizer, batch, device="cpu")
    assert outcome.logits_finite
    assert outcome.grads_finite is True
    assert outcome.loss is not None and outcome.loss == outcome.loss  # not NaN
    assert outcome.logits_device == "cpu"
    assert outcome.loss_device == "cpu"


def test_stub_backbone_exposes_the_hugging_face_output_contract():
    stub = StubVisionBackbone(hidden_size=16, layers=1, heads=2, patch_size=8)
    output = stub(torch.randn(3, 3, 32, 32))
    assert output.last_hidden_state.shape == (3, 17, 16)  # 4x4 patches plus CLS
    assert output.pooler_output.shape == (3, 16)
    assert stub.config.hidden_size == 16


def test_stub_backbone_rejects_sizes_it_cannot_tile():
    stub = StubVisionBackbone(hidden_size=16, layers=1, heads=2, patch_size=8)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        stub(torch.randn(1, 3, 30, 30))


def test_a_misplaced_buffer_is_reported_by_name(tiny_config):
    """The silent-CPU-fallback trap: one tensor left behind on the host."""
    model = build_detector("dinov2", "fusion", tiny_config.model)
    assert assert_on_device(model, "cpu")["misplaced"] == []

    # Comparing against a device the tensors are not on is exactly the silent
    # fallback this guard exists to catch, and needs no GPU to exercise.
    with pytest.raises(GpuCheckError, match="are not on cuda"):
        assert_on_device(model, "cuda")


def test_logits_shape_violations_name_the_offending_shape(tiny_config):
    class WrongShape(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(images.shape[0], 4)

    batch = synthetic_batch(2, 32, device="cpu", paired=False)
    with pytest.raises(GpuCheckError, match=r"logits shaped \(B,\)"):
        inference_step(WrongShape(), batch, device="cpu")


def test_non_tensor_output_is_rejected():
    class NotATensor(torch.nn.Module):
        def forward(self, images):
            return {"logits": 1.0}

    batch = synthetic_batch(2, 32, device="cpu", paired=False)
    with pytest.raises(GpuCheckError, match="did not return a tensor"):
        inference_step(NotATensor(), batch, device="cpu")


def test_synthetic_batches_are_reproducible_and_paired_shapes_match():
    first = synthetic_batch(4, 32, device="cpu", generator=torch.Generator().manual_seed(3))
    second = synthetic_batch(4, 32, device="cpu", generator=torch.Generator().manual_seed(3))
    assert torch.equal(first["clean"], second["clean"])
    assert torch.equal(first["label"], second["label"])
    assert first["augmented"].shape == first["clean"].shape
    assert set(first["label"].unique().tolist()) <= {0.0, 1.0}


def test_smoke_variant_produces_placement_and_precision_checks(tiny_config):
    results = smoke_variant("dinov2", "visual", tiny_config, device="cpu")
    names = {result.name: result.status for result in results}
    assert names["placement.weights.dinov2/visual"] == STATUS_PASS
    assert names["placement.compute.dinov2/visual"] == STATUS_PASS
    assert names["smoke.fp32.dinov2/visual"] == STATUS_PASS
    # AMP is honestly skipped off CUDA rather than reported as passing.
    assert names["amp.dinov2/visual"] == STATUS_SKIP


def test_a_nan_producing_model_fails_loudly_rather_than_scoring_zero(tiny_config):
    model = build_detector("dinov2", "visual", tiny_config.model)
    with torch.no_grad():
        model.classifier.layers[-1].bias.fill_(float("nan"))
    objective = build_objective(paired=False)
    optimizer = build_step_optimizer(model, tiny_config)
    batch = synthetic_batch(2, 32, device="cpu", paired=False)
    with pytest.raises(GpuCheckError, match="NaN|infinity|non-finite"):
        training_step(model, objective, optimizer, batch, device="cpu")
