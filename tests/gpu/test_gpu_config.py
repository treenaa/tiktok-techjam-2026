from __future__ import annotations

import json

import pytest

from src.gpu import GpuConfigError, config_from_mapping, load_gpu_config


def test_defaults_cover_all_three_backbones_and_both_architectures():
    config = config_from_mapping({})
    assert config.model.backbones == ("ijepa", "dinov2", "clip")
    assert len(config.model.variants) == 6
    assert config.budget.parameter_limit == 2_000_000_000


def test_unknown_keys_are_rejected_rather_than_ignored():
    with pytest.raises(GpuConfigError, match="unknown config key"):
        config_from_mapping({"devcie": "cuda"})
    with pytest.raises(GpuConfigError, match="unknown benchmark config key"):
        config_from_mapping({"benchmark": {"batch_size": 8}})
    with pytest.raises(GpuConfigError, match="unknown model.stub key"):
        config_from_mapping({"model": {"stub": {"width": 4}}})


def test_require_cuda_and_allow_cpu_cannot_both_be_set():
    with pytest.raises(GpuConfigError, match="contradict"):
        config_from_mapping({"require_cuda": True, "allow_cpu": True})


def test_invalid_values_are_caught_at_construction():
    with pytest.raises(GpuConfigError, match="positive"):
        config_from_mapping({"benchmark": {"batch_sizes": [0]}})
    with pytest.raises(GpuConfigError, match="must be one of"):
        config_from_mapping({"benchmark": {"precisions": ["int8"]}})
    with pytest.raises(GpuConfigError, match="vram_headroom_fraction"):
        config_from_mapping({"budget": {"vram_headroom_fraction": 1.5}})
    with pytest.raises(GpuConfigError, match="divisible"):
        config_from_mapping({"model": {"stub": {"hidden_size": 10, "heads": 3}}})


def test_stub_image_size_must_tile_into_patches():
    with pytest.raises(GpuConfigError, match="divisible by the stub patch_size"):
        config_from_mapping({"model": {"image_size": 30, "stub": {"patch_size": 16}}})


def test_backbone_aliases_are_normalised_to_lower_case():
    config = config_from_mapping({"model": {"backbones": ["  DINO  ", "CLIP"]}})
    assert config.model.backbones == ("dino", "clip")


def test_shipped_configs_load(tmp_path):
    for name in ("configs/gpu_check.yaml", "configs/gpu_check_smoke.yaml"):
        config = load_gpu_config(name)
        assert config.model.variants
        assert config.budget.parameter_limit == 2_000_000_000
    assert load_gpu_config("configs/gpu_check.yaml").model.backbone_source == "pretrained"
    assert load_gpu_config("configs/gpu_check_smoke.yaml").allow_cpu is True


def test_json_configs_are_accepted_too(tmp_path):
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"seed": 5, "model": {"backbones": ["clip"]}}), encoding="utf-8")
    config = load_gpu_config(str(path))
    assert config.seed == 5
    assert config.model.backbones == ("clip",)


def test_missing_config_names_the_path():
    with pytest.raises(GpuConfigError, match="gpu config not found"):
        load_gpu_config("configs/does-not-exist.yaml")


def test_to_dict_round_trips_through_config_from_mapping(tiny_config):
    assert config_from_mapping(tiny_config.to_dict()) == tiny_config
