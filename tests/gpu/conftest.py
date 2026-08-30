"""Shared fixtures for the GPU validation suite.

Every test here must run on a machine with no GPU: the CUDA-only tests are
marked and skipped rather than silently passing, so a green suite on a laptop
never gets mistaken for a validated GPU.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

from src.gpu import config_from_mapping  # noqa: E402


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "cuda: requires a working CUDA device; skipped otherwise"
    )


#: Small enough that the whole suite stays in seconds on CPU.
TINY_OVERRIDES: Dict[str, Any] = {
    "seed": 11,
    "model": {
        "backbones": ["dinov2"],
        "architectures": ["visual"],
        "backbone_source": "stub",
        "image_size": 32,
        "head_hidden_dim": 8,
        "stub": {"hidden_size": 16, "layers": 1, "heads": 2, "patch_size": 8},
    },
    "smoke": {"batch_size": 2, "steps": 1, "paired": True, "amp": True},
    "benchmark": {
        "batch_sizes": [2],
        "warmup_steps": 0,
        "measure_steps": 2,
        "precisions": ["fp32"],
        "modes": ["train", "inference"],
    },
    "determinism": {"batch_size": 2},
}


def make_config(**overrides: Any):
    """A tiny, CPU-runnable config, deep-merged with ``overrides``."""
    values: Dict[str, Any] = {
        "device": "cpu",
        "require_cuda": False,
        "allow_cpu": True,
        **TINY_OVERRIDES,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            merged = dict(values[key])
            merged.update(value)
            values[key] = merged
        else:
            values[key] = value
    return config_from_mapping(values)


@pytest.fixture
def config_factory():
    """Factory fixture, since tests/gpu is not an importable package."""
    return make_config


@pytest.fixture
def tiny_config():
    return make_config()


@pytest.fixture
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return "cuda"
