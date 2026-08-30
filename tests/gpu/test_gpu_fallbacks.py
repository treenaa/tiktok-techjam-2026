"""The 8 GB question: what does not fit, and what should we run instead?"""

from __future__ import annotations

import pytest

from src.gpu import (
    budget_recommendations,
    largest_fitting_batch,
    load_gpu_config,
    planning_budget_bytes,
    suggest_fallbacks,
    vram_budget_check,
)
from src.gpu.report import STATUS_PASS, STATUS_SKIP, STATUS_WARN

GIB = 1024 ** 3
EIGHT_GB = 8172 * 1024 * 1024


def _row(batch_size, *, peak=None, status="ok", precision="fp32", mode="train", fits=None):
    return {
        "backbone": "ijepa",
        "architecture": "fusion",
        "mode": mode,
        "precision": precision,
        "batch_size": batch_size,
        "status": status,
        "images_per_second": 100.0,
        "time_to_first_batch_seconds": 1.0,
        "peak_memory_reserved_bytes": peak,
        "device_total_memory_bytes": EIGHT_GB,
        "vram_budget_bytes": EIGHT_GB,
        "budget_fraction": None if peak is None else peak / EIGHT_GB,
        "fits_vram_budget": fits if fits is not None else (peak is not None and peak / EIGHT_GB <= 0.85),
    }


@pytest.fixture
def eight_gb_config(config_factory):
    return config_factory(
        budget={
            "vram_budget_mb": 8172,
            "vram_headroom_fraction": 0.85,
            "target_train_batch_size": 32,
        }
    )


def test_an_explicit_budget_wins_over_a_larger_measuring_device(eight_gb_config, tiny_config):
    """Measuring on a 24 GB card must not bless a config the 8 GB card cannot run."""
    assert planning_budget_bytes(eight_gb_config, 24 * GIB) == EIGHT_GB
    assert planning_budget_bytes(eight_gb_config, None) == EIGHT_GB
    # A smaller device still wins: you cannot use VRAM that is not there.
    assert planning_budget_bytes(eight_gb_config, 4 * GIB) == 4 * GIB
    # With no configured budget, the device's own capacity is the ceiling.
    assert planning_budget_bytes(tiny_config, 6 * GIB) == 6 * GIB
    assert planning_budget_bytes(tiny_config, None) is None


def test_largest_fitting_batch_comes_from_measurements(eight_gb_config):
    rows = [
        _row(4, peak=2 * GIB),
        _row(8, peak=4 * GIB),
        _row(16, peak=int(7.5 * GIB)),
        _row(32, status="oom", fits=False),
    ]
    assert largest_fitting_batch(rows, rows[-1]) == 8  # 16 is over the 85% headroom


def test_oom_configs_get_batch_and_accumulation_fallbacks(eight_gb_config):
    rows = [_row(4, peak=2 * GIB), _row(8, peak=4 * GIB), _row(32, status="oom", fits=False)]
    suggestions = suggest_fallbacks(rows[-1], rows, eight_gb_config)
    joined = " | ".join(suggestions)
    assert "drop batch_size 32 -> 8" in joined
    assert "gradient accumulation: 4 micro-batches of 8" in joined
    assert "mixed precision" in joined
    assert "gradient checkpointing" in joined
    assert "freeze_backbone" in joined


def test_a_measured_amp_alternative_is_preferred_over_generic_advice(eight_gb_config):
    rows = [
        _row(4, peak=2 * GIB),
        _row(16, status="oom", fits=False),
        _row(16, peak=int(5.0 * GIB), precision="amp"),
    ]
    suggestions = suggest_fallbacks(rows[1], rows, eight_gb_config)
    assert any("amp fp16 at batch 16 was measured to fit" in text for text in suggestions)


def test_when_nothing_fits_the_advice_is_to_re_sweep(eight_gb_config):
    rows = [_row(8, status="oom", fits=False), _row(16, status="oom", fits=False)]
    suggestions = suggest_fallbacks(rows[0], rows, eight_gb_config)
    assert any("no measured batch size fit" in text for text in suggestions)


def test_oversized_images_are_called_out(config_factory):
    config = config_factory(
        model={"image_size": 512, "stub": {"patch_size": 16}},
        budget={"vram_budget_mb": 8172, "target_train_batch_size": 32},
    )
    rows = [_row(32, status="oom", fits=False)]
    assert any("reduce model.image_size from 512" in t for t in suggest_fallbacks(rows[0], rows, config))


def test_recommendations_only_cover_configs_that_do_not_fit(eight_gb_config):
    rows = [
        _row(4, peak=2 * GIB),
        _row(16, peak=int(7.9 * GIB)),  # over the 85% headroom
        _row(32, status="oom", fits=False),
        _row(8, status="skipped"),
    ]
    recommendations = budget_recommendations(rows, eight_gb_config)
    assert [item["batch_size"] for item in recommendations] == [16, 32]
    assert all(item["fallbacks"] for item in recommendations)


def test_budget_check_states_the_ceiling_and_the_offenders(eight_gb_config):
    rows = [_row(4, peak=2 * GIB), _row(32, status="oom", fits=False)]
    recommendations = budget_recommendations(rows, eight_gb_config)
    result = vram_budget_check(rows, recommendations, eight_gb_config)
    assert result.status == STATUS_WARN
    assert "ijepa/fusion train fp32 b32" in result.summary
    assert result.details["vram_budget_bytes"] == EIGHT_GB
    assert result.details["recommendations"]


def test_budget_check_passes_when_everything_fits(eight_gb_config):
    rows = [_row(4, peak=2 * GIB), _row(8, peak=3 * GIB)]
    result = vram_budget_check(rows, budget_recommendations(rows, eight_gb_config), eight_gb_config)
    assert result.status == STATUS_PASS


def test_budget_check_is_skipped_rather_than_faked_without_vram_numbers(tiny_config):
    rows = [dict(_row(4), vram_budget_bytes=None, budget_fraction=None, fits_vram_budget=None)]
    result = vram_budget_check(rows, [], tiny_config)
    assert result.status == STATUS_SKIP
    assert "set budget.vram_budget_mb" in result.summary


def test_the_shipped_8gb_profile_targets_the_real_card():
    config = load_gpu_config("configs/gpu_check_8gb.yaml")
    assert config.budget.vram_budget_mb == 8172
    assert config.budget.vram_budget_bytes == EIGHT_GB
    assert config.budget.target_train_batch_size == 32
    assert config.require_cuda is True
    assert config.model.freeze_backbone is True
    # The sweep must cross the ceiling, or it cannot report where the ceiling is.
    assert max(config.benchmark.batch_sizes) >= 32
    assert "amp" in config.benchmark.precisions
