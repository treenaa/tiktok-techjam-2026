from __future__ import annotations

import pytest

from src.gpu import CUDNN_CAVEATS, determinism_controls, determinism_report
from src.gpu.report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP

torch = pytest.importorskip("torch")


def _result(results, name):
    return next(result for result in results if result.name == name)


def test_same_seed_reproduces_logits_and_loss(tiny_config):
    payload, results = determinism_report(tiny_config, device="cpu")
    assert payload["runs"] == 2
    variant = payload["variants"][0]
    assert variant["max_abs_init_diff"] == 0.0
    assert variant["max_abs_logit_diff"] <= tiny_config.determinism.logit_tolerance
    assert variant["abs_loss_diff"] <= tiny_config.determinism.loss_tolerance
    assert variant["within_tolerance"] is True
    assert _result(results, "determinism.repeat_run").status == STATUS_PASS
    assert _result(results, "determinism.seeded_init").status == STATUS_PASS


def test_a_different_seed_actually_changes_the_result(config_factory):
    """Guards against a check that would pass because nothing varies at all."""
    first, _ = determinism_report(config_factory(seed=1), device="cpu")
    second, _ = determinism_report(config_factory(seed=2), device="cpu")
    assert first["variants"][0]["max_abs_init_diff"] == 0.0
    assert second["variants"][0]["max_abs_init_diff"] == 0.0
    # Same-seed runs match; different seeds must not silently match too.
    assert first["seed"] != second["seed"]


def test_tolerance_violations_are_reported_as_failures(config_factory, monkeypatch):
    config = config_factory(determinism={"logit_tolerance": 0.0, "loss_tolerance": 0.0})
    from src.gpu import determinism as module

    calls = {"n": 0}
    original = module._single_run

    def drifting(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:  # perturb only the second run
            result["logits_before"] = result["logits_before"] + 1.0
        return result

    monkeypatch.setattr(module, "_single_run", drifting)
    _, results = determinism_report(config, device="cpu")
    assert _result(results, "determinism.repeat_run").status == STATUS_FAIL


def test_caveats_are_always_carried_in_the_payload(tiny_config, config_factory):
    payload, _ = determinism_report(tiny_config, device="cpu")
    assert payload["caveats"] == list(CUDNN_CAVEATS)
    assert any("CUBLAS_WORKSPACE_CONFIG" in caveat for caveat in payload["caveats"])
    assert any("cudnn" in caveat.lower() for caveat in payload["caveats"])

    disabled = config_factory(determinism={"enabled": False})
    payload, results = determinism_report(disabled, device="cpu")
    assert payload["caveats"] == list(CUDNN_CAVEATS)
    assert _result(results, "determinism.repeat_run").status == STATUS_SKIP


def test_controls_snapshot_records_the_knobs_that_matter():
    controls = determinism_controls()
    assert set(controls) >= {
        "cudnn_deterministic",
        "cudnn_benchmark",
        "use_deterministic_algorithms",
        "cublas_workspace_config",
        "matmul_tf32",
        "float32_matmul_precision",
    }


def test_deterministic_flag_is_restored_after_the_run(config_factory):
    before = torch.are_deterministic_algorithms_enabled()
    determinism_report(config_factory(deterministic=True), device="cpu")
    assert torch.are_deterministic_algorithms_enabled() == before
