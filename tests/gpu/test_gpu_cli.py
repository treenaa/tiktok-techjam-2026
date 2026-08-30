from __future__ import annotations

import json
import os

import pytest

from scripts import benchmark_gpu, gpu_check

torch = pytest.importorskip("torch")


def test_gpu_check_runs_end_to_end_and_writes_both_reports(tmp_path, capsys):
    exit_code = gpu_check.main(
        [
            "--config",
            "configs/gpu_check_smoke.yaml",
            "--allow-cpu",
            "--backbones",
            "dinov2",
            "--architectures",
            "visual",
            "--skip-benchmarks",
            "--skip-determinism",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    text = capsys.readouterr().out
    assert "GPU readiness report" in text
    assert "smoke.fp32.dinov2/visual" in text
    assert os.path.exists(tmp_path / "gpu_report.json")
    payload = json.loads((tmp_path / "gpu_report.json").read_text(encoding="utf-8"))
    assert payload["config"]["model"]["backbones"] == ["dinov2"]


def test_json_output_is_machine_readable(tmp_path, capsys):
    exit_code = gpu_check.main(
        [
            "--config",
            "configs/gpu_check_smoke.yaml",
            "--allow-cpu",
            "--backbones",
            "clip",
            "--architectures",
            "visual",
            "--skip-benchmarks",
            "--skip-determinism",
            "--no-report",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] >= 1
    assert payload["status"] in {"pass", "warn", "fail", "skip"}
    assert any(check["name"].startswith("placement.") for check in payload["checks"])


def test_a_bad_config_exits_with_the_usage_code(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nonsense": True}), encoding="utf-8")
    assert gpu_check.main(["--config", str(bad), "--no-report"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_a_missing_config_exits_with_the_usage_code(capsys):
    assert gpu_check.main(["--config", "configs/nope.yaml", "--no-report"]) == 2


def test_requiring_cuda_without_cuda_exits_nonzero(capsys):
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    exit_code = gpu_check.main(
        [
            "--config",
            "configs/gpu_check_smoke.yaml",
            "--require-cuda",
            "--no-report",
            "--quiet",
        ]
    )
    assert exit_code == 1


def test_strict_turns_warnings_into_a_nonzero_exit(capsys):
    if torch.cuda.is_available():
        pytest.skip("warnings differ once a real GPU is present")
    # A CPU-only torch build warns; --strict must make that blocking.
    assert (
        gpu_check.main(
            [
                "--config",
                "configs/gpu_check_smoke.yaml",
                "--allow-cpu",
                "--backbones",
                "dinov2",
                "--architectures",
                "visual",
                "--skip-benchmarks",
                "--skip-determinism",
                "--no-report",
                "--quiet",
                "--strict",
            ]
        )
        == 1
    )


def test_cli_overrides_reach_the_config():
    parser = gpu_check.build_parser()
    args = parser.parse_args(
        ["--device", "cuda:1", "--seed", "7", "--backbone-source", "stub", "--allow-cpu"]
    )
    merged = gpu_check.apply_overrides({"device": "auto", "seed": 42}, args)
    assert merged["device"] == "cuda:1"
    assert merged["seed"] == 7
    assert merged["allow_cpu"] is True
    assert merged["require_cuda"] is False
    assert merged["model"]["backbone_source"] == "stub"


def test_benchmark_overrides_reach_the_benchmark_section():
    args = benchmark_gpu.build_benchmark_parser().parse_args(
        ["--batch-sizes", "4", "8", "--modes", "train", "--measure-steps", "3", "--image-size", "64"]
    )
    merged = benchmark_gpu.benchmark_overrides({}, args)
    assert merged["benchmark"]["batch_sizes"] == [4, 8]
    assert merged["benchmark"]["modes"] == ["train"]
    assert merged["benchmark"]["measure_steps"] == 3
    assert merged["model"]["image_size"] == 64
    assert args.basename == "benchmark"


def test_benchmark_cli_writes_a_benchmark_report(tmp_path):
    exit_code = benchmark_gpu.main(
        [
            "--config",
            "configs/gpu_check_smoke.yaml",
            "--allow-cpu",
            "--backbones",
            "dinov2",
            "--architectures",
            "visual",
            "--batch-sizes",
            "2",
            "--modes",
            "inference",
            "--measure-steps",
            "2",
            "--warmup-steps",
            "0",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert exit_code == 0
    payload = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["benchmarks"]
    assert payload["benchmarks"][0]["mode"] == "inference"
    assert payload["benchmarks"][0]["batch_size"] == 2


def test_deterministic_flag_sets_the_cublas_workspace_before_torch_initialises(monkeypatch):
    # setenv first so monkeypatch records (and restores) the original state,
    # whether or not the variable was already exported.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "sentinel")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    notes = gpu_check.prepare_environment(deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert any("CUBLAS_WORKSPACE_CONFIG" in note for note in notes)
    assert gpu_check.prepare_environment(deterministic=False) == []
