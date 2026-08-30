from __future__ import annotations

import copy

import pytest

from src.gpu import collect_environment, environment_checks
from src.gpu.config import config_from_mapping
from src.gpu.environment import parse_version, query_nvcc, query_nvidia_smi
from src.gpu.report import STATUS_FAIL, STATUS_PASS, STATUS_WARN

torch = pytest.importorskip("torch")


def _status(results, name):
    for result in results:
        if result.name == name:
            return result.status
    raise AssertionError("no check named %r in %s" % (name, [r.name for r in results]))


def test_collect_environment_reports_what_is_installed_without_assuming():
    environment = collect_environment("cpu")
    assert environment["torch"]["version"] == torch.__version__
    # torch.version.cuda is None on a CPU-only wheel; that is a fact to report,
    # not an error to raise.
    assert environment["torch"]["cuda_version"] == torch.version.cuda
    assert environment["torch"]["cuda_available"] == torch.cuda.is_available()
    assert environment["resolved_device"] == "cpu"
    assert set(environment) >= {
        "python",
        "platform",
        "devices",
        "display_adapters",
        "nvidia_smi",
        "nvcc",
        "env",
    }


def test_probe_helpers_degrade_gracefully_when_the_tool_is_absent():
    smi = query_nvidia_smi()
    assert isinstance(smi["available"], bool)
    if not smi["available"]:
        assert smi["reason"]
    nvcc = query_nvcc()
    assert isinstance(nvcc["available"], bool)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12.4", (12, 4)),
        ("release 11.8, V11.8.89", (11, 8)),  # stops at the first version token
        ("12.4.1", (12, 4, 1)),
        ("", None),
        (None, None),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def _fake_environment(**overrides):
    environment = {
        "python": {"version": "3.11.0", "executable": "python"},
        "platform": {"system": "Linux", "release": "6.2", "machine": "x86_64"},
        "torch": {
            "version": "2.4.0+cu121",
            "cuda_version": "12.1",
            "hip_version": None,
            "cuda_available": True,
            "device_count": 1,
            "cudnn_version": 8907,
            "cudnn_enabled": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "matmul_tf32": True,
            "cudnn_tf32": True,
            "float32_matmul_precision": "highest",
            "bf16_supported": True,
        },
        "driver": {"version": "535.104.05", "source": "torch"},
        "devices": [
            {
                "index": 0,
                "name": "Some GPU",
                "total_memory_bytes": 24 * 1024 ** 3,
                "capability": "8.6",
                "multi_processor_count": 84,
            }
        ],
        "nvidia_smi": {
            "available": True,
            "driver_version": "535.104.05",
            "driver_max_cuda_version": "12.2",
            "gpus": [],
        },
        "nvcc": {"available": True, "version": "12.1"},
        "display_adapters": {
            "available": True,
            "adapters": [{"name": "NVIDIA A10G", "driver_version": "535.104.05"}],
            "source": "lspci",
        },
        "requested_device": "auto",
        "resolved_device": "cuda",
        "device_resolution_error": None,
        "env": {"CUDA_VISIBLE_DEVICES": None},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(environment.get(key), dict):
            environment[key] = {**environment[key], **value}
        else:
            environment[key] = value
    return environment


def test_a_healthy_gpu_box_passes_every_environment_check():
    config = config_from_mapping({"require_cuda": True})
    results = environment_checks(_fake_environment(), config)
    assert _status(results, "torch.cuda_build") == STATUS_PASS
    assert _status(results, "torch.cuda_available") == STATUS_PASS
    assert _status(results, "device.resolved") == STATUS_PASS
    assert _status(results, "driver.supports_torch_cuda") == STATUS_PASS


def test_cpu_only_wheel_on_a_gpu_box_is_a_failure_not_a_warning():
    config = config_from_mapping({"require_cuda": True})
    environment = _fake_environment(
        torch={"cuda_version": None, "cuda_available": False, "device_count": 0},
        devices=[],
        resolved_device="cpu",
    )
    results = environment_checks(environment, config)
    assert _status(results, "torch.cuda_build") == STATUS_FAIL
    assert _status(results, "torch.cuda_available") == STATUS_FAIL
    assert _status(results, "device.resolved") == STATUS_FAIL
    assert _status(results, "device.inventory") == STATUS_FAIL


def test_a_driver_older_than_the_torch_cuda_runtime_fails():
    config = config_from_mapping({"require_cuda": True})
    environment = _fake_environment(
        torch={"cuda_version": "12.4"},
        nvidia_smi={"available": True, "driver_version": "470.1", "driver_max_cuda_version": "11.4"},
    )
    results = environment_checks(environment, config)
    assert _status(results, "driver.supports_torch_cuda") == STATUS_FAIL


def test_unknown_driver_ceiling_warns_rather_than_guessing():
    config = config_from_mapping({"require_cuda": True})
    environment = _fake_environment(nvidia_smi={"available": False, "reason": "not found"})
    results = environment_checks(environment, config)
    assert _status(results, "driver.supports_torch_cuda") == STATUS_WARN
    assert _status(results, "driver.version") == STATUS_PASS  # torch still knew it


def test_nvidia_smi_driver_string_wins_over_torchs_rounded_version():
    config = config_from_mapping({"require_cuda": True})
    environment = _fake_environment(driver={"version": "535.10", "source": "torch"})
    environment_checks(environment, config)
    assert environment["driver"] == {"version": "535.104.05", "source": "nvidia-smi"}


def test_environment_checks_do_not_mutate_the_caller_beyond_the_driver_field():
    config = config_from_mapping({"require_cuda": True})
    environment = _fake_environment()
    before = copy.deepcopy(environment)
    environment_checks(environment, config)
    before.pop("driver")
    environment.pop("driver")
    assert environment == before


def test_display_adapters_are_enumerated_on_this_machine():
    from src.gpu.environment import detect_display_adapters, has_nvidia_adapter

    adapters = detect_display_adapters()
    assert isinstance(adapters["available"], bool)
    if adapters["available"]:
        assert adapters["adapters"]
        assert all(adapter["name"] for adapter in adapters["adapters"])
        assert has_nvidia_adapter(adapters) in (True, False)
    else:
        # Unknown must stay unknown; it must never be reported as "no GPU".
        assert adapters["reason"]
        assert has_nvidia_adapter(adapters) is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA GeForce RTX 4090", True),
        ("Tesla T4", True),
        ("Quadro P1000", True),
        ("Intel(R) Iris(R) Xe Graphics", False),
        ("AMD Radeon RX 6800", False),
        ("Microsoft Basic Display Adapter", False),
    ],
)
def test_nvidia_adapters_are_recognised_by_name(name, expected):
    from src.gpu.environment import has_nvidia_adapter

    adapters = {"available": True, "adapters": [{"name": name}]}
    assert has_nvidia_adapter(adapters) is expected


def _no_cuda_environment(**adapters):
    return _fake_environment(
        torch={"cuda_version": None, "cuda_available": False, "device_count": 0},
        devices=[],
        resolved_device="cpu",
        display_adapters=adapters,
    )


def test_integrated_graphics_is_diagnosed_as_hardware_not_a_bad_wheel():
    """The dead end this check exists to prevent: no NVIDIA card at all."""
    config = config_from_mapping({"require_cuda": True})
    environment = _no_cuda_environment(
        available=True,
        adapters=[{"name": "Intel(R) Iris(R) Xe Graphics", "driver_version": "30.0.101.1109"}],
        source="Win32_VideoController",
    )
    results = environment_checks(environment, config)
    assert _status(results, "hardware.cuda_capable_gpu") == STATUS_FAIL

    hardware = next(r for r in results if r.name == "hardware.cuda_capable_gpu")
    assert "no NVIDIA adapter" in hardware.summary
    assert "Iris" in hardware.summary

    # And the build check must NOT send anyone after a CUDA wheel.
    build = next(r for r in results if r.name == "torch.cuda_build")
    assert "reinstall a CUDA wheel" not in build.summary
    assert "correct build for this machine" in build.summary


def test_an_nvidia_card_without_cuda_is_blamed_on_software():
    config = config_from_mapping({"require_cuda": True})
    environment = _no_cuda_environment(
        available=True,
        adapters=[{"name": "NVIDIA GeForce RTX 3070 Laptop GPU", "driver_version": "537.13"}],
        source="Win32_VideoController",
    )
    results = environment_checks(environment, config)
    assert _status(results, "hardware.cuda_capable_gpu") == STATUS_PASS
    hardware = next(r for r in results if r.name == "hardware.cuda_capable_gpu")
    assert "driver or the PyTorch build, not the hardware" in hardware.summary
    # Here the CUDA-wheel advice is correct and must still appear.
    build = next(r for r in results if r.name == "torch.cuda_build")
    assert "reinstall a CUDA wheel" in build.summary


def test_unknown_adapters_warn_rather_than_asserting_no_gpu():
    config = config_from_mapping({"require_cuda": True})
    environment = _no_cuda_environment(available=False, adapters=[], reason="lspci not found")
    results = environment_checks(environment, config)
    assert _status(results, "hardware.cuda_capable_gpu") == STATUS_WARN
    hardware = next(r for r in results if r.name == "hardware.cuda_capable_gpu")
    assert "cannot be attributed" in hardware.summary


def test_a_working_cuda_device_passes_the_hardware_check_regardless_of_adapters():
    config = config_from_mapping({"require_cuda": True})
    results = environment_checks(_fake_environment(), config)
    assert _status(results, "hardware.cuda_capable_gpu") == STATUS_PASS
