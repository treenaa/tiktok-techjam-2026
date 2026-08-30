"""Detect -- never assume -- the GPU environment, and judge it against config.

Nothing here hard-codes a GPU model, driver, or CUDA version. Facts are read
from PyTorch and, where available, from ``nvidia-smi`` and ``nvcc``; the checks
then compare those facts against what the run actually requires.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import GpuCheckConfig
from .report import STATUS_FAIL, STATUS_PASS, STATUS_SKIP, STATUS_WARN, CheckResult

#: Environment variables that silently change CUDA behaviour, so they belong in
#: any report someone might have to reproduce.
RELEVANT_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NVIDIA_VISIBLE_DEVICES",
    "TORCH_CUDNN_V8_API_ENABLED",
)

_SUBPROCESS_TIMEOUT_SECONDS = 20.0


def _run(command: List[str]) -> Optional[str]:
    """Run a probe command, returning ``None`` if it is missing or fails."""
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def parse_version(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """``"12.4.1"`` -> ``(12, 4, 1)``; unparseable input yields ``None``."""
    if not text:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(text))
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _driver_version_from_torch() -> Optional[str]:
    """PyTorch reports the driver as an int such as ``12040`` (= 12.4)."""
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if getter is None or not torch.cuda.is_available():
        return None
    try:
        raw = int(getter())
    except Exception:  # pragma: no cover - depends on the local driver stack
        return None
    if raw <= 0:
        return None
    return "%d.%d" % (raw // 1000, (raw % 1000) // 10)


def query_nvidia_smi() -> Dict[str, Any]:
    """Structured ``nvidia-smi`` facts, or ``{"available": False, ...}``."""
    csv = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if csv is None:
        return {"available": False, "reason": "nvidia-smi not found or returned an error"}
    gpus: List[Dict[str, Any]] = []
    driver_version: Optional[str] = None
    for line in csv.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        driver_version = parts[2]
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "total_memory_mib": float(parts[3]) if parts[3] else None,
                "used_memory_mib": float(parts[4]) if parts[4] else None,
            }
        )
    banner = _run(["nvidia-smi"]) or ""
    match = re.search(r"CUDA Version:\s*([\d.]+)", banner)
    return {
        "available": True,
        "driver_version": driver_version,
        # The driver's maximum supported CUDA runtime, not the installed toolkit.
        "driver_max_cuda_version": match.group(1) if match else None,
        "gpus": gpus,
    }


#: Matched against adapter names to decide whether CUDA is even possible here.
NVIDIA_ADAPTER_PATTERN = re.compile(
    r"nvidia|geforce|quadro|tesla|\brtx\b|\bgtx\b", re.IGNORECASE
)


def detect_display_adapters() -> Dict[str, Any]:
    """Enumerate the physical display adapters, independently of CUDA.

    Without this, "no CUDA device" is ambiguous between two situations that
    need opposite remedies: a CUDA-capable card whose PyTorch build or driver
    is wrong (fixable by reinstalling), and a machine with no NVIDIA hardware
    at all (no PyTorch build will ever help). Telling someone to reinstall a
    CUDA wheel on an integrated-graphics laptop wastes a large download and
    ends where it started.
    """
    system = platform.system()
    if system == "Windows":
        return _windows_adapters()
    if system == "Linux":
        return _linux_adapters()
    return {
        "available": False,
        "adapters": [],
        "reason": "adapter enumeration is not implemented for %s" % system,
    }


def _windows_adapters() -> Dict[str, Any]:
    output = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
        ]
    )
    if not output or not output.strip():
        return {"available": False, "adapters": [], "reason": "Win32_VideoController query failed"}
    try:
        payload = json.loads(output)
    except ValueError:
        return {"available": False, "adapters": [], "reason": "could not parse WMI JSON output"}
    # ConvertTo-Json emits a bare object when there is exactly one adapter.
    records = payload if isinstance(payload, list) else [payload]
    adapters = [
        {
            "name": str(record.get("Name") or "unknown"),
            "driver_version": record.get("DriverVersion"),
            "dedicated_memory_bytes": record.get("AdapterRAM"),
        }
        for record in records
        if isinstance(record, dict)
    ]
    return {"available": bool(adapters), "adapters": adapters, "source": "Win32_VideoController"}


def _linux_adapters() -> Dict[str, Any]:
    output = _run(["lspci"])
    if output is None:
        return {"available": False, "adapters": [], "reason": "lspci not found"}
    adapters = [
        {"name": line.split(":", 2)[-1].strip(), "driver_version": None, "dedicated_memory_bytes": None}
        for line in output.splitlines()
        if "VGA compatible controller" in line or "3D controller" in line
    ]
    return {"available": bool(adapters), "adapters": adapters, "source": "lspci"}


def has_nvidia_adapter(display_adapters: Dict[str, Any]) -> Optional[bool]:
    """``True``/``False`` when adapters were enumerated, ``None`` when unknown.

    ``None`` matters: "we could not tell" must never be reported as "you have
    no GPU".
    """
    if not display_adapters.get("available"):
        return None
    adapters = display_adapters.get("adapters") or []
    if not adapters:
        return None
    return any(NVIDIA_ADAPTER_PATTERN.search(adapter.get("name") or "") for adapter in adapters)


def query_nvcc() -> Dict[str, Any]:
    """CUDA toolkit version from ``nvcc``. Absent on runtime-only images."""
    output = _run(["nvcc", "--version"])
    if output is None:
        return {
            "available": False,
            "reason": "nvcc not found; a runtime-only CUDA image does not ship the toolkit",
        }
    match = re.search(r"release\s+([\d.]+)", output)
    return {"available": True, "version": match.group(1) if match else None, "raw": output.strip()}


def _device_records() -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    records = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        records.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "capability": "%d.%d" % (properties.major, properties.minor),
                "multi_processor_count": int(getattr(properties, "multi_processor_count", 0)),
            }
        )
    return records


def _bf16_supported() -> Optional[bool]:
    if not torch.cuda.is_available():
        return None
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - older torch on exotic devices
        return None


def resolve_device(requested: str = "auto") -> Tuple[str, Optional[str]]:
    """Resolve a device string, returning ``(device, error_or_None)``.

    Delegates to the evaluation subsystem's resolver so the whole repository
    agrees on what ``auto`` means, but reports a failure instead of raising --
    an unavailable device is a finding this report exists to state.
    """
    from src.evaluation import EvaluationError
    from src.evaluation import resolve_device as _resolve

    try:
        return _resolve(requested), None
    except EvaluationError as exc:
        return "cpu", str(exc)


def collect_environment(requested_device: str = "auto") -> Dict[str, Any]:
    """Gather every environment fact the checks and the report need."""
    resolved, resolution_error = resolve_device(requested_device)
    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover - CPU-only builds without cudnn
        cudnn_version = None
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": torch.__version__,
            # ``None`` here means a CPU-only wheel -- the classic silent failure
            # on a freshly rented GPU box.
            "cuda_version": torch.version.cuda,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_version": cudnn_version,
            "cudnn_enabled": bool(getattr(torch.backends.cudnn, "enabled", False)),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
            "matmul_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "cudnn_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_supported": _bf16_supported(),
        },
        "driver": {
            "version": _driver_version_from_torch(),
            "source": "torch",
        },
        "devices": _device_records(),
        "display_adapters": detect_display_adapters(),
        "nvidia_smi": query_nvidia_smi(),
        "nvcc": query_nvcc(),
        "requested_device": requested_device,
        "resolved_device": resolved,
        "device_resolution_error": resolution_error,
        "env": {name: os.environ.get(name) for name in RELEVANT_ENV_VARS},
    }


def _merge_driver_version(environment: Dict[str, Any]) -> Optional[str]:
    """Prefer ``nvidia-smi``'s driver string; fall back to PyTorch's."""
    smi = environment.get("nvidia_smi", {})
    if smi.get("available") and smi.get("driver_version"):
        environment["driver"] = {"version": smi["driver_version"], "source": "nvidia-smi"}
    return environment.get("driver", {}).get("version")


def environment_checks(
    environment: Dict[str, Any], config: GpuCheckConfig
) -> List[CheckResult]:
    """Judge the collected facts against what this run requires."""
    torch_info = environment["torch"]
    require_cuda = bool(config.require_cuda)
    results: List[CheckResult] = []
    driver_version = _merge_driver_version(environment)

    adapters = environment.get("display_adapters", {"available": False, "adapters": []})
    nvidia_present = has_nvidia_adapter(adapters)
    results.append(_hardware_check(adapters, nvidia_present, torch_info, require_cuda))

    built_for_cuda = torch_info["cuda_version"] is not None
    if built_for_cuda:
        build_summary = "torch %s built for CUDA %s" % (
            torch_info["version"],
            torch_info["cuda_version"],
        )
    elif nvidia_present is False:
        # Recommending a CUDA wheel here would send someone after a download
        # that cannot possibly help.
        build_summary = (
            "torch %s is a CPU-only build, which is the correct build for this machine: "
            "no NVIDIA adapter is present, so a CUDA wheel would not run either"
            % torch_info["version"]
        )
    else:
        build_summary = (
            "torch %s is a CPU-only build (torch.version.cuda is None); reinstall a CUDA wheel"
            % torch_info["version"]
        )
    results.append(
        CheckResult(
            "torch.cuda_build",
            STATUS_PASS if built_for_cuda else (STATUS_FAIL if require_cuda else STATUS_WARN),
            build_summary,
            {
                "torch_version": torch_info["version"],
                "cuda_version": torch_info["cuda_version"],
                "nvidia_adapter_present": nvidia_present,
            },
        )
    )

    available = torch_info["cuda_available"]
    results.append(
        CheckResult(
            "torch.cuda_available",
            STATUS_PASS if available else (STATUS_FAIL if require_cuda else STATUS_WARN),
            (
                "torch.cuda.is_available() is True with %d device(s)" % torch_info["device_count"]
                if available
                else "torch.cuda.is_available() is False"
            ),
            {"device_count": torch_info["device_count"]},
        )
    )

    resolved = environment["resolved_device"]
    resolution_error = environment.get("device_resolution_error")
    device_ok = resolution_error is None and (not require_cuda or resolved.startswith("cuda"))
    results.append(
        CheckResult(
            "device.resolved",
            STATUS_PASS if device_ok else STATUS_FAIL,
            (
                "device %r could not be resolved: %s"
                % (environment["requested_device"], resolution_error)
                if resolution_error is not None
                else "requested %r resolved to %r" % (environment["requested_device"], resolved)
                if device_ok
                else "requested %r resolved to %r, but this run requires CUDA; nothing below "
                "would have validated a GPU" % (environment["requested_device"], resolved)
            ),
            {
                "requested": environment["requested_device"],
                "resolved": resolved,
                "require_cuda": require_cuda,
                "error": resolution_error,
            },
        )
    )

    devices = environment["devices"]
    if devices:
        results.append(
            CheckResult(
                "device.inventory",
                STATUS_PASS,
                "; ".join(
                    "%s (%.1f GiB, sm_%s)"
                    % (
                        device["name"],
                        device["total_memory_bytes"] / (1024 ** 3),
                        device["capability"].replace(".", ""),
                    )
                    for device in devices
                ),
                {"devices": devices},
            )
        )
    else:
        results.append(
            CheckResult(
                "device.inventory",
                STATUS_FAIL if require_cuda else STATUS_SKIP,
                "no CUDA devices visible to this process",
                {"cuda_visible_devices": environment["env"].get("CUDA_VISIBLE_DEVICES")},
            )
        )

    if available:
        results.append(
            CheckResult(
                "driver.version",
                STATUS_PASS if driver_version else STATUS_WARN,
                (
                    "NVIDIA driver %s (via %s)"
                    % (driver_version, environment["driver"].get("source"))
                    if driver_version
                    else "driver version could not be read from torch or nvidia-smi"
                ),
                dict(environment["driver"]),
            )
        )
        results.append(_driver_runtime_check(environment))
    else:
        results.append(
            CheckResult("driver.version", STATUS_SKIP, "no CUDA runtime to query a driver from", {})
        )
        results.append(
            CheckResult(
                "driver.supports_torch_cuda",
                STATUS_SKIP,
                "skipped: CUDA is unavailable",
                {},
            )
        )

    results.append(
        CheckResult(
            "toolkit.nvcc",
            STATUS_PASS if environment["nvcc"].get("available") else STATUS_WARN,
            (
                "CUDA toolkit (nvcc) %s" % environment["nvcc"].get("version")
                if environment["nvcc"].get("available")
                else "%s -- PyTorch ships its own runtime, so this is informational"
                % environment["nvcc"].get("reason")
            ),
            dict(environment["nvcc"]),
        )
    )

    results.append(
        CheckResult(
            "precision.support",
            STATUS_PASS,
            "bf16 supported: %s | TF32 matmul: %s | float32 matmul precision: %s"
            % (
                torch_info["bf16_supported"],
                torch_info["matmul_tf32"],
                torch_info["float32_matmul_precision"],
            ),
            {
                "bf16_supported": torch_info["bf16_supported"],
                "matmul_tf32": torch_info["matmul_tf32"],
                "cudnn_tf32": torch_info["cudnn_tf32"],
                "float32_matmul_precision": torch_info["float32_matmul_precision"],
            },
        )
    )
    return results


def _hardware_check(
    adapters: Dict[str, Any],
    nvidia_present: Optional[bool],
    torch_info: Dict[str, Any],
    require_cuda: bool,
) -> CheckResult:
    """Is CUDA even possible on this machine, regardless of software?

    This runs before the PyTorch-build check so the report answers "can this
    box ever run CUDA?" before it answers "is it configured to?".
    """
    names = [adapter.get("name") for adapter in adapters.get("adapters") or []]
    details = {
        "adapters": adapters.get("adapters"),
        "source": adapters.get("source"),
        "nvidia_adapter_present": nvidia_present,
    }
    if torch_info["cuda_available"]:
        return CheckResult(
            "hardware.cuda_capable_gpu",
            STATUS_PASS,
            "CUDA is initialised, so a capable device is present",
            details,
        )
    if nvidia_present is None:
        return CheckResult(
            "hardware.cuda_capable_gpu",
            STATUS_WARN,
            "could not enumerate display adapters (%s), so 'no CUDA' cannot be attributed to "
            "hardware or to software here" % adapters.get("reason", "unknown reason"),
            details,
        )
    if nvidia_present:
        return CheckResult(
            "hardware.cuda_capable_gpu",
            STATUS_PASS,
            "NVIDIA adapter present (%s) but CUDA is not initialised; the fault is in the "
            "driver or the PyTorch build, not the hardware" % ", ".join(names),
            details,
        )
    return CheckResult(
        "hardware.cuda_capable_gpu",
        STATUS_FAIL if require_cuda else STATUS_WARN,
        "no NVIDIA adapter on this machine (found: %s). CUDA cannot be enabled by any driver "
        "or PyTorch build -- this run needs a different machine, a cloud GPU, or CPU-only mode"
        % (", ".join(names) or "none"),
        details,
    )


def _driver_runtime_check(environment: Dict[str, Any]) -> CheckResult:
    """Is the installed driver new enough for the CUDA runtime torch was built against?"""
    runtime = environment["torch"]["cuda_version"]
    smi = environment.get("nvidia_smi", {})
    driver_max = smi.get("driver_max_cuda_version") if smi.get("available") else None
    runtime_version = parse_version(runtime)
    driver_max_version = parse_version(driver_max)
    details = {
        "torch_cuda_runtime": runtime,
        "driver_max_cuda_version": driver_max,
        "driver_version": environment.get("driver", {}).get("version"),
    }
    if runtime_version is None:
        return CheckResult(
            "driver.supports_torch_cuda",
            STATUS_FAIL,
            "torch has no CUDA runtime version to compare against the driver",
            details,
        )
    if driver_max_version is None:
        return CheckResult(
            "driver.supports_torch_cuda",
            STATUS_WARN,
            "driver's maximum CUDA version is unknown (nvidia-smi unavailable); "
            "CUDA initialised, so the pairing is probably fine",
            details,
        )
    if driver_max_version[:2] < runtime_version[:2]:
        return CheckResult(
            "driver.supports_torch_cuda",
            STATUS_FAIL,
            "driver supports CUDA up to %s but torch was built for %s" % (driver_max, runtime),
            details,
        )
    return CheckResult(
        "driver.supports_torch_cuda",
        STATUS_PASS,
        "driver supports CUDA up to %s; torch runtime is %s" % (driver_max, runtime),
        details,
    )


__all__ = [
    "RELEVANT_ENV_VARS",
    "NVIDIA_ADAPTER_PATTERN",
    "collect_environment",
    "detect_display_adapters",
    "has_nvidia_adapter",
    "environment_checks",
    "parse_version",
    "query_nvcc",
    "query_nvidia_smi",
    "resolve_device",
]
