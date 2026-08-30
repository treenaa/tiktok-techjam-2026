"""Pass/fail accounting and the JSON + human-readable GPU report.

The report is the artefact other agents read before committing to a full
training run, so it is versioned, machine-readable, and written atomically.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

GPU_REPORT_SCHEMA_VERSION = 1

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"

#: Worst-first, so ``max(..., key=_SEVERITY.get)`` yields the overall status.
_SEVERITY = {STATUS_SKIP: 0, STATUS_PASS: 1, STATUS_WARN: 2, STATUS_FAIL: 3}


@dataclass(frozen=True)
class CheckResult:
    """One named verdict plus everything needed to act on it."""

    name: str
    status: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _SEVERITY:
            raise ValueError("check status must be one of %s" % sorted(_SEVERITY))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


def check(name: str, ok: bool, summary: str, **details: Any) -> CheckResult:
    """Shorthand for a boolean check."""
    return CheckResult(name, STATUS_PASS if ok else STATUS_FAIL, summary, details)


def overall_status(checks: Sequence[CheckResult], strict: bool = False) -> str:
    """Worst status across ``checks``; ``strict`` promotes warnings to failures."""
    if not checks:
        return STATUS_SKIP
    worst = max((result.status for result in checks), key=lambda status: _SEVERITY[status])
    if strict and worst == STATUS_WARN:
        return STATUS_FAIL
    return worst


@dataclass
class GpuReport:
    """Everything one ``gpu_check`` / ``benchmark_gpu`` invocation produced."""

    config: Dict[str, Any]
    environment: Dict[str, Any]
    checks: List[CheckResult] = field(default_factory=list)
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    determinism: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: int = GPU_REPORT_SCHEMA_VERSION

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    def status(self, strict: bool = False) -> str:
        return overall_status(self.checks, strict=strict)

    def by_status(self, status: str) -> List[CheckResult]:
        return [result for result in self.checks if result.status == status]

    def to_dict(self, strict: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "status": self.status(strict=strict),
            "strict": bool(strict),
            "duration_seconds": round(float(self.duration_seconds), 3),
            "environment": self.environment,
            "config": self.config,
            "parameters": self.parameters,
            "determinism": self.determinism,
            "recommendations": self.recommendations,
            "checks": [result.to_dict() for result in self.checks],
            "benchmarks": self.benchmarks,
            "notes": self.notes,
        }


def _format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return "%.2f %s" % (number, unit)
        number /= 1024.0
    return "%.2f TiB" % number  # pragma: no cover - unreachable, loop returns first


def _format_number(value: Any, spec: str = "%.1f") -> str:
    if value is None:
        return "n/a"
    try:
        return spec % float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


_MARKS = {
    STATUS_PASS: "PASS",
    STATUS_WARN: "WARN",
    STATUS_FAIL: "FAIL",
    STATUS_SKIP: "SKIP",
}


def render_text(report: GpuReport, strict: bool = False) -> str:
    """Render the terminal summary a human reads before starting a real run."""
    environment = report.environment or {}
    torch_info = environment.get("torch", {})
    devices = environment.get("devices", []) or []
    lines: List[str] = []
    lines.append("GPU readiness report (schema v%d)" % report.schema_version)
    lines.append("generated %s in %.1fs" % (report.created_at, report.duration_seconds))
    lines.append("")
    lines.append("Environment")
    lines.append(
        "  torch %s (built for CUDA %s) | cuda available: %s | device: %s"
        % (
            torch_info.get("version", "?"),
            torch_info.get("cuda_version") or "none (CPU-only build)",
            torch_info.get("cuda_available"),
            environment.get("resolved_device", "?"),
        )
    )
    lines.append(
        "  driver: %s | cudnn: %s | python %s on %s"
        % (
            environment.get("driver", {}).get("version") or "unknown",
            torch_info.get("cudnn_version") or "unknown",
            environment.get("python", {}).get("version", "?"),
            environment.get("platform", {}).get("system", "?"),
        )
    )
    for device in devices:
        lines.append(
            "  [%s] %s | %s VRAM | capability %s | %s SMs"
            % (
                device.get("index"),
                device.get("name"),
                _format_bytes(device.get("total_memory_bytes")),
                device.get("capability"),
                device.get("multi_processor_count", "?"),
            )
        )
    if not devices:
        lines.append("  no CUDA devices visible")
        adapters = (environment.get("display_adapters") or {}).get("adapters") or []
        for adapter in adapters:
            lines.append(
                "  display adapter: %s%s"
                % (
                    adapter.get("name"),
                    ""
                    if not adapter.get("driver_version")
                    else " (driver %s)" % adapter["driver_version"],
                )
            )

    if report.parameters:
        lines.append("")
        lines.append("Parameter budget (<%s)" % f"{report.parameters[0].get('limit', 0):,}")
        for row in report.parameters:
            lines.append(
                "  %-8s %-8s total %13s | trainable %13s | %s"
                % (
                    row.get("backbone"),
                    row.get("architecture"),
                    f"{row.get('total', 0):,}",
                    f"{row.get('trainable', 0):,}",
                    "ok" if row.get("within_limit") else "OVER LIMIT",
                )
            )

    if report.benchmarks:
        lines.append("")
        lines.append("Benchmarks")
        header = "  %-8s %-8s %-9s %-5s %-6s %10s %10s %9s %9s" % (
            "backbone",
            "arch",
            "mode",
            "prec",
            "batch",
            "img/s",
            "peak VRAM",
            "% budget",
            "first (s)",
        )
        lines.append(header)
        for row in report.benchmarks:
            if row.get("status") == "oom":
                lines.append(
                    "  %-8s %-8s %-9s %-5s %-6s %s"
                    % (
                        row.get("backbone"),
                        row.get("architecture"),
                        row.get("mode"),
                        row.get("precision"),
                        row.get("batch_size"),
                        "OOM -- config exceeds this device",
                    )
                )
                continue
            budget_fraction = row.get("budget_fraction")
            lines.append(
                "  %-8s %-8s %-9s %-5s %-6s %10s %10s %9s %9s"
                % (
                    row.get("backbone"),
                    row.get("architecture"),
                    row.get("mode"),
                    row.get("precision"),
                    row.get("batch_size"),
                    _format_number(row.get("images_per_second"), "%.1f"),
                    _format_bytes(row.get("peak_memory_reserved_bytes")),
                    (
                        "n/a"
                        if budget_fraction is None
                        else "%.0f%%%s" % (budget_fraction * 100, "" if row.get("fits_vram_budget") else " !")
                    ),
                    _format_number(row.get("time_to_first_batch_seconds"), "%.3f"),
                )
            )

    lines.append("")
    lines.append("Checks")
    for result in report.checks:
        lines.append("  [%s] %-38s %s" % (_MARKS[result.status], result.name, result.summary))

    if report.recommendations:
        budget = report.recommendations[0].get("vram_budget_bytes")
        lines.append("")
        lines.append(
            "Fallbacks for configurations that do not fit%s"
            % ("" if budget is None else " in %s" % _format_bytes(budget))
        )
        for item in report.recommendations:
            lines.append(
                "  %s/%s %s %s batch %s -- %s"
                % (
                    item.get("backbone"),
                    item.get("architecture"),
                    item.get("mode"),
                    item.get("precision"),
                    item.get("batch_size"),
                    "OOM"
                    if item.get("status") == "oom"
                    else "%s peak (%s of budget)"
                    % (
                        _format_bytes(item.get("peak_memory_reserved_bytes")),
                        _format_number(
                            (item.get("budget_fraction") or 0) * 100, "%.0f%%"
                        ),
                    ),
                )
            )
            for fallback in item.get("fallbacks", []):
                lines.append("      -> %s" % fallback)

    caveats = report.determinism.get("caveats") or []
    if caveats:
        lines.append("")
        lines.append("Reproducibility caveats (nondeterminism this run cannot rule out)")
        for caveat in caveats:
            lines.append("  - %s" % caveat)

    if report.notes:
        lines.append("")
        lines.append("Notes")
        for note in report.notes:
            lines.append("  - %s" % note)

    status = report.status(strict=strict)
    counts = {name: len(report.by_status(name)) for name in _MARKS}
    lines.append("")
    lines.append(
        "OVERALL: %s (%d pass, %d warn, %d fail, %d skip)%s"
        % (
            status.upper(),
            counts[STATUS_PASS],
            counts[STATUS_WARN],
            counts[STATUS_FAIL],
            counts[STATUS_SKIP],
            " [strict]" if strict else "",
        )
    )
    return "\n".join(lines)


def _write_atomic(text: str, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".gpu-report-", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_report(
    report: GpuReport,
    output_dir: str,
    *,
    basename: str = "gpu_report",
    strict: bool = False,
) -> Dict[str, str]:
    """Write ``<basename>.json`` and ``<basename>.txt``; return both paths."""
    json_path = os.path.join(output_dir, "%s.json" % basename)
    text_path = os.path.join(output_dir, "%s.txt" % basename)
    payload = json.dumps(report.to_dict(strict=strict), indent=2, sort_keys=True, default=str)
    _write_atomic(payload + "\n", json_path)
    _write_atomic(render_text(report, strict=strict) + "\n", text_path)
    return {"json": json_path, "text": text_path}


def read_report(path: str) -> Dict[str, Any]:
    """Read a written report back, for CI or another agent's gate."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("gpu report root must be a mapping, got %r" % type(payload).__name__)
    return dict(payload)


__all__ = [
    "GPU_REPORT_SCHEMA_VERSION",
    "STATUS_PASS",
    "STATUS_WARN",
    "STATUS_FAIL",
    "STATUS_SKIP",
    "CheckResult",
    "GpuReport",
    "check",
    "overall_status",
    "render_text",
    "write_report",
    "read_report",
]
