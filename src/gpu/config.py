"""Configuration for the GPU validation and benchmark subsystem.

Everything the checks are allowed to assume -- devices, backbones, batch
sizes, tolerances and compute budgets -- comes from here, so a run on a
different rented instance is a config edit rather than a code edit. Nothing in
this package hard-codes a GPU model, driver, or CUDA version.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class GpuConfigError(ValueError):
    """Raised when a GPU-check configuration would produce a meaningless run."""


PARAMETER_LIMIT = 2_000_000_000

BACKBONE_SOURCES = ("stub", "pretrained")
ARCHITECTURES = ("visual", "fusion")
PRECISIONS = ("fp32", "amp")
MODES = ("train", "inference")
AMP_DTYPES = ("float16", "bfloat16")


def _positive_ints(values: Sequence[Any], name: str) -> Tuple[int, ...]:
    if not values:
        raise GpuConfigError("%s must list at least one value" % name)
    numbers = []
    for value in values:
        number = int(value)
        if number < 1:
            raise GpuConfigError("%s entries must be positive, got %r" % (name, value))
        numbers.append(number)
    return tuple(numbers)


def _members(values: Sequence[Any], allowed: Sequence[str], name: str) -> Tuple[str, ...]:
    if not values:
        raise GpuConfigError("%s must list at least one value" % name)
    chosen: list = []
    for value in values:
        text = str(value).strip().lower()
        if text not in allowed:
            raise GpuConfigError(
                "%s entries must be one of %s, got %r" % (name, tuple(allowed), value)
            )
        if text not in chosen:
            chosen.append(text)
    return tuple(chosen)


def _check_optional_positive(value: Any, name: str) -> None:
    if value is None:
        return
    if float(value) <= 0:
        raise GpuConfigError("%s must be positive or null" % name)


@dataclass(frozen=True)
class StubConfig:
    """Shape of the download-free stand-in backbone.

    The stub is a small patch-embedding transformer. It exists so this
    subsystem can exercise the real detector, loss and optimizer code on the
    GPU without pulling pretrained weights; it says nothing about accuracy.
    """

    hidden_size: int = 192
    layers: int = 2
    heads: int = 3
    patch_size: int = 16

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.layers, self.heads, self.patch_size) < 1:
            raise GpuConfigError("stub hidden_size/layers/heads/patch_size must be positive")
        if self.hidden_size % self.heads:
            raise GpuConfigError(
                "stub hidden_size %d must be divisible by heads %d"
                % (self.hidden_size, self.heads)
            )


@dataclass(frozen=True)
class ModelConfig:
    """Which detectors to exercise, and how to construct them."""

    backbones: Tuple[str, ...] = ("ijepa", "dinov2", "clip")
    architectures: Tuple[str, ...] = ("visual", "fusion")
    backbone_source: str = "stub"
    freeze_backbone: bool = True
    image_size: int = 224
    head_hidden_dim: Optional[int] = 256
    local_files_only: bool = False
    stub: StubConfig = field(default_factory=StubConfig)

    def __post_init__(self) -> None:
        if not self.backbones:
            raise GpuConfigError("model.backbones must list at least one backbone")
        if self.backbone_source not in BACKBONE_SOURCES:
            raise GpuConfigError("model.backbone_source must be one of %s" % (BACKBONE_SOURCES,))
        if self.image_size < 8:
            raise GpuConfigError("model.image_size must be at least 8")
        if self.backbone_source == "stub" and self.image_size % self.stub.patch_size:
            raise GpuConfigError(
                "model.image_size %d must be divisible by the stub patch_size %d"
                % (self.image_size, self.stub.patch_size)
            )

    @property
    def variants(self) -> Tuple[Tuple[str, str], ...]:
        """Every ``(backbone, architecture)`` pair the checks should cover."""
        return tuple(
            (backbone, architecture)
            for backbone in self.backbones
            for architecture in self.architectures
        )


@dataclass(frozen=True)
class SmokeConfig:
    """Tiny forward/backward settings used to prove the pipeline runs at all."""

    batch_size: int = 4
    steps: int = 2
    paired: bool = True
    amp: bool = True
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.steps < 1:
            raise GpuConfigError("smoke.batch_size and smoke.steps must be positive")
        if self.amp_dtype not in AMP_DTYPES:
            raise GpuConfigError("smoke.amp_dtype must be one of %s" % (AMP_DTYPES,))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Throughput and memory sweep settings."""

    batch_sizes: Tuple[int, ...] = (8, 16, 32)
    warmup_steps: int = 2
    measure_steps: int = 8
    precisions: Tuple[str, ...] = ("fp32", "amp")
    modes: Tuple[str, ...] = ("train", "inference")
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise GpuConfigError("benchmark.warmup_steps cannot be negative")
        if self.measure_steps < 1:
            raise GpuConfigError("benchmark.measure_steps must be positive")
        if not self.batch_sizes:
            raise GpuConfigError("benchmark.batch_sizes must list at least one value")
        if self.amp_dtype not in AMP_DTYPES:
            raise GpuConfigError("benchmark.amp_dtype must be one of %s" % (AMP_DTYPES,))


@dataclass(frozen=True)
class DeterminismConfig:
    """Same-seed repeat-run tolerances.

    Bitwise equality is deliberately *not* asserted: cuDNN algorithm selection,
    TF32 matmuls and atomic reductions can each move the last few bits. The
    check measures the divergence, compares it against these tolerances, and
    records which nondeterminism controls were actually active.
    """

    enabled: bool = True
    batch_size: int = 4
    logit_tolerance: float = 1e-5
    loss_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise GpuConfigError("determinism.batch_size must be positive")
        if self.logit_tolerance < 0 or self.loss_tolerance < 0:
            raise GpuConfigError("determinism tolerances cannot be negative")


@dataclass(frozen=True)
class BudgetConfig:
    """Pass/fail thresholds for hackathon-scale compute."""

    parameter_limit: int = PARAMETER_LIMIT
    #: VRAM the run must plan for, in MiB. ``None`` means "whatever this device
    #: reports". Set it explicitly (e.g. 8172 for an 8 GB card) so a sweep can
    #: flag configurations that would not fit the *target* GPU even when it is
    #: measured somewhere else -- the alternative is quietly assuming a bigger
    #: card than the team actually has.
    vram_budget_mb: Optional[float] = None
    vram_headroom_fraction: float = 0.9
    #: Batch size training is expected to use; fallbacks are sized to preserve
    #: it via gradient accumulation when the physical batch has to shrink.
    target_train_batch_size: Optional[int] = None
    min_train_images_per_second: Optional[float] = None
    min_inference_images_per_second: Optional[float] = None
    max_time_to_first_batch_seconds: Optional[float] = 60.0
    max_total_seconds: Optional[float] = 900.0
    amp_logit_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.parameter_limit < 1:
            raise GpuConfigError("budget.parameter_limit must be positive")
        if not 0 < self.vram_headroom_fraction <= 1:
            raise GpuConfigError("budget.vram_headroom_fraction must be in (0, 1]")
        if self.amp_logit_tolerance < 0:
            raise GpuConfigError("budget.amp_logit_tolerance cannot be negative")
        if self.target_train_batch_size is not None and self.target_train_batch_size < 1:
            raise GpuConfigError("budget.target_train_batch_size must be positive or null")
        for name in (
            "vram_budget_mb",
            "min_train_images_per_second",
            "min_inference_images_per_second",
            "max_time_to_first_batch_seconds",
            "max_total_seconds",
        ):
            _check_optional_positive(getattr(self, name), "budget.%s" % name)

    @property
    def vram_budget_bytes(self) -> Optional[int]:
        if self.vram_budget_mb is None:
            return None
        return int(self.vram_budget_mb * 1024 * 1024)


@dataclass(frozen=True)
class GpuCheckConfig:
    """Complete GPU-validation configuration."""

    device: str = "auto"
    require_cuda: bool = True
    allow_cpu: bool = False
    seed: int = 42
    deterministic: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self) -> None:
        if not str(self.device).strip():
            raise GpuConfigError("device must be a non-empty string such as 'auto' or 'cuda:0'")
        if self.require_cuda and self.allow_cpu:
            raise GpuConfigError(
                "require_cuda and allow_cpu contradict each other; set require_cuda=false "
                "for a CPU dry run"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "smoke": SmokeConfig,
    "benchmark": BenchmarkConfig,
    "determinism": DeterminismConfig,
    "budget": BudgetConfig,
}


def _build_stub(raw: Any) -> StubConfig:
    if raw is None:
        return StubConfig()
    if not isinstance(raw, Mapping):
        raise GpuConfigError("config section 'model.stub' must be a mapping")
    unknown = set(raw) - set(StubConfig.__dataclass_fields__)
    if unknown:
        raise GpuConfigError("unknown model.stub key(s) %s" % sorted(unknown))
    return StubConfig(**{key: int(value) for key, value in raw.items()})


def _build_section(name: str, raw: Any) -> Any:
    cls = _SECTIONS[name]
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise GpuConfigError("config section %r must be a mapping" % name)
    values = dict(raw)
    if name == "model":
        values["stub"] = _build_stub(values.pop("stub", None))
        if values.get("backbones") is not None:
            values["backbones"] = tuple(str(item).strip().lower() for item in values["backbones"])
        if values.get("architectures") is not None:
            values["architectures"] = _members(
                values["architectures"], ARCHITECTURES, "model.architectures"
            )
        if values.get("backbone_source") is not None:
            values["backbone_source"] = str(values["backbone_source"]).strip().lower()
    if name == "benchmark":
        if values.get("batch_sizes") is not None:
            values["batch_sizes"] = _positive_ints(values["batch_sizes"], "benchmark.batch_sizes")
        if values.get("precisions") is not None:
            values["precisions"] = _members(values["precisions"], PRECISIONS, "benchmark.precisions")
        if values.get("modes") is not None:
            values["modes"] = _members(values["modes"], MODES, "benchmark.modes")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise GpuConfigError(
            "unknown %s config key(s) %s; allowed: %s"
            % (name, sorted(unknown), sorted(cls.__dataclass_fields__))
        )
    try:
        return cls(**values)
    except GpuConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise GpuConfigError("invalid %s config: %s" % (name, exc)) from exc


def config_from_mapping(values: Optional[Mapping[str, Any]] = None) -> GpuCheckConfig:
    """Build a validated config from a plain mapping, rejecting unknown keys."""
    raw = dict(values or {})
    sections = {name: _build_section(name, raw.pop(name, None)) for name in _SECTIONS}
    allowed = set(GpuCheckConfig.__dataclass_fields__) - set(_SECTIONS)
    unknown = set(raw) - allowed
    if unknown:
        raise GpuConfigError(
            "unknown config key(s) %s; allowed: %s"
            % (sorted(unknown), sorted(allowed | set(_SECTIONS)))
        )
    try:
        return GpuCheckConfig(**raw, **sections)
    except GpuConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise GpuConfigError("invalid gpu config: %s" % exc) from exc


def load_gpu_config(path: Optional[str] = None) -> GpuCheckConfig:
    """Load a JSON or YAML GPU-check config; ``None`` yields the defaults."""
    if path is None:
        return config_from_mapping({})
    if not os.path.exists(path):
        raise GpuConfigError("gpu config not found: %s" % path)
    text = open(path, encoding="utf-8").read()
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:  # pragma: no cover - pyyaml is a hard dependency
            raise GpuConfigError(
                "%s is YAML but pyyaml is not installed; install pyyaml or use JSON" % path
            ) from None
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise GpuConfigError("gpu config root must be a mapping, got %r" % type(raw).__name__)
    return config_from_mapping(raw)


__all__ = [
    "GpuConfigError",
    "PARAMETER_LIMIT",
    "BACKBONE_SOURCES",
    "ARCHITECTURES",
    "PRECISIONS",
    "MODES",
    "AMP_DTYPES",
    "StubConfig",
    "ModelConfig",
    "SmokeConfig",
    "BenchmarkConfig",
    "DeterminismConfig",
    "BudgetConfig",
    "GpuCheckConfig",
    "config_from_mapping",
    "load_gpu_config",
]
