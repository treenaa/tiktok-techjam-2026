"""Shared experiment configuration.

Rule 21 lists *incomparable backbone benchmarks* as a failure mode: if CLIP,
DINO and I-JEPA are each trained on a different split, head or schedule, the
comparison is meaningless and the architecture decision rests on noise.

This module defines one config object covering the whole run, and a
:func:`comparability_report` that says -- mechanically -- whether two runs may
be compared.  The data layer owns it because the split is the part that must be
identical, and the split is mine.

Sections
--------
``data``
    Consumed by :mod:`src.data.config` (datasets, ratios, split policy).
``model`` / ``training`` / ``evaluation``
    Passed through untouched to the owning subsystem.  This module validates
    *structure* and records what varies between runs; it does not interpret the
    values, so Mateo and Trina can add keys without changing this file.

The rule enforced here: when comparing runs, only ``model`` may differ.
"""

from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from .config import DatasetConfigError, build_from_config

__all__ = [
    "ExperimentConfig",
    "load_experiment",
    "save_experiment",
    "comparability_report",
    "assert_comparable",
    "COMPARABILITY_KEYS",
    "DEFAULT_EXPERIMENT",
]

#: Sections that MUST match for two runs to be comparable.  ``model`` is
#: deliberately absent -- that is the thing under test.
COMPARABILITY_KEYS: Tuple[str, ...] = ("data", "training", "evaluation", "seed")

DEFAULT_EXPERIMENT: Dict[str, Any] = {
    "name": "unnamed",
    "seed": 0,
    "data": dict(DEFAULT_DATA_CONFIG),
    "model": {},
    "training": {},
    "evaluation": {},
}

_SECTIONS = ("name", "seed", "data", "model", "training", "evaluation")


class ExperimentConfig(object):
    """A whole-run configuration: data + model + training + evaluation.

    Attribute access mirrors the sections::

        config = load_experiment("configs/baseline_clip.yaml")
        config.seed, config.model, config.data
        splits = config.build_splits()
    """

    def __init__(self, raw: Mapping[str, Any], path: Optional[str] = None):
        if not isinstance(raw, Mapping):
            raise DatasetConfigError(
                "experiment config must be a mapping, got %r" % type(raw).__name__
            )
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise DatasetConfigError(
                "unknown experiment section(s) %s; allowed: %s"
                % (sorted(unknown), list(_SECTIONS))
            )
        merged = copy.deepcopy(DEFAULT_EXPERIMENT)
        for key, value in raw.items():
            if key in ("data", "model", "training", "evaluation"):
                if not isinstance(value, Mapping):
                    raise DatasetConfigError(
                        "section %r must be a mapping, got %r" % (key, type(value).__name__)
                    )
                section = dict(merged[key])
                section.update(value)
                merged[key] = section
            else:
                merged[key] = value

        # One seed governs the whole run; the data section inherits it unless it
        # deliberately overrides, so a split cannot silently differ from the
        # training seed.
        if "seed" not in raw.get("data", {}):
            merged["data"]["seed"] = merged["seed"]

        if not merged["data"].get("datasets"):
            raise DatasetConfigError("experiment config has no data.datasets")

        self.raw = merged
        self.path = path

    # -- section access ---------------------------------------------------
    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw["model"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw["evaluation"]

    def build_splits(self, validate: bool = True):
        """Build the leakage-checked splits this experiment trains on."""
        return build_from_config(self.data, validate=validate)

    def fingerprint(self, keys: Sequence[str] = COMPARABILITY_KEYS) -> str:
        """Stable hash of the sections that must match across compared runs.

        Two runs with the same fingerprint saw the same data, split, schedule and
        seed -- so a difference in their metrics is attributable to ``model``.
        """
        import hashlib

        payload = {k: self.raw[k] for k in keys}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def __repr__(self) -> str:
        return "ExperimentConfig(name=%r, seed=%d, fingerprint=%s)" % (
            self.name,
            self.seed,
            self.fingerprint(),
        )


def load_experiment(path: str) -> ExperimentConfig:
    """Load a YAML or JSON experiment config."""
    if not os.path.exists(path):
        raise DatasetConfigError("experiment config not found: %s" % path)
    text = open(path, encoding="utf-8").read()
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise DatasetConfigError(
                "%s is YAML but pyyaml is not installed; install pyyaml or use JSON" % path
            ) from None
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return ExperimentConfig(raw, path=path)


def save_experiment(config: ExperimentConfig, path: str) -> str:
    """Write a config back out (JSON, or YAML when the extension asks for it).

    Persist the resolved config next to a checkpoint so a run can be
    reconstructed exactly.
    """
    payload = config.to_dict()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    return path


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------
def _differences(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Recursive diff of two config sections, as dotted paths."""
    out: List[str] = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b)):
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            if key not in a:
                out.append("%s: missing in first, %r in second" % (path, b[key]))
            elif key not in b:
                out.append("%s: %r in first, missing in second" % (path, a[key]))
            else:
                out.extend(_differences(a[key], b[key], path))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if list(a) != list(b):
            out.append("%s: %r vs %r" % (prefix, list(a), list(b)))
    elif a != b:
        out.append("%s: %r vs %r" % (prefix, a, b))
    return out


def comparability_report(
    configs: Sequence[ExperimentConfig],
    keys: Sequence[str] = COMPARABILITY_KEYS,
) -> Dict[str, Any]:
    """Are these runs comparable?  Returns the verdict and every difference.

    Comparable means: identical data, split, seed, training schedule and
    evaluation protocol, differing only in ``model``.  Anything else and a
    metric gap cannot be attributed to the backbone.
    """
    configs = list(configs)
    if len(configs) < 2:
        raise ValueError("need at least two configs to compare")

    reference = configs[0]
    fingerprints = OrderedDict((c.name, c.fingerprint(keys)) for c in configs)
    problems: "OrderedDict[str, List[str]]" = OrderedDict()
    for other in configs[1:]:
        diffs: List[str] = []
        for key in keys:
            diffs.extend(_differences(reference.raw[key], other.raw[key], key))
        if diffs:
            problems["%s vs %s" % (reference.name, other.name)] = diffs

    models = OrderedDict((c.name, c.model) for c in configs)
    identical_models = [
        (a.name, b.name)
        for i, a in enumerate(configs)
        for b in configs[i + 1:]
        if a.model == b.model
    ]
    return {
        "comparable": not problems,
        "fingerprints": dict(fingerprints),
        "differences": {k: v for k, v in problems.items()},
        "models": dict(models),
        "identical_models": identical_models,
    }


def assert_comparable(
    configs: Sequence[ExperimentConfig],
    keys: Sequence[str] = COMPARABILITY_KEYS,
    require_distinct_models: bool = True,
) -> Dict[str, Any]:
    """Raise unless the runs differ *only* in their model section.

    Use before publishing a backbone comparison table.
    """
    report = comparability_report(configs, keys=keys)
    if not report["comparable"]:
        lines = ["these runs are NOT comparable -- a metric gap cannot be attributed "
                 "to the model (rule 21: incomparable backbone benchmarks):"]
        for pair, diffs in report["differences"].items():
            lines.append("  %s" % pair)
            lines.extend("    - %s" % d for d in diffs[:10])
            if len(diffs) > 10:
                lines.append("    ... and %d more" % (len(diffs) - 10))
        raise AssertionError("\n".join(lines))
    if require_distinct_models and report["identical_models"]:
        raise AssertionError(
            "these runs share an identical model section, so the comparison is "
            "vacuous: %s" % report["identical_models"]
        )
    return report
