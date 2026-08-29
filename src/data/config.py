"""Config-driven dataset specification (project rule 20.8).

Keeps dataset paths and split policy out of code.  A config is JSON or YAML
(YAML only if ``pyyaml`` is installed; JSON always works):

.. code-block:: yaml

    seed: 42
    ratios: {train: 0.7, val: 0.15, test: 0.15}
    stratify_keys: [label]
    group_keys: [source_id]
    datasets:
      - name: cifake
        adapter: cifake
        root: /data/cifake
      - name: wildfake
        adapter: wildfake
        root: /data/wildfake
        generator_depth: 0
    demo:                      # demonstration-only; never trained on
      - name: coco_val2017
        adapter: folder
        root: /data/coco/val2017
        fixed_label: 0

``build_from_config`` returns leakage-checked splits, with the demonstration
subset kept in a separate ``demo`` split that the protected-data guard permits.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters import build_manifest
from .protected import DEMO_SPLIT_NAMES, assert_not_trainable
from .schema import DataError, ManifestRecord
from .splitting import split_records
from .validation import validate_splits

__all__ = ["DatasetConfigError", "load_config", "build_from_config", "DEFAULT_CONFIG"]


class DatasetConfigError(ValueError):
    """Raised for malformed dataset configuration."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,
    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    "stratify_keys": ["label"],
    "group_keys": ["source_id"],
    "datasets": [],
    "demo": [],
}

_ADAPTER_PASSTHROUGH = (
    "class_map", "source_id_policy", "generator_depth", "generator", "label",
    "extensions", "on_unlabelled", "namespace_source_ids", "split",
    "tampered_share_real_source_id", "dataset",
)


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML dataset config and fill in defaults."""
    if not os.path.exists(path):
        raise DatasetConfigError("config not found: %s" % path)
    text = open(path, encoding="utf-8").read()
    extension = os.path.splitext(path)[1].lower()

    if extension in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise DatasetConfigError(
                "%s is YAML but pyyaml is not installed; install pyyaml or use JSON" % path
            ) from None
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, Mapping):
        raise DatasetConfigError("config root must be a mapping, got %r" % type(raw).__name__)

    config = dict(DEFAULT_CONFIG)
    config.update(raw)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise DatasetConfigError(
            "unknown config key(s) %s; allowed: %s" % (sorted(unknown), sorted(DEFAULT_CONFIG))
        )
    if not config["datasets"]:
        raise DatasetConfigError("config lists no datasets")
    for section in ("datasets", "demo"):
        for entry in config[section]:
            if "root" not in entry:
                raise DatasetConfigError("%s entry %r has no 'root'" % (section, entry))
    return config


def _records_for(entry: Mapping[str, Any], section: str) -> List[ManifestRecord]:
    entry = dict(entry)
    root = entry.pop("root")
    name = entry.pop("name", None)
    adapter = entry.pop("adapter", "folder")
    fixed_label = entry.pop("fixed_label", None)

    kwargs = {k: v for k, v in entry.items() if k in _ADAPTER_PASSTHROUGH}
    unknown = set(entry) - set(_ADAPTER_PASSTHROUGH)
    if unknown:
        raise DatasetConfigError(
            "unknown key(s) %s in %s entry %r" % (sorted(unknown), section, name or root)
        )
    kwargs.setdefault("dataset", name or adapter)

    if fixed_label is not None:
        # A directory that is entirely one class (e.g. COCO val2017 = all real).
        # `from_folder` labels by directory *name*, which cannot work when the
        # images sit directly in the root or under arbitrary subfolders -- so
        # accept every image and stamp the declared label on it.
        label = int(fixed_label)
        if label not in (0, 1):
            raise DatasetConfigError("fixed_label must be 0 or 1, got %r" % fixed_label)
        kwargs.pop("class_map", None)
        kwargs["on_unlabelled"] = "skip"
        kwargs["label"] = label
        return build_manifest("folder", root, **kwargs)
    return build_manifest(adapter, root, **kwargs)


def build_from_config(
    config: Any,
    validate: bool = True,
    demo_split_name: str = "demo",
) -> "OrderedDict[str, List[ManifestRecord]]":
    """Build leakage-checked splits from a config path or mapping.

    Datasets under ``datasets`` are pooled and split by ``source_id``.  Anything
    under ``demo`` is kept out of train/val/test entirely and returned as a
    separate split, since it is demonstration-only (rule 11.B).

    Returns an ordered mapping ``{"train", "val", "test"[, "demo"]}``.
    """
    if isinstance(config, str):
        config = load_config(config)
    else:
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)
        config = merged
    if demo_split_name.lower() not in {n.lower() for n in DEMO_SPLIT_NAMES}:
        raise DatasetConfigError(
            "demo_split_name %r must be one of %s so the protected-data guard "
            "recognises it" % (demo_split_name, list(DEMO_SPLIT_NAMES))
        )

    trainable: List[ManifestRecord] = []
    for entry in config["datasets"]:
        trainable.extend(_records_for(entry, "datasets"))
    if not trainable:
        raise DatasetConfigError("no records were produced from 'datasets'")

    # Hard stop: a protected subset listed under `datasets` is a config error.
    assert_not_trainable(trainable, context="the 'datasets' section of the config")

    splits = split_records(
        trainable,
        ratios=config["ratios"],
        seed=config["seed"],
        stratify_keys=tuple(config["stratify_keys"]) if config["stratify_keys"] else None,
        group_keys=tuple(config["group_keys"]),
        verify=True,
    )

    demo: List[ManifestRecord] = []
    for entry in config.get("demo", []):
        demo.extend(_records_for(entry, "demo"))
    if demo:
        splits[demo_split_name] = demo

    if validate:
        validate_splits(
            splits.get("train"),
            splits.get("val"),
            splits.get("test"),
            extra_splits={demo_split_name: demo} if demo else None,
            group_keys=tuple(config["group_keys"]),
        )
    return splits
