"""Protected (demonstration-only) data that must never be trained on.

The competition provides a demonstration/validation subset that is explicitly
**not** for training:

===========================  ======  =====
subset                       label   count
===========================  ======  =====
COCO val2017                 real     4998
DALL-E Advanced              AIGC     8843
===========================  ======  =====

Rule 11.B of the project brief is non-negotiable: *no code should silently
include this subset in training*.  This module makes that structural rather than
a matter of remembering to pass a flag:

* :func:`classify_protected` recognises protected data from its ``dataset``
  value **or** its path, so a mislabelled ``dataset`` column does not defeat it;
* :func:`assert_not_trainable` refuses a record collection destined for
  training;
* :func:`validate_splits` runs these checks by default (opt out only via an
  explicit ``allow_protected=True``).

Detection is heuristic on purpose -- it is designed to produce a loud false
alarm rather than a silent miss.  Add project-specific spellings to
:data:`PROTECTED_DATASETS` rather than disabling the check.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import LABEL_AIGC, LABEL_REAL, ManifestRecord

__all__ = [
    "ProtectedDataError",
    "PROTECTED_DATASETS",
    "DEMO_SPLIT_NAMES",
    "classify_protected",
    "find_protected_records",
    "assert_not_trainable",
    "partition_protected",
    "protected_report",
    "register_protected_dataset",
]


class ProtectedDataError(AssertionError):
    """Raised when demonstration-only data reaches a training split."""


class _ProtectedSpec(object):
    """One protected subset: how to recognise it and what it should be."""

    def __init__(
        self,
        key: str,
        description: str,
        expected_label: Optional[int] = None,
        expected_count: Optional[int] = None,
        dataset_patterns: Sequence[str] = (),
        path_patterns: Sequence[str] = (),
    ):
        self.key = key
        self.description = description
        self.expected_label = expected_label
        self.expected_count = expected_count
        self.dataset_res = [re.compile(p, re.IGNORECASE) for p in dataset_patterns]
        self.path_res = [re.compile(p, re.IGNORECASE) for p in path_patterns]

    def matches(self, record: ManifestRecord) -> bool:
        dataset = (record.dataset or "").strip()
        if dataset and any(r.search(dataset) for r in self.dataset_res):
            return True
        # Normalise separators so Windows-style manifests match too.
        path = str(record.image_path).replace("\\", "/")
        return any(r.search(path) for r in self.path_res)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "_ProtectedSpec(%r)" % self.key


#: The competition's demonstration/validation subset.  Keys are stable ids used
#: in reports; patterns are matched case-insensitively.
PROTECTED_DATASETS: "OrderedDict[str, _ProtectedSpec]" = OrderedDict()


def register_protected_dataset(
    key: str,
    description: str,
    expected_label: Optional[int] = None,
    expected_count: Optional[int] = None,
    dataset_patterns: Sequence[str] = (),
    path_patterns: Sequence[str] = (),
) -> None:
    """Add a protected subset.

    Use this to extend the guard for a new demonstration set; never to remove
    one.  Patterns are regular expressions matched against the ``dataset``
    column and the image path respectively.
    """
    PROTECTED_DATASETS[key] = _ProtectedSpec(
        key,
        description,
        expected_label=expected_label,
        expected_count=expected_count,
        dataset_patterns=dataset_patterns,
        path_patterns=path_patterns,
    )


register_protected_dataset(
    "coco_val2017",
    "COCO val2017 (demonstration subset, real / non-AIGC, 4998 images)",
    expected_label=LABEL_REAL,
    expected_count=4998,
    dataset_patterns=(r"coco[_\-\s]*val[_\-\s]*2017", r"^coco[_\-]?val$", r"^coco$"),
    path_patterns=(r"(^|/)val2017(/|$)", r"coco[_\-\s]*val[_\-\s]*2017"),
)
register_protected_dataset(
    "dalle_advanced",
    "DALL-E Advanced (demonstration subset, AIGC, 8843 images)",
    expected_label=LABEL_AIGC,
    expected_count=8843,
    dataset_patterns=(r"dall[\-_\s·]?e.*adv", r"dalle[_\-]?advanced"),
    path_patterns=(r"dall[\-_·]?e[_\-\s]*advanced", r"(^|/)dalle[_\-]?adv"),
)

#: Split names that are *allowed* to contain protected data.  Anything else is
#: treated as a training/model-selection split.
DEMO_SPLIT_NAMES: Tuple[str, ...] = ("demo", "demonstration", "benchmark", "reference")


def classify_protected(record: ManifestRecord) -> Optional[str]:
    """Return the protected-subset key this record belongs to, else ``None``."""
    for key, spec in PROTECTED_DATASETS.items():
        if spec.matches(record):
            return key
    return None


def find_protected_records(
    records: Iterable[ManifestRecord],
) -> "OrderedDict[str, List[ManifestRecord]]":
    """``{protected_key: [records]}`` for everything recognised as protected."""
    found: "OrderedDict[str, List[ManifestRecord]]" = OrderedDict()
    for record in records:
        key = classify_protected(record)
        if key is not None:
            found.setdefault(key, []).append(record)
    return found


def assert_not_trainable(
    records: Iterable[ManifestRecord],
    context: str = "training data",
    max_report: int = 5,
) -> None:
    """Raise :class:`ProtectedDataError` if any record is demonstration-only.

    Call this on anything about to be trained or fitted on.
    """
    found = find_protected_records(records)
    if not found:
        return
    lines = []
    for key, members in found.items():
        spec = PROTECTED_DATASETS[key]
        examples = [r.image_path for r in members[:max_report]]
        lines.append(
            "  %s -- %d record(s): %s%s"
            % (key, len(members), examples, " ..." if len(members) > max_report else "")
        )
        lines.append("    %s" % spec.description)
    raise ProtectedDataError(
        "demonstration-only data found in %s; this subset must NEVER be used for "
        "training (project rule 11.B):\n%s\n"
        "Remove these records, or route them to a split named one of %s."
        % (context, "\n".join(lines), list(DEMO_SPLIT_NAMES))
    )


def partition_protected(
    records: Iterable[ManifestRecord],
) -> Tuple[List[ManifestRecord], List[ManifestRecord]]:
    """Split into ``(trainable, protected)``.

    The safe way to consume a pooled manifest: train on the first list, and use
    the second only for demonstration/benchmark reporting.
    """
    trainable: List[ManifestRecord] = []
    protected: List[ManifestRecord] = []
    for record in records:
        (protected if classify_protected(record) is not None else trainable).append(record)
    return trainable, protected


def protected_report(records: Iterable[ManifestRecord]) -> Dict[str, Any]:
    """Summary of protected data present, including expected-count drift.

    A count that differs from the published size usually means the subset was
    filtered or partially copied -- worth knowing before it is used as the
    demonstration benchmark.
    """
    records = list(records)
    found = find_protected_records(records)
    subsets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for key, members in found.items():
        spec = PROTECTED_DATASETS[key]
        labels = sorted({r.label for r in members})
        entry: Dict[str, Any] = {
            "n_images": len(members),
            "expected_count": spec.expected_count,
            "labels_present": labels,
            "expected_label": spec.expected_label,
            "description": spec.description,
        }
        if spec.expected_count is not None and len(members) != spec.expected_count:
            entry["count_mismatch"] = (
                "found %d, published size is %d" % (len(members), spec.expected_count)
            )
        if spec.expected_label is not None and labels not in ([spec.expected_label], []):
            entry["label_mismatch"] = (
                "expected label %d, found %s" % (spec.expected_label, labels)
            )
        subsets[key] = entry
    return {
        "n_protected": sum(len(v) for v in found.values()),
        "n_total": len(records),
        "subsets": subsets,
    }
