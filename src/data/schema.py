"""Common dataset representation for binary AIGC detection.

Everything downstream (adapters, manifests, splitting, datasets) speaks in
:class:`ManifestRecord`.  A record describes *one image file on disk*; several
records may share a ``source_id`` when they are transformed derivatives of the
same original image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Binary task convention -- do not change without updating every adapter.
LABEL_REAL = 0
LABEL_AIGC = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_AIGC: "aigc"}

#: Columns every manifest must carry.
REQUIRED_COLUMNS = ("image_path", "label", "source_id")
#: Columns written by :func:`src.data.manifest.write_manifest`, in order.
MANIFEST_COLUMNS = (
    "image_path",
    "label",
    "source_id",
    "dataset",
    "generator",
    "split",
)


class DataError(ValueError):
    """Raised for malformed manifests / records."""


@dataclass
class ManifestRecord:
    """One image on disk.

    Attributes
    ----------
    image_path:
        Path to the image.  May be relative; datasets resolve it against their
        ``root``.
    label:
        ``0`` real, ``1`` AIGC.  Binary by construction.
    source_id:
        Stable identifier of the *underlying original image*.  Every transformed
        derivative of an image MUST carry the same ``source_id`` so that the
        splitter can keep them together.
    dataset:
        Free-form source dataset name (``"cifake"``, ``"sid_set"``, ...).
    generator:
        Free-form generator / model name for AIGC samples (``"sdxl"``, ...).
        Empty for real images.
    split:
        Optional split assignment (``"train"`` / ``"val"`` / ``"test"``).  Empty
        until :func:`src.data.splitting.assign_splits` fills it in.
    extra:
        Any additional metadata.  Kept out of the default collate path.
    """

    image_path: str
    label: int
    source_id: str
    dataset: str = ""
    generator: str = ""
    split: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_path = str(self.image_path)
        try:
            self.label = int(self.label)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise DataError("label %r is not an int" % (self.label,)) from exc
        if self.label not in (LABEL_REAL, LABEL_AIGC):
            raise DataError(
                "label must be %d (real) or %d (aigc), got %r"
                % (LABEL_REAL, LABEL_AIGC, self.label)
            )
        self.source_id = str(self.source_id)
        if not self.source_id:
            raise DataError("source_id must be a non-empty string (%s)" % self.image_path)
        self.dataset = "" if self.dataset is None else str(self.dataset)
        self.generator = "" if self.generator is None else str(self.generator)
        self.split = "" if self.split is None else str(self.split)

    # -- convenience ------------------------------------------------------
    @property
    def is_aigc(self) -> bool:
        return self.label == LABEL_AIGC

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]

    def resolve_path(self, root: Optional[str] = None) -> str:
        if root and not os.path.isabs(self.image_path):
            return os.path.join(root, self.image_path)
        return self.image_path

    def key(self, fields: Sequence[str]) -> tuple:
        """Tuple of ``fields`` -- used for grouping and stratification."""
        return tuple(str(getattr_field(self, f)) for f in fields)

    def with_fields(self, **kwargs: Any) -> "ManifestRecord":
        return replace(self, **kwargs)

    def to_row(self) -> Dict[str, Any]:
        row = {c: getattr(self, c) for c in MANIFEST_COLUMNS}
        for k, v in self.extra.items():
            row.setdefault(k, v)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ManifestRecord":
        missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c] in (None, "")]
        # label == 0 is falsy but valid, so re-check it explicitly.
        missing = [c for c in missing if not (c == "label" and row.get(c) in (0, "0"))]
        if missing:
            raise DataError("manifest row missing required column(s) %s: %r" % (missing, dict(row)))
        known = set(MANIFEST_COLUMNS)
        extra = {k: v for k, v in row.items() if k not in known and v not in (None, "")}
        return cls(
            image_path=row["image_path"],
            label=row["label"],
            source_id=row["source_id"],
            dataset=row.get("dataset") or "",
            generator=row.get("generator") or "",
            split=row.get("split") or "",
            extra=extra,
        )


def getattr_field(record: ManifestRecord, name: str) -> Any:
    """Attribute lookup that falls back to ``record.extra``."""
    if hasattr(record, name):
        return getattr(record, name)
    if name in record.extra:
        return record.extra[name]
    raise DataError("unknown record field %r" % name)


def validate_records(
    records: Iterable[ManifestRecord],
    check_paths_exist: bool = False,
    root: Optional[str] = None,
    require_both_labels: bool = False,
) -> List[ManifestRecord]:
    """Validate a record collection and return it as a list.

    Checks label domain (enforced by the dataclass), duplicate image paths and,
    optionally, that every file exists on disk.
    """
    records = list(records)
    if not records:
        raise DataError("empty record collection")

    seen: Dict[str, int] = {}
    missing_files: List[str] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, ManifestRecord):
            raise DataError("expected ManifestRecord at index %d, got %r" % (i, type(rec)))
        path = rec.resolve_path(root)
        if path in seen:
            raise DataError(
                "duplicate image_path %r at indices %d and %d" % (path, seen[path], i)
            )
        seen[path] = i
        if check_paths_exist and not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        raise DataError(
            "%d image path(s) do not exist, e.g. %s"
            % (len(missing_files), missing_files[:5])
        )
    if require_both_labels:
        labels = {r.label for r in records}
        if labels != {LABEL_REAL, LABEL_AIGC}:
            raise DataError("expected both labels {0, 1}, found %s" % sorted(labels))
    return records


def label_counts(records: Iterable[ManifestRecord]) -> Dict[int, int]:
    counts = {LABEL_REAL: 0, LABEL_AIGC: 0}
    for rec in records:
        counts[rec.label] += 1
    return counts


def source_ids(records: Iterable[ManifestRecord]) -> List[str]:
    """Unique source ids, order-stable."""
    out: List[str] = []
    seen = set()
    for rec in records:
        if rec.source_id not in seen:
            seen.add(rec.source_id)
            out.append(rec.source_id)
    return out
