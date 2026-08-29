"""Manifest I/O -- the interchange format between adapters and datasets.

A manifest is a CSV or JSON file with the columns::

    image_path,label,source_id,dataset,generator[,split][,...extra]

CSV is the default; JSON (a list of objects) is used when the path ends in
``.json``/``.jsonl``.  Unknown columns survive a read/write round-trip via
``ManifestRecord.extra``.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schema import (
    MANIFEST_COLUMNS,
    REQUIRED_COLUMNS,
    DataError,
    ManifestRecord,
    label_counts,
    validate_records,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "write_manifest",
    "read_manifest",
    "records_to_dataframe",
    "records_from_dataframe",
    "filter_records",
    "describe_records",
    "merge_manifests",
]


def _fmt_of(path: str, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower().lstrip(".")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("csv", "tsv", "json", "jsonl"):
        return ext
    return "csv"


#: Always written, even when empty -- this is the documented manifest format
#: (``image_path,label,source_id,dataset,generator``) that every consumer may
#: rely on.  ``split`` and ``extra`` columns appear only when populated.
CANONICAL_COLUMNS = ("image_path", "label", "source_id", "dataset", "generator")


def _columns(records: Sequence[ManifestRecord], drop_empty: bool = True) -> List[str]:
    cols = list(MANIFEST_COLUMNS)
    for rec in records:
        for key in rec.extra:
            if key not in cols:
                cols.append(key)
    if drop_empty:
        rows = [rec.to_row() for rec in records]
        for col in [c for c in cols if c not in CANONICAL_COLUMNS]:
            if all(not str(row.get(col, "")) for row in rows):
                cols.remove(col)
    return cols


def write_manifest(
    records: Iterable[ManifestRecord],
    path: str,
    fmt: Optional[str] = None,
    relative_to: Optional[str] = None,
    drop_empty_columns: bool = True,
) -> str:
    """Write records to ``path`` (format inferred from the extension).

    ``relative_to`` rewrites ``image_path`` relative to that directory, which
    keeps manifests portable across machines -- pair it with the dataset's
    ``root`` argument on read.
    """
    records = list(records)
    if not records:
        raise DataError("refusing to write an empty manifest to %s" % path)
    if relative_to:
        records = [
            rec.with_fields(image_path=os.path.relpath(rec.image_path, relative_to))
            for rec in records
        ]

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    kind = _fmt_of(path, fmt)
    cols = _columns(records, drop_empty=drop_empty_columns)
    rows = [{c: rec.to_row().get(c, "") for c in cols} for rec in records]

    if kind in ("csv", "tsv"):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, delimiter="\t" if kind == "tsv" else ",")
            writer.writeheader()
            writer.writerows(rows)
    elif kind == "jsonl":
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    return path


def read_manifest(
    path: str,
    fmt: Optional[str] = None,
    root: Optional[str] = None,
    split: Optional[str] = None,
    check_paths_exist: bool = False,
) -> List[ManifestRecord]:
    """Read a manifest into records.

    ``root`` prefixes relative ``image_path`` values.  ``split`` keeps only rows
    whose ``split`` column matches (raises if the column is absent/empty).
    """
    if not os.path.exists(path):
        raise DataError("manifest not found: %s" % path)
    kind = _fmt_of(path, fmt)

    if kind in ("csv", "tsv"):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t" if kind == "tsv" else ","))
    elif kind == "jsonl":
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        if isinstance(rows, dict):  # tolerate {"records": [...]}
            rows = rows.get("records", [])

    if not rows:
        raise DataError("manifest %s contains no rows" % path)
    header = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise DataError(
            "manifest %s is missing required column(s) %s (found %s)"
            % (path, missing, sorted(header))
        )

    records = [ManifestRecord.from_row(row) for row in rows]
    if root:
        records = [rec.with_fields(image_path=rec.resolve_path(root)) for rec in records]
    if split is not None:
        if not any(rec.split for rec in records):
            raise DataError(
                "manifest %s has no populated 'split' column; run assign_splits first"
                % path
            )
        records = [rec for rec in records if rec.split == split]
        if not records:
            raise DataError("no rows with split=%r in %s" % (split, path))
    return validate_records(records, check_paths_exist=check_paths_exist)


# --------------------------------------------------------------------------
# pandas interop (optional -- pandas is imported lazily)
# --------------------------------------------------------------------------
def records_to_dataframe(records: Iterable[ManifestRecord]):
    import pandas as pd

    records = list(records)
    return pd.DataFrame([rec.to_row() for rec in records], columns=_columns(records, False))


def records_from_dataframe(df) -> List[ManifestRecord]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError("dataframe missing required column(s) %s" % missing)
    return [ManifestRecord.from_row(row) for row in df.to_dict(orient="records")]


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------
def filter_records(
    records: Iterable[ManifestRecord],
    label: Optional[int] = None,
    dataset: Optional[Any] = None,
    generator: Optional[Any] = None,
    split: Optional[Any] = None,
    predicate=None,
) -> List[ManifestRecord]:
    """Filter by field; scalar values match exactly, collections match by ``in``."""

    def _match(value, wanted) -> bool:
        if wanted is None:
            return True
        if isinstance(wanted, (list, tuple, set, frozenset)):
            return value in wanted
        return value == wanted

    return [
        rec
        for rec in records
        if _match(rec.label, label)
        and _match(rec.dataset, dataset)
        and _match(rec.generator, generator)
        and _match(rec.split, split)
        and (predicate is None or predicate(rec))
    ]


def describe_records(records: Iterable[ManifestRecord]) -> Dict[str, Any]:
    """Summary stats: counts, label balance, datasets, generators."""
    records = list(records)
    counts = label_counts(records)
    n = len(records)
    datasets: Dict[str, int] = {}
    generators: Dict[str, int] = {}
    for rec in records:
        datasets[rec.dataset or "?"] = datasets.get(rec.dataset or "?", 0) + 1
        if rec.generator:
            generators[rec.generator] = generators.get(rec.generator, 0) + 1
    return {
        "n_images": n,
        "n_source_ids": len({rec.source_id for rec in records}),
        "n_real": counts[0],
        "n_aigc": counts[1],
        "aigc_fraction": (counts[1] / n) if n else 0.0,
        "views_per_source": (n / len({rec.source_id for rec in records})) if n else 0.0,
        "datasets": datasets,
        "generators": generators,
    }


def merge_manifests(
    *record_lists: Iterable[ManifestRecord], namespace_source_ids: bool = True
) -> List[ManifestRecord]:
    """Concatenate record collections from different datasets.

    ``namespace_source_ids`` prefixes each ``source_id`` with its ``dataset`` so
    that identical filenames in two datasets cannot be treated as the same
    original image (which would silently over-constrain the splitter).
    """
    out: List[ManifestRecord] = []
    for records in record_lists:
        for rec in records:
            if namespace_source_ids and rec.dataset and not rec.source_id.startswith(
                rec.dataset + ":"
            ):
                rec = rec.with_fields(source_id="%s:%s" % (rec.dataset, rec.source_id))
            out.append(rec)
    return validate_records(out)
