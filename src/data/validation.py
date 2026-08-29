"""Pre-training split validation -- the last gate before a training run.

``validate_splits`` is the loud, reusable check the training owner should call
on the manifests it is about to consume.  It accepts manifest *paths* or
in-memory record lists, and raises :class:`~src.data.splitting.LeakageError`
listing every violation it found rather than the first one.

Deliberately paranoid: it re-derives derivative relationships from filenames
even when ``source_id`` claims the splits are clean, because a wrong
``source_id`` policy is exactly the failure this is meant to catch.
"""

from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .manifest import read_manifest
from .schema import ManifestRecord
from .source_id import strip_transform_suffixes
from .splitting import LeakageError, split_report

__all__ = [
    "ValidationReport",
    "validate_splits",
    "normalized_stem",
    "find_derivative_leakage",
    "find_forbidden_combinations",
]

ManifestLike = Union[str, Iterable[ManifestRecord]]


def _load(manifest: ManifestLike, name: str) -> List[ManifestRecord]:
    if manifest is None:
        return []
    if isinstance(manifest, str):
        return read_manifest(manifest)
    records = list(manifest)
    if records and not isinstance(records[0], ManifestRecord):
        raise TypeError(
            "%s must be a manifest path or an iterable of ManifestRecord, got %r"
            % (name, type(records[0]).__name__)
        )
    return records


def normalized_stem(path: str) -> str:
    """Filename stem with transform suffixes stripped, lowercased.

    The fallback identity used to spot derivatives whose ``source_id`` failed to
    tie them together.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return strip_transform_suffixes(stem).lower()


class ValidationReport(object):
    """Result of :func:`validate_splits`: per-split stats plus every problem."""

    def __init__(self, splits: "OrderedDict[str, List[ManifestRecord]]"):
        self.splits = splits
        self.problems: "OrderedDict[str, List[str]]" = OrderedDict()
        self.stats: Dict[str, Any] = {}

    def add(self, category: str, messages: Sequence[str]) -> None:
        if messages:
            self.problems.setdefault(category, []).extend(messages)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = ["split validation: %s" % ("PASSED" if self.ok else "FAILED")]
        for name, info in self.stats.get("splits", {}).items():
            lines.append(
                "  %-8s %6d images  %6d sources  real=%d aigc=%d"
                % (name, info["n_images"], info["n_groups"], info["n_real"], info["n_aigc"])
            )
        for category, messages in self.problems.items():
            lines.append("  [%s] %d problem(s):" % (category, len(messages)))
            lines.extend("    - " + m for m in messages)
        return "\n".join(lines)

    def raise_if_failed(self) -> "ValidationReport":
        if not self.ok:
            raise LeakageError(self.summary())
        return self

    def __repr__(self) -> str:
        return "ValidationReport(ok=%s, problems=%d)" % (
            self.ok,
            sum(len(v) for v in self.problems.values()),
        )


def _pairwise_overlap(
    sets: "OrderedDict[str, set]", label: str, max_report: int = 5
) -> List[str]:
    names = list(sets)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            if shared:
                sample = sorted(map(str, shared))[:max_report]
                out.append(
                    "%s shared between %r and %r: %d (e.g. %s)"
                    % (label, a, b, len(shared), sample)
                )
    return out


def find_derivative_leakage(
    splits: Mapping[str, Sequence[ManifestRecord]], max_report: int = 5
) -> List[str]:
    """Files that look like transforms of one original but sit in two splits.

    Catches the case where ``source_id`` was derived badly (e.g. the raw stem
    was used, so ``cat_jpeg70.png`` got its own id) -- metadata says the splits
    are disjoint while the pixels say otherwise.

    Scoped **per dataset**: two datasets may legitimately both contain
    ``0001.png`` without those being the same image.

    A group is only reported when at least two of its members have *different*
    raw stems -- i.e. a transform suffix was genuinely stripped from one of
    them.  Plain filename reuse (CIFAKE stores ``0001.png`` under both
    ``train/REAL`` and ``test/REAL``) is not a derivative relationship and is
    ignored.

    Known limit: files whose raw names are identical but live in
    transform-named *directories* (``orig/cat.png`` vs ``jpeg70/cat.png``) are
    not caught here -- use a ``relpath`` source_id policy for such layouts.
    """
    owners: Dict[tuple, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    raw_stems: Dict[tuple, set] = defaultdict(set)
    for split_name, records in splits.items():
        for rec in records:
            key = (rec.dataset, normalized_stem(rec.image_path))
            owners[key][split_name].append(rec.image_path)
            raw_stems[key].add(os.path.splitext(os.path.basename(rec.image_path))[0].lower())

    problems = []
    for key, by_split in owners.items():
        dataset, stem = key
        if len(by_split) <= 1:
            continue
        if len(raw_stems[key]) <= 1:
            # Same filename in several places -- reuse, not a derivative pair.
            continue
        example = {k: v[0] for k, v in list(by_split.items())[:3]}
        problems.append(
            "images sharing normalized stem %r%s appear in splits %s (e.g. %s)"
            % (stem, (" in dataset %r" % dataset) if dataset else "", sorted(by_split), example)
        )
        if len(problems) >= max_report:
            break
    return problems


def find_forbidden_combinations(
    splits: Mapping[str, Sequence[ManifestRecord]],
    forbidden: Mapping[str, Mapping[str, Sequence[Any]]],
    max_report: int = 5,
) -> List[str]:
    """Check configured "this must not appear in that split" rules.

    ``forbidden`` maps a split name to ``{field: [disallowed values]}``::

        {"test": {"dataset": ["cifake"], "generator": ["sdxl"]}}

    reads as "no CIFAKE images and no SDXL images may appear in test" -- the
    mechanism behind unseen-generator and unseen-dataset protocols.
    """
    problems: List[str] = []
    for split_name, rules in forbidden.items():
        records = splits.get(split_name)
        if records is None:
            problems.append("forbidden rule names unknown split %r" % split_name)
            continue
        for field, values in rules.items():
            banned = set(values)
            hits = [
                rec.image_path
                for rec in records
                if str(getattr(rec, field, rec.extra.get(field, ""))) in banned
            ]
            if hits:
                problems.append(
                    "split %r contains %d record(s) with forbidden %s in %s (e.g. %s)"
                    % (split_name, len(hits), field, sorted(banned), hits[:max_report])
                )
    return problems


def validate_splits(
    train_manifest: ManifestLike = None,
    val_manifest: ManifestLike = None,
    test_manifest: ManifestLike = None,
    extra_splits: Optional[Mapping[str, ManifestLike]] = None,
    group_keys: Sequence[str] = ("source_id",),
    check_derivatives: bool = True,
    forbidden: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
    require_nonempty: bool = True,
    require_both_labels: bool = True,
    check_paths_exist: bool = False,
    raise_on_failure: bool = True,
) -> ValidationReport:
    """Validate a set of splits before training.  Fails loudly by default.

    Accepts manifest paths or record lists::

        validate_splits("manifests/train.csv", "manifests/val.csv",
                        "manifests/test.csv")

    Detects
    -------
    * the same ``source_id`` in more than one split;
    * the same file path in more than one split;
    * transformed derivatives split apart, inferred from filenames even when
      ``source_id`` disagrees (``check_derivatives``);
    * configured forbidden dataset/generator/source combinations
      (``forbidden``);
    * empty splits and single-class splits, which silently break training.

    Parameters
    ----------
    raise_on_failure:
        ``True`` (default) raises :class:`LeakageError` with the full report.
        ``False`` returns the report so callers can inspect ``.problems``.
    """
    splits: "OrderedDict[str, List[ManifestRecord]]" = OrderedDict()
    for name, manifest in (
        ("train", train_manifest),
        ("val", val_manifest),
        ("test", test_manifest),
    ):
        if manifest is not None:
            splits[name] = _load(manifest, name + "_manifest")
    for name, manifest in (extra_splits or {}).items():
        splits[name] = _load(manifest, name)

    if not splits:
        raise ValueError("validate_splits requires at least one manifest")

    report = ValidationReport(splits)
    group_keys = tuple(group_keys)

    # -- structural problems ---------------------------------------------
    structural: List[str] = []
    for name, records in splits.items():
        if not records:
            if require_nonempty:
                structural.append("split %r is empty" % name)
            continue
        labels = {rec.label for rec in records}
        if require_both_labels and labels != {0, 1}:
            structural.append(
                "split %r contains only label(s) %s -- both 0 (real) and 1 (aigc) are required"
                % (name, sorted(labels))
            )
        if check_paths_exist:
            missing = [r.image_path for r in records if not os.path.exists(r.image_path)]
            if missing:
                structural.append(
                    "split %r references %d missing file(s) (e.g. %s)"
                    % (name, len(missing), missing[:5])
                )
        duplicates = _duplicates([r.image_path for r in records])
        if duplicates:
            structural.append(
                "split %r lists %d duplicate path(s) (e.g. %s)"
                % (name, len(duplicates), duplicates[:5])
            )
    report.add("structure", structural)

    populated = OrderedDict((k, v) for k, v in splits.items() if v)

    # -- the leakage checks proper ---------------------------------------
    report.add(
        "source_id_overlap",
        _pairwise_overlap(
            OrderedDict(
                (name, {rec.key(group_keys) for rec in records})
                for name, records in populated.items()
            ),
            "source_id",
        ),
    )
    report.add(
        "path_overlap",
        _pairwise_overlap(
            OrderedDict(
                (name, {rec.image_path for rec in records})
                for name, records in populated.items()
            ),
            "image path",
        ),
    )
    if check_derivatives:
        report.add("derivative_leakage", find_derivative_leakage(populated))
    if forbidden:
        report.add("forbidden_combination", find_forbidden_combinations(populated, forbidden))

    report.stats = split_report(populated, group_keys=group_keys) if populated else {"splits": {}}
    report.stats["n_splits"] = len(splits)

    if raise_on_failure:
        report.raise_if_failed()
    return report


def _duplicates(values: Sequence[str]) -> List[str]:
    seen, dupes = set(), []
    for value in values:
        if value in seen:
            dupes.append(value)
        else:
            seen.add(value)
    return dupes
