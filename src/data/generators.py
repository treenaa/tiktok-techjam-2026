"""Generator-aware filtering, grouping and splitting.

For "train on some generators, test on unseen ones" protocols.  Everything is
driven by the ``generator`` metadata column -- no generator name is hard-coded
anywhere, so a new dataset or a new model family needs no code change.

Real images (``generator == ""``) are never treated as a generator; they are
distributed across splits by the usual leakage-safe splitter so every split
keeps both classes.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import LABEL_AIGC, LABEL_REAL, DataError, ManifestRecord
from .splitting import (
    LeakageError,
    assert_no_source_id_leakage,
    split_records,
)

__all__ = [
    "list_generators",
    "generator_counts",
    "group_by_generator",
    "filter_by_generator",
    "partition_generators",
    "split_by_generator_holdout",
    "assert_generators_disjoint",
]

REAL_GENERATOR = ""  # real images carry an empty generator field


def list_generators(records: Iterable[ManifestRecord], include_real: bool = False) -> List[str]:
    """Sorted unique generator names present in ``records``."""
    names = {rec.generator for rec in records}
    if not include_real:
        names.discard(REAL_GENERATOR)
    return sorted(names)


def generator_counts(records: Iterable[ManifestRecord]) -> "OrderedDict[str, int]":
    """``{generator: n_images}``, most frequent first (real images included)."""
    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec.generator] = counts.get(rec.generator, 0) + 1
    return OrderedDict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def group_by_generator(
    records: Iterable[ManifestRecord], include_real: bool = True
) -> "OrderedDict[str, List[ManifestRecord]]":
    """``{generator: [records]}``; real images land under ``""``."""
    groups: "OrderedDict[str, List[ManifestRecord]]" = OrderedDict()
    for rec in records:
        if not include_real and rec.generator == REAL_GENERATOR:
            continue
        groups.setdefault(rec.generator, []).append(rec)
    return groups


def filter_by_generator(
    records: Iterable[ManifestRecord],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    keep_real: bool = True,
    predicate: Optional[Callable[[str], bool]] = None,
) -> List[ManifestRecord]:
    """Select records by generator.

    Parameters
    ----------
    include:
        Keep only these generators (real images still kept when ``keep_real``).
    exclude:
        Drop these generators.
    keep_real:
        Keep ``label == 0`` / empty-generator records regardless of the filters
        -- usually what you want, since real images are generator-agnostic.
    predicate:
        Arbitrary ``generator_name -> bool``, applied to AIGC records.  Lets
        callers express prefix/family rules without this module knowing any
        names.
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude or ())

    out = []
    for rec in records:
        if rec.generator == REAL_GENERATOR or rec.label == LABEL_REAL:
            if keep_real:
                out.append(rec)
            continue
        if include_set is not None and rec.generator not in include_set:
            continue
        if rec.generator in exclude_set:
            continue
        if predicate is not None and not predicate(rec.generator):
            continue
        out.append(rec)
    return out


def partition_generators(
    records: Iterable[ManifestRecord],
    holdout: Optional[Sequence[str]] = None,
    n_holdout: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Choose which generators are 'seen' and which are held out.

    Either name the ``holdout`` generators explicitly, or ask for ``n_holdout``
    of them to be chosen deterministically from the sorted list of names.
    """
    available = list_generators(records)
    if not available:
        raise DataError(
            "no generator metadata present -- generator-aware splitting needs a "
            "populated 'generator' column"
        )
    if holdout is not None:
        holdout_list = sorted(set(holdout))
        unknown = [g for g in holdout_list if g not in available]
        if unknown:
            raise DataError(
                "holdout generator(s) %s not present; available: %s" % (unknown, available)
            )
    elif n_holdout is not None:
        if not 0 < n_holdout < len(available):
            raise DataError(
                "n_holdout must be in (0, %d) for %d generators, got %r"
                % (len(available), len(available), n_holdout)
            )
        rng = random.Random(seed)
        holdout_list = sorted(rng.sample(available, n_holdout))
    else:
        raise DataError("pass either holdout=[...] or n_holdout=N")

    seen = [g for g in available if g not in set(holdout_list)]
    if not seen:
        raise DataError("holding out %s would leave no training generators" % holdout_list)
    return seen, holdout_list


def split_by_generator_holdout(
    records: Iterable[ManifestRecord],
    holdout: Optional[Sequence[str]] = None,
    n_holdout: Optional[int] = None,
    seed: int = 0,
    val_ratio: float = 0.15,
    real_ratios: Optional[Mapping[str, float]] = None,
    holdout_split: str = "test",
    verify: bool = True,
) -> "OrderedDict[str, List[ManifestRecord]]":
    """Split so that the held-out generators appear **only** in one split.

    Produces ``train`` / ``val`` / ``<holdout_split>`` where:

    * AIGC images from held-out generators go exclusively to ``holdout_split``;
    * AIGC images from seen generators are split between ``train`` and ``val``;
    * real images are split across all three by the usual leakage-safe splitter,
      so every split keeps both classes.

    ``source_id`` grouping is preserved throughout, so no transformed derivative
    crosses a split either.

    Returns
    -------
    Ordered mapping with a ``"_generators"``-free payload; use
    :func:`list_generators` on each split to inspect what landed where.
    """
    records = list(records)
    seen, held = partition_generators(records, holdout=holdout, n_holdout=n_holdout, seed=seed)

    real = [r for r in records if r.label == LABEL_REAL]
    seen_aigc = [r for r in records if r.label == LABEL_AIGC and r.generator in set(seen)]
    held_aigc = [r for r in records if r.label == LABEL_AIGC and r.generator in set(held)]
    if not held_aigc:
        raise DataError("no AIGC records for held-out generators %s" % held)

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1), got %r" % (val_ratio,))

    out: "OrderedDict[str, List[ManifestRecord]]" = OrderedDict(
        (name, []) for name in ("train", "val", holdout_split)
    )

    # Seen generators -> train/val, grouped by source_id and stratified by generator.
    if seen_aigc:
        aigc_splits = split_records(
            seen_aigc,
            ratios={"train": 1.0 - val_ratio, "val": val_ratio} if val_ratio else {"train": 1.0},
            seed=seed,
            stratify_keys=("generator",),
            verify=False,
        )
        for name, members in aigc_splits.items():
            out[name].extend(members)

    # Real images -> all splits, so none is single-class.
    ratios = OrderedDict(real_ratios) if real_ratios else OrderedDict(
        (("train", 1.0 - val_ratio - 0.15), ("val", val_ratio), (holdout_split, 0.15))
    )
    if real:
        real_splits = split_records(
            real, ratios=ratios, seed=seed, stratify_keys=None, verify=False
        )
        for name, members in real_splits.items():
            out.setdefault(name, []).extend(members)

    out[holdout_split].extend(held_aigc)

    if verify:
        assert_no_source_id_leakage(out)
        assert_generators_disjoint(out, holdout_split=holdout_split, holdout=held)
    return out


def assert_generators_disjoint(
    splits: Mapping[str, Sequence[ManifestRecord]],
    holdout_split: str = "test",
    holdout: Optional[Sequence[str]] = None,
) -> None:
    """Assert held-out generators appear only in ``holdout_split``.

    With ``holdout=None`` the check is symmetric: no AIGC generator may be
    shared between ``holdout_split`` and any other split.
    """
    if holdout_split not in splits:
        raise LeakageError("holdout split %r not present in %s" % (holdout_split, list(splits)))

    if holdout is None:
        held = set(list_generators(splits[holdout_split]))
        others = set()
        for name, members in splits.items():
            if name != holdout_split:
                others |= set(list_generators(members))
        shared = held & others
    else:
        held = set(holdout)
        shared = set()
        for name, members in splits.items():
            if name == holdout_split:
                continue
            shared |= held & set(list_generators(members))

    if shared:
        raise LeakageError(
            "generator leakage: %s appear both in %r and in other splits"
            % (sorted(shared), holdout_split)
        )
