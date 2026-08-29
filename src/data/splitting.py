"""Leakage-safe train/val/test splitting.

Splitting happens over **groups**, never over individual images.  A group is
identified by ``source_id`` (optionally namespaced by ``dataset``), so every
transformed derivative of an original image lands in exactly one split.

Stratification is best-effort: groups are bucketed by a stratum key (label by
default, optionally including ``dataset`` / ``generator``) and each bucket is
split independently with the requested ratios.  Tiny buckets therefore cannot
always honour the ratios exactly -- :func:`split_report` shows what happened.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import ManifestRecord, getattr_field, label_counts

__all__ = [
    "LeakageError",
    "DEFAULT_RATIOS",
    "split_records",
    "assign_splits",
    "split_by_source_id",
    "assert_no_source_id_leakage",
    "assert_no_path_overlap",
    "check_split_integrity",
    "split_report",
]

DEFAULT_RATIOS: "OrderedDict[str, float]" = OrderedDict(
    (("train", 0.7), ("val", 0.15), ("test", 0.15))
)


class LeakageError(AssertionError):
    """Raised when the same ``source_id`` (or file) appears in >1 split."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _normalise_ratios(ratios: Any) -> "OrderedDict[str, float]":
    if ratios is None:
        return OrderedDict(DEFAULT_RATIOS)
    if isinstance(ratios, Mapping):
        items = list(ratios.items())
    else:
        seq = list(ratios)
        names = ("train", "val", "test")[: len(seq)]
        if len(seq) > 3:
            raise ValueError("pass a mapping to name more than 3 splits")
        items = list(zip(names, seq))
    if not items:
        raise ValueError("no splits requested")
    out: "OrderedDict[str, float]" = OrderedDict()
    for name, value in items:
        value = float(value)
        if value < 0:
            raise ValueError("negative ratio for split %r" % name)
        out[str(name)] = value
    total = sum(out.values())
    if total <= 0:
        raise ValueError("split ratios sum to 0")
    for name in out:
        out[name] /= total
    return out


def _apportion(
    n: int,
    ratios: "OrderedDict[str, float]",
    carry: Optional[Dict[str, float]] = None,
) -> "OrderedDict[str, int]":
    """Apportion ``n`` groups across splits; counts sum to exactly ``n``.

    Largest-remainder, with the leftover fraction *carried* into the next call
    via ``carry`` (mutated in place).  Without the carry, each stratum rounds
    independently and the biases add up -- 70/15/15 over two 50-group strata
    would land at 70/16/14.  Carrying keeps the global totals on target while
    staying fully deterministic.
    """
    carry = {} if carry is None else carry
    exact = {k: n * v + carry.get(k, 0.0) for k, v in ratios.items()}
    counts = OrderedDict((k, max(0, int(math.floor(exact[k])))) for k in ratios)
    order = list(ratios)
    leftover = n - sum(counts.values())
    if leftover > 0:
        # Hand the spare groups to the largest fractional parts.
        for k in sorted(
            order, key=lambda k: (-(exact[k] - math.floor(exact[k])), order.index(k))
        )[:leftover]:
            counts[k] += 1
    elif leftover < 0:
        # A positive carry can push the floors past ``n``; claw back from the
        # smallest fractional parts first.
        ranked = sorted(
            order, key=lambda k: ((exact[k] - math.floor(exact[k])), order.index(k))
        )
        i = 0
        while leftover < 0:
            k = ranked[i % len(ranked)]
            if counts[k] > 0:
                counts[k] -= 1
                leftover += 1
            i += 1
    for k in order:
        carry[k] = exact[k] - counts[k]
    return counts


def _group_records(
    records: Sequence[ManifestRecord], group_keys: Sequence[str]
) -> "OrderedDict[tuple, List[ManifestRecord]]":
    groups: "OrderedDict[tuple, List[ManifestRecord]]" = OrderedDict()
    for rec in records:
        groups.setdefault(rec.key(group_keys), []).append(rec)
    return groups


def _stratum_key(members: Sequence[ManifestRecord], keys: Sequence[str]) -> tuple:
    """Stratum of a group.

    A group whose members disagree on a key (a real image and its tampered
    derivative sharing a ``source_id``) gets the sorted set of values, so it
    still lands in a single, well-defined stratum.
    """
    out = []
    for key in keys:
        values = sorted({str(getattr_field(m, key)) for m in members})
        out.append(values[0] if len(values) == 1 else "|".join(values))
    return tuple(out)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def split_records(
    records: Iterable[ManifestRecord],
    ratios: Any = None,
    seed: int = 0,
    group_keys: Sequence[str] = ("source_id",),
    stratify_keys: Optional[Sequence[str]] = ("label",),
    verify: bool = True,
) -> "OrderedDict[str, List[ManifestRecord]]":
    """Split records into ``{split_name: [records]}`` without group leakage.

    Parameters
    ----------
    ratios:
        Mapping ``{name: weight}`` or a sequence interpreted as
        ``(train, val, test)``.  Normalised internally, so ``(7, 2, 1)`` works.
        Defaults to 70/15/15.
    seed:
        Deterministic: the same records + seed always produce the same split,
        regardless of input order (groups are sorted before shuffling).
    group_keys:
        Fields defining the indivisible unit.  ``("source_id",)`` by default;
        use ``("dataset", "source_id")`` when pooling datasets whose source ids
        could collide.
    stratify_keys:
        Fields whose distribution should be preserved across splits, e.g.
        ``("label",)`` or ``("dataset", "label")``.  ``None`` disables it.
    verify:
        Run :func:`assert_no_source_id_leakage` on the result.

    Returns
    -------
    Ordered mapping in the order the splits were requested.
    """
    records = list(records)
    if not records:
        raise ValueError("cannot split an empty record collection")
    ratios = _normalise_ratios(ratios)
    group_keys = tuple(group_keys)
    if not group_keys:
        raise ValueError("group_keys must name at least one field")

    groups = _group_records(records, group_keys)

    strata: "OrderedDict[tuple, List[tuple]]" = OrderedDict()
    if stratify_keys:
        stratify_keys = tuple(stratify_keys)
        for gkey, members in groups.items():
            strata.setdefault(_stratum_key(members, stratify_keys), []).append(gkey)
    else:
        strata[("all",)] = list(groups.keys())

    out: "OrderedDict[str, List[ManifestRecord]]" = OrderedDict((k, []) for k in ratios)
    carry: Dict[str, float] = {k: 0.0 for k in ratios}
    # Sort strata and their groups so the result is independent of input order.
    for stratum in sorted(strata, key=lambda t: tuple(map(str, t))):
        gkeys = sorted(strata[stratum], key=lambda t: tuple(map(str, t)))
        # Per-stratum seed: adding a stratum does not reshuffle the others.
        rng = random.Random("%s|%s" % (seed, "|".join(map(str, stratum))))
        rng.shuffle(gkeys)
        counts = _apportion(len(gkeys), ratios, carry)
        pos = 0
        for split_name, count in counts.items():
            for gkey in gkeys[pos : pos + count]:
                out[split_name].extend(groups[gkey])
            pos += count

    if verify:
        assert_no_source_id_leakage(out, group_keys=group_keys)
        assert_no_path_overlap(out)
    return out


#: Backwards/intent-friendly alias.
def split_by_source_id(records, **kwargs):
    """Alias for :func:`split_records` (splitting is by ``source_id``)."""
    return split_records(records, **kwargs)


def assign_splits(
    records: Iterable[ManifestRecord],
    ratios: Any = None,
    seed: int = 0,
    group_keys: Sequence[str] = ("source_id",),
    stratify_keys: Optional[Sequence[str]] = ("label",),
    verify: bool = True,
) -> List[ManifestRecord]:
    """Same as :func:`split_records` but returns one flat list with ``.split`` set.

    Handy for persisting a single manifest that carries its own split column.
    Input records are not mutated; copies are returned in the original order.
    """
    records = list(records)
    splits = split_records(
        records,
        ratios=ratios,
        seed=seed,
        group_keys=group_keys,
        stratify_keys=stratify_keys,
        verify=verify,
    )
    assignment = {}
    for name, members in splits.items():
        for rec in members:
            assignment[id(rec)] = name
    return [rec.with_fields(split=assignment[id(rec)]) for rec in records]


# --------------------------------------------------------------------------
# leakage checks
# --------------------------------------------------------------------------
def _split_group_sets(
    splits: Mapping[str, Sequence[ManifestRecord]], group_keys: Sequence[str]
) -> "OrderedDict[str, set]":
    return OrderedDict(
        (name, {rec.key(tuple(group_keys)) for rec in members})
        for name, members in splits.items()
    )


def assert_no_source_id_leakage(
    splits: Mapping[str, Sequence[ManifestRecord]],
    group_keys: Sequence[str] = ("source_id",),
    max_report: int = 10,
) -> None:
    """Raise :class:`LeakageError` if any group spans two splits."""
    sets = _split_group_sets(splits, group_keys)
    names = list(sets)
    problems: List[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            if shared:
                sample = sorted(map(str, list(shared)))[:max_report]
                problems.append(
                    "%s <-> %s: %d shared group(s), e.g. %s"
                    % (a, b, len(shared), sample)
                )
    if problems:
        raise LeakageError(
            "source_id leakage detected across splits (grouped by %s):\n  %s"
            % (list(group_keys), "\n  ".join(problems))
        )


def assert_no_path_overlap(
    splits: Mapping[str, Sequence[ManifestRecord]], max_report: int = 10
) -> None:
    """Raise :class:`LeakageError` if the same image file appears in two splits."""
    sets = OrderedDict(
        (name, {rec.image_path for rec in members}) for name, members in splits.items()
    )
    names = list(sets)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            if shared:
                problems.append(
                    "%s <-> %s: %d shared path(s), e.g. %s"
                    % (a, b, len(shared), sorted(shared)[:max_report])
                )
    if problems:
        raise LeakageError("duplicate image paths across splits:\n  " + "\n  ".join(problems))


def check_split_integrity(
    splits: Mapping[str, Sequence[ManifestRecord]],
    original: Optional[Sequence[ManifestRecord]] = None,
    group_keys: Sequence[str] = ("source_id",),
    require_all_splits_nonempty: bool = True,
) -> Dict[str, Any]:
    """Full integrity sweep.  Raises on any violation, else returns the report.

    Checks: no group leakage, no duplicate files, every split non-empty
    (optional), and -- when ``original`` is given -- that the split is a
    partition of the input (nothing lost, nothing invented).
    """
    assert_no_source_id_leakage(splits, group_keys=group_keys)
    assert_no_path_overlap(splits)

    if require_all_splits_nonempty:
        empty = [name for name, members in splits.items() if not members]
        if empty:
            raise LeakageError(
                "empty split(s) %s -- too few source_id groups for the requested "
                "ratios; lower the number of splits or add data" % empty
            )

    if original is not None:
        want = sorted(rec.image_path for rec in original)
        got = sorted(rec.image_path for split in splits.values() for rec in split)
        if want != got:
            missing = sorted(set(want) - set(got))[:10]
            extra = sorted(set(got) - set(want))[:10]
            raise LeakageError(
                "split is not a partition of the input: %d missing (e.g. %s), "
                "%d unexpected (e.g. %s)"
                % (len(set(want) - set(got)), missing, len(set(got) - set(want)), extra)
            )
    return split_report(splits, group_keys=group_keys)


def split_report(
    splits: Mapping[str, Sequence[ManifestRecord]],
    group_keys: Sequence[str] = ("source_id",),
) -> Dict[str, Any]:
    """Per-split counts, label balance, group counts and dataset breakdown."""
    total = sum(len(m) for m in splits.values()) or 1
    report: Dict[str, Any] = {"n_images": total - 0, "splits": OrderedDict()}
    for name, members in splits.items():
        counts = label_counts(members)
        by_dataset: Dict[str, int] = defaultdict(int)
        for rec in members:
            by_dataset[rec.dataset or "?"] += 1
        n = len(members)
        report["splits"][name] = {
            "n_images": n,
            "fraction": n / total,
            "n_groups": len({rec.key(tuple(group_keys)) for rec in members}),
            "n_real": counts[0],
            "n_aigc": counts[1],
            "aigc_fraction": (counts[1] / n) if n else 0.0,
            "by_dataset": dict(by_dataset),
        }
    return report


def format_split_report(report: Mapping[str, Any]) -> str:
    """Human-readable one-line-per-split rendering of :func:`split_report`."""
    lines = ["%-8s %8s %8s %8s %8s %7s" % ("split", "images", "groups", "real", "aigc", "frac")]
    for name, info in report["splits"].items():
        lines.append(
            "%-8s %8d %8d %8d %8d %6.1f%%"
            % (
                name,
                info["n_images"],
                info["n_groups"],
                info["n_real"],
                info["n_aigc"],
                100 * info["fraction"],
            )
        )
    return "\n".join(lines)
