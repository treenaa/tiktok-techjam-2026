"""Dataset-shortcut auditing.

Rule 11.C: a detector must learn *AI-ness*, not dataset identity.  The classic
trap is real=COCO / AIGC=one generator, where the model can score 98% by
recognising COCO's photographic style and then collapse on new sources.

This module surfaces the properties a model could exploit as a shortcut:

* **provenance** -- is a label almost perfectly predicted by ``dataset``?
* **resolution** -- do the classes have systematically different image sizes?
* **encoding** -- is one class mostly PNG and the other mostly JPEG?
* **file size** -- does compressed size alone separate the classes?
* **generator concentration** -- is the AIGC side effectively one generator?

Nothing here proves a model *will* cheat; it reports how easily it *could*.
Findings are advisory by default (``severity`` on each), because some skew is
unavoidable at hackathon scale -- what matters is knowing about it and saying so
in the write-up.

Reading image dimensions requires opening files; PIL reads only the header, so
this stays cheap, and ``sample_size`` caps the work on large manifests.
"""

from __future__ import annotations

import os
import random
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import LABEL_AIGC, LABEL_REAL, ManifestRecord

__all__ = [
    "ShortcutFinding",
    "audit_shortcuts",
    "format_audit_report",
    "provenance_shortcut",
    "resolution_shortcut",
    "encoding_shortcut",
    "generator_concentration",
]

#: Above this, a single feature predicts the label well enough to be a shortcut.
DEFAULT_PURITY_THRESHOLD = 0.95
#: Fraction of a class that must share a property before it counts as skewed.
DEFAULT_DOMINANCE_THRESHOLD = 0.90


class ShortcutFinding(object):
    """One potential shortcut, with the evidence behind it."""

    def __init__(
        self,
        kind: str,
        severity: str,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
    ):
        if severity not in ("info", "warning", "critical"):
            raise ValueError("severity must be info/warning/critical, got %r" % severity)
        self.kind = kind
        self.severity = severity
        self.summary = summary
        self.detail = detail or {}

    def __repr__(self) -> str:
        return "ShortcutFinding(%s, %s, %r)" % (self.kind, self.severity, self.summary)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "detail": self.detail,
        }


def _label_counts_by(records: Sequence[ManifestRecord], key) -> "OrderedDict[Any, Counter]":
    table: "OrderedDict[Any, Counter]" = OrderedDict()
    for record in records:
        table.setdefault(key(record), Counter())[record.label] += 1
    return table


# --------------------------------------------------------------------------
# provenance: dataset -> label
# --------------------------------------------------------------------------
def provenance_shortcut(
    records: Sequence[ManifestRecord],
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
) -> List[ShortcutFinding]:
    """Is the label recoverable from the ``dataset`` column alone?

    If every dataset is single-label, a model that merely identifies the source
    corpus scores perfectly -- the headline risk in rule 11.C.
    """
    findings: List[ShortcutFinding] = []
    by_dataset = _label_counts_by(records, lambda r: r.dataset or "?")
    if len(by_dataset) < 2:
        findings.append(
            ShortcutFinding(
                "provenance",
                "warning",
                "all records come from a single dataset %r; cross-source "
                "generalization cannot be assessed from this manifest"
                % list(by_dataset)[0],
                {"datasets": list(by_dataset)},
            )
        )
        return findings

    pure = OrderedDict()
    for dataset, counts in by_dataset.items():
        total = sum(counts.values())
        majority = max(counts.values()) / total if total else 0.0
        pure[dataset] = {
            "n": total,
            "purity": majority,
            "labels": dict(counts),
        }

    n_pure = sum(1 for v in pure.values() if v["purity"] >= purity_threshold)
    covered = sum(v["n"] for v in pure.values() if v["purity"] >= purity_threshold)
    fraction = covered / max(len(records), 1)

    if n_pure == len(pure):
        findings.append(
            ShortcutFinding(
                "provenance",
                "critical",
                "every dataset is single-label (purity >= %.2f): the label is "
                "perfectly predictable from provenance, so high accuracy may "
                "reflect dataset identity rather than AI-ness"
                % purity_threshold,
                {"per_dataset": pure},
            )
        )
    elif fraction > 0.5:
        findings.append(
            ShortcutFinding(
                "provenance",
                "warning",
                "%.0f%% of records sit in single-label datasets; provenance is a "
                "strong partial shortcut" % (100 * fraction),
                {"per_dataset": pure},
            )
        )
    else:
        findings.append(
            ShortcutFinding(
                "provenance",
                "info",
                "datasets contribute both labels; provenance alone is a weak predictor",
                {"per_dataset": pure},
            )
        )
    return findings


# --------------------------------------------------------------------------
# resolution / encoding / file size
# --------------------------------------------------------------------------
def _probe(
    records: Sequence[ManifestRecord],
    root: Optional[str],
    sample_size: Optional[int],
    seed: int,
) -> "OrderedDict[int, Dict[str, List[Any]]]":
    """Collect per-label size/format/bytes, reading only image headers."""
    from PIL import Image

    chosen = list(records)
    if sample_size is not None and len(chosen) > sample_size:
        chosen = random.Random(seed).sample(chosen, sample_size)

    out: "OrderedDict[int, Dict[str, List[Any]]]" = OrderedDict(
        (label, {"sizes": [], "formats": [], "bytes": []}) for label in (LABEL_REAL, LABEL_AIGC)
    )
    for record in chosen:
        path = record.resolve_path(root)
        if not os.path.exists(path):
            continue
        bucket = out.setdefault(record.label, {"sizes": [], "formats": [], "bytes": []})
        try:
            with Image.open(path) as img:  # header only; pixels are not decoded
                bucket["sizes"].append(img.size)
                bucket["formats"].append((img.format or "?").upper())
        except Exception:
            # A file we cannot even probe is a data problem, not a shortcut one;
            # validate_splits(check_paths_exist=True) is the right place for it.
            continue
        try:
            bucket["bytes"].append(os.path.getsize(path))
        except OSError:
            pass
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def resolution_shortcut(
    probe: Mapping[int, Mapping[str, Sequence[Any]]],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> List[ShortcutFinding]:
    """Do the two classes differ systematically in image dimensions?"""
    findings: List[ShortcutFinding] = []
    real_sizes = list(probe.get(LABEL_REAL, {}).get("sizes", []))
    aigc_sizes = list(probe.get(LABEL_AIGC, {}).get("sizes", []))
    if not real_sizes or not aigc_sizes:
        return findings

    detail = {
        "real": {
            "n": len(real_sizes),
            "median_wh": (_median([w for w, _ in real_sizes]), _median([h for _, h in real_sizes])),
            "distinct": len(set(real_sizes)),
            "most_common": Counter(real_sizes).most_common(3),
        },
        "aigc": {
            "n": len(aigc_sizes),
            "median_wh": (_median([w for w, _ in aigc_sizes]), _median([h for _, h in aigc_sizes])),
            "distinct": len(set(aigc_sizes)),
            "most_common": Counter(aigc_sizes).most_common(3),
        },
    }

    # Disjoint size vocabularies are the strongest form of this shortcut.
    real_set, aigc_set = set(real_sizes), set(aigc_sizes)
    overlap = real_set & aigc_set
    shared_real = sum(1 for s in real_sizes if s in overlap) / len(real_sizes)
    shared_aigc = sum(1 for s in aigc_sizes if s in overlap) / len(aigc_sizes)
    if not overlap:
        findings.append(
            ShortcutFinding(
                "resolution",
                "critical",
                "real and AIGC images share no image size at all: resolution "
                "alone separates the classes",
                detail,
            )
        )
    elif min(shared_real, shared_aigc) < 1.0 - dominance_threshold:
        findings.append(
            ShortcutFinding(
                "resolution",
                "warning",
                "image sizes barely overlap between classes (%.0f%% of real and "
                "%.0f%% of AIGC use a shared size)"
                % (100 * shared_real, 100 * shared_aigc),
                detail,
            )
        )
    else:
        median_real = detail["real"]["median_wh"]
        median_aigc = detail["aigc"]["median_wh"]
        ratio = max(
            (median_aigc[0] or 1) / (median_real[0] or 1),
            (median_real[0] or 1) / (median_aigc[0] or 1),
        )
        if ratio >= 2.0:
            findings.append(
                ShortcutFinding(
                    "resolution",
                    "warning",
                    "median width differs by %.1fx between classes (real %s vs AIGC %s)"
                    % (ratio, median_real, median_aigc),
                    detail,
                )
            )
        else:
            findings.append(
                ShortcutFinding("resolution", "info", "image sizes overlap across classes", detail)
            )
    return findings


def encoding_shortcut(
    probe: Mapping[int, Mapping[str, Sequence[Any]]],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> List[ShortcutFinding]:
    """Is one class mostly PNG and the other mostly JPEG?

    A decisive difference lets the model read compression history instead of
    generation evidence -- and it disappears the moment images are reposted.
    """
    findings: List[ShortcutFinding] = []
    real_formats = list(probe.get(LABEL_REAL, {}).get("formats", []))
    aigc_formats = list(probe.get(LABEL_AIGC, {}).get("formats", []))
    if not real_formats or not aigc_formats:
        return findings

    real_counts, aigc_counts = Counter(real_formats), Counter(aigc_formats)
    real_top, real_n = real_counts.most_common(1)[0]
    aigc_top, aigc_n = aigc_counts.most_common(1)[0]
    real_share = real_n / len(real_formats)
    aigc_share = aigc_n / len(aigc_formats)
    detail = {"real": dict(real_counts), "aigc": dict(aigc_counts)}

    if (
        real_top != aigc_top
        and real_share >= dominance_threshold
        and aigc_share >= dominance_threshold
    ):
        findings.append(
            ShortcutFinding(
                "encoding",
                "critical",
                "file format almost perfectly separates the classes: real is %.0f%% "
                "%s while AIGC is %.0f%% %s"
                % (100 * real_share, real_top, 100 * aigc_share, aigc_top),
                detail,
            )
        )
    elif real_top != aigc_top:
        findings.append(
            ShortcutFinding(
                "encoding",
                "warning",
                "the dominant file format differs by class (real %s, AIGC %s)"
                % (real_top, aigc_top),
                detail,
            )
        )
    else:
        findings.append(
            ShortcutFinding("encoding", "info", "both classes share the dominant format", detail)
        )
    return findings


def _filesize_shortcut(probe: Mapping[int, Mapping[str, Sequence[Any]]]) -> List[ShortcutFinding]:
    real_bytes = list(probe.get(LABEL_REAL, {}).get("bytes", []))
    aigc_bytes = list(probe.get(LABEL_AIGC, {}).get("bytes", []))
    if not real_bytes or not aigc_bytes:
        return []
    median_real, median_aigc = _median(real_bytes), _median(aigc_bytes)
    if not median_real or not median_aigc:
        return []
    ratio = max(median_real / median_aigc, median_aigc / median_real)
    detail = {"median_bytes_real": median_real, "median_bytes_aigc": median_aigc, "ratio": ratio}
    if ratio >= 3.0:
        return [
            ShortcutFinding(
                "file_size",
                "warning",
                "median file size differs by %.1fx between classes (%d vs %d bytes)"
                % (ratio, median_real, median_aigc),
                detail,
            )
        ]
    return [ShortcutFinding("file_size", "info", "file sizes are comparable across classes", detail)]


# --------------------------------------------------------------------------
# generator concentration
# --------------------------------------------------------------------------
def generator_concentration(
    records: Sequence[ManifestRecord],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> List[ShortcutFinding]:
    """Is the AIGC class effectively a single generator?

    Cross-generator generalization cannot be claimed -- or measured -- from a
    corpus dominated by one model.
    """
    aigc = [r for r in records if r.label == LABEL_AIGC]
    if not aigc:
        return []
    counts = Counter(r.generator or "?" for r in aigc)
    top, n = counts.most_common(1)[0]
    share = n / len(aigc)
    detail = {"generators": dict(counts), "n_aigc": len(aigc)}

    if len(counts) == 1:
        return [
            ShortcutFinding(
                "generator_concentration",
                "warning",
                "all AIGC images come from a single generator %r; unseen-generator "
                "generalization cannot be evaluated" % top,
                detail,
            )
        ]
    if share >= dominance_threshold:
        return [
            ShortcutFinding(
                "generator_concentration",
                "warning",
                "%.0f%% of AIGC images come from one generator %r" % (100 * share, top),
                detail,
            )
        ]
    return [
        ShortcutFinding(
            "generator_concentration",
            "info",
            "%d generators present, largest share %.0f%%" % (len(counts), 100 * share),
            detail,
        )
    ]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def audit_shortcuts(
    records: Iterable[ManifestRecord],
    root: Optional[str] = None,
    inspect_files: bool = True,
    sample_size: Optional[int] = 400,
    seed: int = 0,
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    raise_on_critical: bool = False,
) -> Dict[str, Any]:
    """Audit a record collection for dataset-shortcut risk (rule 11.C).

    Parameters
    ----------
    inspect_files:
        Open image headers to check resolution/format/size.  Set ``False`` for a
        metadata-only audit when the files are not present.
    sample_size:
        Cap on files probed (default 400) -- the distributions are stable well
        before that and the audit stays fast on large manifests.
    raise_on_critical:
        Raise ``AssertionError`` when a ``critical`` finding appears.  Off by
        default: this is advisory analysis, and some skew is expected at
        hackathon scale.  What matters is that it is reported, not hidden.

    Returns
    -------
    ``{"findings": [...], "n_records": int, "counts": {...}, "worst_severity": str}``
    """
    records = list(records)
    if not records:
        raise ValueError("cannot audit an empty record collection")

    findings: List[ShortcutFinding] = []
    findings.extend(provenance_shortcut(records, purity_threshold=purity_threshold))
    findings.extend(generator_concentration(records, dominance_threshold=dominance_threshold))

    if inspect_files:
        probe = _probe(records, root, sample_size, seed)
        findings.extend(resolution_shortcut(probe, dominance_threshold=dominance_threshold))
        findings.extend(encoding_shortcut(probe, dominance_threshold=dominance_threshold))
        findings.extend(_filesize_shortcut(probe))

    order = {"info": 0, "warning": 1, "critical": 2}
    worst = max((f.severity for f in findings), key=lambda s: order[s], default="info")
    counts = Counter(r.label for r in records)
    result = {
        "n_records": len(records),
        "counts": {"real": counts.get(LABEL_REAL, 0), "aigc": counts.get(LABEL_AIGC, 0)},
        "findings": [f.to_dict() for f in findings],
        "worst_severity": worst,
        "n_critical": sum(1 for f in findings if f.severity == "critical"),
    }
    if raise_on_critical and result["n_critical"]:
        raise AssertionError(format_audit_report(result))
    return result


def format_audit_report(report: Mapping[str, Any]) -> str:
    """Human-readable rendering of :func:`audit_shortcuts`."""
    marks = {"info": "ok  ", "warning": "WARN", "critical": "CRIT"}
    lines = [
        "shortcut audit: %d records (real=%d, aigc=%d), worst=%s"
        % (
            report["n_records"],
            report["counts"]["real"],
            report["counts"]["aigc"],
            report["worst_severity"].upper(),
        )
    ]
    for finding in report["findings"]:
        lines.append("  [%s] %-24s %s" % (marks[finding["severity"]], finding["kind"], finding["summary"]))
    return "\n".join(lines)
