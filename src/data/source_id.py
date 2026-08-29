"""Derivation of stable ``source_id`` values.

A ``source_id`` identifies the *underlying original image*.  Two files must get
the same ``source_id`` when one is a transformed derivative of the other
(``cat.png`` and ``cat_jpeg70.png``), otherwise the leakage-safe splitter cannot
keep them on the same side of a split.

Policies are plain ``str -> str`` callables so callers can supply their own.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, List, Optional, Sequence

SourceIdFn = Callable[[str], str]

#: Suffix tokens produced by :mod:`src.data.transforms` and by common dataset
#: dumps.  Stripped from the filename stem by :func:`strip_transform_suffixes`.
DEFAULT_TRANSFORM_TOKENS: Sequence[str] = (
    r"jpe?g_?q?\d{1,3}",
    r"blur_?(sigma_?)?\d+(\.\d+)?",
    r"resize_?\d*(\.\d+)?x?",
    r"rescale_?\d*(\.\d+)?x?",
    r"noise_?(sigma_?)?\d+(\.\d+)?",
    r"jitter(_[a-z]+)*(_(up|down))?",
    r"bright(ness)?_?\d*(\.\d+)?(_(up|down))?",
    r"contrast_?\d*(\.\d+)?(_(up|down))?",
    r"sat(uration)?_?\d*(\.\d+)?(_(up|down))?",
    r"crop_?\d*(\.\d+)?",
    r"centercrop_?\d*(\.\d+)?",
    r"aug\d*",
    r"clean",
    r"orig(inal)?",
    r"copy\d*",
    r"v\d+",
)

_SEP = r"[_\-.]"


def _compile(tokens: Iterable[str]) -> re.Pattern:
    alt = "|".join("(?:%s)" % t for t in tokens)
    # One or more trailing transform tokens, each preceded by a separator.
    return re.compile(r"(?:%s(?:%s))+$" % (_SEP, alt), re.IGNORECASE)


_DEFAULT_RE = _compile(DEFAULT_TRANSFORM_TOKENS)


def stem_source_id(path: str) -> str:
    """Filename without directory or extension.

    Cheap and adequate when filenames are globally unique (CIFAKE, WildFake).
    """
    return os.path.splitext(os.path.basename(path))[0]


def relpath_source_id(path: str, root: Optional[str] = None) -> str:
    """Path relative to ``root``, extension dropped, ``/``-normalised.

    Use when filenames repeat across class/generator folders.
    """
    p = os.path.relpath(path, root) if root else path
    p = os.path.splitext(p)[0]
    return p.replace(os.sep, "/")


def strip_transform_suffixes(
    stem: str, tokens: Optional[Sequence[str]] = None
) -> str:
    """Remove trailing transform markers from a filename stem.

    ``cat_017_jpeg70`` -> ``cat_017``; ``cat_017_blur_1.0_jpeg30`` -> ``cat_017``.
    Never returns an empty string -- if everything would be stripped, the
    original stem is kept.
    """
    pattern = _compile(tokens) if tokens is not None else _DEFAULT_RE
    out = pattern.sub("", stem)
    return out if out else stem


def canonical_source_id(
    path: str,
    root: Optional[str] = None,
    use_relpath: bool = False,
    strip_suffixes: bool = True,
    prefix: str = "",
    tokens: Optional[Sequence[str]] = None,
) -> str:
    """Default policy: (optionally relative) path, transform suffixes stripped.

    ``prefix`` namespaces the id, which matters when several datasets are pooled
    and their filenames could collide.
    """
    if use_relpath:
        base = relpath_source_id(path, root)
        head, stem = os.path.split(base)
    else:
        head, stem = "", stem_source_id(path)
    if strip_suffixes:
        stem = strip_transform_suffixes(stem, tokens)
    sid = "%s/%s" % (head, stem) if head else stem
    return "%s:%s" % (prefix, sid) if prefix else sid


def make_source_id_fn(
    policy: str = "stem",
    root: Optional[str] = None,
    prefix: str = "",
    strip_suffixes: bool = True,
    regex: Optional[str] = None,
    tokens: Optional[Sequence[str]] = None,
) -> SourceIdFn:
    """Build a ``path -> source_id`` callable.

    Policies
    --------
    ``"stem"``
        Filename stem (default).
    ``"relpath"``
        Path relative to ``root``, keeps directory structure.
    ``"parent"``
        Parent directory name -- for datasets that store all views of one image
        in a per-image folder.
    ``"regex"``
        First capture group of ``regex`` applied to the path.
    """
    if policy == "regex":
        if not regex:
            raise ValueError("policy='regex' requires a regex")
        compiled = re.compile(regex)

        def _regex_fn(path: str) -> str:
            m = compiled.search(path.replace(os.sep, "/"))
            if not m:
                raise ValueError("source_id regex %r did not match %r" % (regex, path))
            sid = m.group(1) if m.groups() else m.group(0)
            return "%s:%s" % (prefix, sid) if prefix else sid

        return _regex_fn

    if policy == "parent":
        def _parent_fn(path: str) -> str:
            sid = os.path.basename(os.path.dirname(path))
            return "%s:%s" % (prefix, sid) if prefix else sid

        return _parent_fn

    if policy not in ("stem", "relpath"):
        raise ValueError("unknown source_id policy %r" % policy)

    use_relpath = policy == "relpath"

    def _fn(path: str) -> str:
        return canonical_source_id(
            path,
            root=root,
            use_relpath=use_relpath,
            strip_suffixes=strip_suffixes,
            prefix=prefix,
            tokens=tokens,
        )

    return _fn


def group_paths_by_source_id(paths: Iterable[str], fn: SourceIdFn) -> dict:
    """Debug helper: ``{source_id: [paths]}``."""
    groups: dict = {}
    for p in paths:
        groups.setdefault(fn(p), []).append(p)
    return groups


def find_source_id_collisions(
    paths: Iterable[str], fn: SourceIdFn, max_report: int = 10
) -> List[tuple]:
    """Source ids mapped from more than one *directory*.

    Collisions across directories usually mean the policy is too coarse (two
    unrelated images share a filename); collisions inside one directory are the
    expected transformed-derivative case.
    """
    groups = group_paths_by_source_id(paths, fn)
    out = []
    for sid, members in groups.items():
        dirs = {os.path.dirname(m) for m in members}
        if len(dirs) > 1:
            out.append((sid, sorted(members)))
        if len(out) >= max_report:
            break
    return out
