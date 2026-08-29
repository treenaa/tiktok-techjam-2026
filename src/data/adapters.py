"""Dataset adapters: on-disk layouts -> :class:`ManifestRecord` lists.

Every adapter is a thin, configurable wrapper around :func:`from_folder`.  The
generic path handles any dataset that separates real from AIGC images by
directory name; the named adapters only encode layout conventions and label
maps, so adding a dataset is usually a few lines here (or a call to
``from_folder`` with a custom ``class_map``).

Adapters never load pixels -- they only walk the filesystem.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import LABEL_AIGC, LABEL_REAL, DataError, ManifestRecord
from .source_id import SourceIdFn, make_source_id_fn

__all__ = [
    "IMAGE_EXTENSIONS",
    "DEFAULT_CLASS_MAP",
    "from_folder",
    "from_class_folders",
    "cifake_adapter",
    "sid_set_adapter",
    "wildfake_adapter",
    "ADAPTERS",
    "build_manifest",
    "list_adapters",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm")

#: Directory-name aliases -> binary label.  Matched case-insensitively against
#: each path segment.  Extend per dataset via ``class_map``.
DEFAULT_CLASS_MAP: Dict[str, int] = {
    "real": LABEL_REAL,
    "reals": LABEL_REAL,
    "0_real": LABEL_REAL,
    "nature": LABEL_REAL,
    "authentic": LABEL_REAL,
    "pristine": LABEL_REAL,
    "genuine": LABEL_REAL,
    "fake": LABEL_AIGC,
    "fakes": LABEL_AIGC,
    "1_fake": LABEL_AIGC,
    "ai": LABEL_AIGC,
    "aigc": LABEL_AIGC,
    "synthetic": LABEL_AIGC,
    "synthesis": LABEL_AIGC,
    "generated": LABEL_AIGC,
    "gan": LABEL_AIGC,
    "diffusion": LABEL_AIGC,
}


def _is_image(name: str, extensions: Sequence[str]) -> bool:
    return os.path.splitext(name)[1].lower() in extensions


def iter_image_paths(
    root: str, extensions: Sequence[str] = IMAGE_EXTENSIONS, follow_links: bool = False
) -> List[str]:
    """All image files under ``root``, sorted for determinism."""
    if not os.path.isdir(root):
        raise DataError("not a directory: %s" % root)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_links):
        dirnames.sort()
        for name in sorted(filenames):
            if _is_image(name, extensions):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _label_from_path(
    path: str, root: str, class_map: Mapping[str, int]
) -> Optional[int]:
    """Label from the *deepest* matching directory segment.

    Deepest-first means ``fake/stylegan/real_faces/...`` still resolves through
    the more specific segment rather than an accidental ancestor match.
    """
    rel = os.path.relpath(path, root)
    segments = rel.replace(os.sep, "/").split("/")[:-1]
    for segment in reversed(segments):
        key = segment.strip().lower()
        if key in class_map:
            return class_map[key]
    return None


def _dataset_rooted_source_id_fn(
    root: str, dataset: str, kwargs: Dict[str, Any]
) -> None:
    """Pin ``source_id`` derivation to the dataset root.

    Adapters that walk one split directory at a time would otherwise derive
    relative paths from the *split* directory, making ``train/REAL/0000`` and
    ``test/REAL/0000`` collide into a single source_id -- silently fusing two
    unrelated images (and defeating the split checks).  Mutates ``kwargs``.
    """
    if kwargs.get("source_id_fn") is not None:
        return
    policy = kwargs.pop("source_id_policy", "relpath")
    prefix = dataset if kwargs.get("namespace_source_ids", True) and dataset else ""
    kwargs["source_id_fn"] = make_source_id_fn(policy, root=root, prefix=prefix)


def _generator_from_path(
    path: str, root: str, label: int, depth: Optional[int], class_map: Mapping[str, int]
) -> str:
    """Directory segment naming the generator, for AIGC samples only."""
    if label != LABEL_AIGC or depth is None:
        return ""
    rel = os.path.relpath(path, root)
    segments = [s for s in rel.replace(os.sep, "/").split("/")[:-1] if s]
    # Drop leading split/class folders so ``depth`` counts from the class dir.
    for i, seg in enumerate(segments):
        if seg.strip().lower() in class_map:
            segments = segments[i + 1 :]
            break
    if len(segments) > depth:
        return segments[depth]
    return segments[-1] if segments else ""


def from_folder(
    root: str,
    class_map: Optional[Mapping[str, int]] = None,
    dataset: str = "",
    source_id_fn: Optional[SourceIdFn] = None,
    source_id_policy: str = "relpath",
    namespace_source_ids: bool = True,
    generator_depth: Optional[int] = None,
    generator: str = "",
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    split: str = "",
    label: Optional[int] = None,
    on_unlabelled: str = "raise",
    extra: Optional[Mapping[str, Any]] = None,
) -> List[ManifestRecord]:
    """Build records by walking ``root`` and labelling from directory names.

    Parameters
    ----------
    class_map:
        ``{directory_name: label}``, merged over :data:`DEFAULT_CLASS_MAP`.
        Matched case-insensitively against every path segment, deepest first.
    label:
        Fixed label for *everything* under ``root``, bypassing ``class_map``.
        Use for datasets whose class is implied by the directory you point at
        rather than by a directory name.
    source_id_fn:
        Custom ``path -> source_id``.  Defaults to
        :func:`~src.data.source_id.make_source_id_fn` with ``source_id_policy``
        (``"relpath"`` by default, since filenames often repeat across class
        folders).  Transform suffixes are stripped, so ``cat_jpeg70.png`` and
        ``cat.png`` share an id.
    namespace_source_ids:
        Prefix source ids with ``dataset`` so pooled datasets cannot collide.
    generator_depth:
        Index of the path segment (counted *after* the class folder) that names
        the generator, e.g. ``fake/stylegan2/img.png`` -> ``generator_depth=0``.
    on_unlabelled:
        ``"raise"`` | ``"skip"`` -- what to do with images whose directory does
        not match ``class_map``.

    Notes
    -----
    Real images keep ``generator=""``; a source_id that spans both labels is
    allowed (a real image and its tampered derivative), and the splitter keeps
    such a group intact.
    """
    root = os.path.abspath(os.path.expanduser(root))
    merged: Dict[str, int] = dict(DEFAULT_CLASS_MAP)
    if class_map:
        merged.update({str(k).strip().lower(): int(v) for k, v in class_map.items()})

    if source_id_fn is None:
        source_id_fn = make_source_id_fn(
            source_id_policy,
            root=root,
            prefix=dataset if (namespace_source_ids and dataset) else "",
        )

    paths = iter_image_paths(root, extensions)
    if not paths:
        raise DataError(
            "no images with extensions %s found under %s" % (list(extensions), root)
        )

    if label is not None and int(label) not in (LABEL_REAL, LABEL_AIGC):
        raise DataError("label override must be 0 or 1, got %r" % (label,))

    records: List[ManifestRecord] = []
    unlabelled: List[str] = []
    for path in paths:
        image_label = int(label) if label is not None else _label_from_path(path, root, merged)
        if image_label is None:
            unlabelled.append(path)
            continue
        gen = generator or _generator_from_path(
            path, root, image_label, generator_depth, merged
        )
        records.append(
            ManifestRecord(
                image_path=path,
                label=image_label,
                source_id=source_id_fn(path),
                dataset=dataset,
                generator=gen,
                split=split,
                extra=dict(extra) if extra else {},
            )
        )

    if unlabelled:
        if on_unlabelled == "raise":
            raise DataError(
                "%d image(s) under %s are in directories not covered by class_map "
                "(e.g. %s); pass class_map=... or on_unlabelled='skip'"
                % (len(unlabelled), root, unlabelled[:3])
            )
        if on_unlabelled != "skip":
            raise ValueError("on_unlabelled must be 'raise' or 'skip'")
    if not records:
        raise DataError("no labelled images found under %s" % root)
    return records


def from_class_folders(
    real_dir: str,
    aigc_dir: str,
    dataset: str = "",
    **kwargs: Any,
) -> List[ManifestRecord]:
    """Two explicit directories, one per class -- for layouts the class_map misses.

    The label comes from which argument a directory was passed as, not from its
    name, so the directories can be called anything.
    """
    out: List[ManifestRecord] = []
    for directory, label in ((real_dir, LABEL_REAL), (aigc_dir, LABEL_AIGC)):
        out.extend(from_folder(directory, label=label, dataset=dataset, **kwargs))
    return out


# --------------------------------------------------------------------------
# Named adapters
# --------------------------------------------------------------------------
def cifake_adapter(
    root: str,
    dataset: str = "cifake",
    split_dirs: Mapping[str, str] = None,
    **kwargs: Any,
) -> List[ManifestRecord]:
    """CIFAKE: ``<root>/{train,test}/{REAL,FAKE}/*.jpg`` (32x32).

    CIFAKE ships its own train/test dirs; those are recorded in ``split`` but
    the leakage-safe splitter may still re-split them.  Filenames repeat between
    REAL and FAKE (``0001.jpg``), so source ids use the relative path.
    """
    root = os.path.abspath(os.path.expanduser(root))
    split_dirs = split_dirs or {"train": "train", "test": "test"}
    kwargs.setdefault("class_map", {"real": LABEL_REAL, "fake": LABEL_AIGC})
    kwargs.setdefault("generator", "")

    present = {s: os.path.join(root, d) for s, d in split_dirs.items() if os.path.isdir(os.path.join(root, d))}
    if present:
        _dataset_rooted_source_id_fn(root, dataset, kwargs)
    if not present:
        # Flat REAL/FAKE layout without split folders.
        return from_folder(root, dataset=dataset, **kwargs)
    out: List[ManifestRecord] = []
    for split_name, path in present.items():
        out.extend(from_folder(path, dataset=dataset, split=split_name, **kwargs))
    return out


#: SID_Set is 3-class (real / synthetic / tampered).  For the binary task both
#: synthetic and tampered are AIGC.  Override to change that.
SID_SET_CLASS_MAP: Dict[str, int] = {
    "real": LABEL_REAL,
    "full_synthetic": LABEL_AIGC,
    "fully_synthetic": LABEL_AIGC,
    "synthetic": LABEL_AIGC,
    "tampered": LABEL_AIGC,
    "tamper": LABEL_AIGC,
    "edited": LABEL_AIGC,
}


def sid_set_adapter(
    root: str,
    dataset: str = "sid_set",
    class_map: Optional[Mapping[str, int]] = None,
    tampered_share_real_source_id: bool = False,
    **kwargs: Any,
) -> List[ManifestRecord]:
    """SID_Set: ``<root>/[split/]{real,synthetic,tampered}/...``.

    ``tampered_share_real_source_id`` strips a ``tampered_``/``_tampered``
    marker from the stem so a tampered image inherits the source id of the real
    image it was edited from -- keeping the pair on one side of the split.  Off
    by default because it depends on the export's naming.
    """
    root = os.path.abspath(os.path.expanduser(root))
    merged = dict(SID_SET_CLASS_MAP)
    if class_map:
        merged.update(class_map)
    kwargs.setdefault("class_map", merged)
    if tampered_share_real_source_id and "source_id_fn" not in kwargs:
        kwargs["source_id_fn"] = make_source_id_fn(
            "stem",
            root=root,
            prefix=dataset,
            tokens=("tampered", "tamper", "edit(ed)?", "manip(ulated)?", "mask"),
        )
    kwargs.setdefault("extra", {"task": "sid_set_binary"})

    split_dirs = [d for d in ("train", "val", "validation", "test") if os.path.isdir(os.path.join(root, d))]
    if not split_dirs:
        return from_folder(root, dataset=dataset, **kwargs)
    _dataset_rooted_source_id_fn(root, dataset, kwargs)
    out: List[ManifestRecord] = []
    for name in split_dirs:
        split_name = "val" if name in ("val", "validation") else name
        out.extend(
            from_folder(os.path.join(root, name), dataset=dataset, split=split_name, **kwargs)
        )
    return out


def wildfake_adapter(
    root: str,
    dataset: str = "wildfake",
    generator_depth: int = 0,
    **kwargs: Any,
) -> List[ManifestRecord]:
    """WildFake: ``<root>/{real,fake}/<architecture>/<model>/...``.

    The first directory below the class folder is recorded as ``generator``
    (``generator_depth=1`` to use the second, i.e. the concrete model).
    """
    kwargs.setdefault("generator_depth", generator_depth)
    kwargs.setdefault("class_map", {"real": LABEL_REAL, "fake": LABEL_AIGC})
    return from_folder(root, dataset=dataset, **kwargs)


#: ``name -> adapter``.  ``"folder"`` is the generic escape hatch.
ADAPTERS: Dict[str, Callable[..., List[ManifestRecord]]] = {
    "folder": from_folder,
    "cifake": cifake_adapter,
    "sid_set": sid_set_adapter,
    "sidset": sid_set_adapter,
    "wildfake": wildfake_adapter,
}


def list_adapters() -> List[str]:
    return sorted(ADAPTERS)


def build_manifest(name: str, root: str, **kwargs: Any) -> List[ManifestRecord]:
    """Dispatch to a registered adapter by name."""
    try:
        adapter = ADAPTERS[name.lower()]
    except KeyError:
        raise KeyError(
            "unknown dataset adapter %r; available: %s" % (name, list_adapters())
        ) from None
    return adapter(root, **kwargs)
