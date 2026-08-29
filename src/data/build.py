"""One-call manifest generation from folder layouts.

Wraps the adapters + splitter + writer so producing a standard manifest is a
single call (or a single command line).  Downloads nothing -- point it at data
that already exists on disk.

CLI::

    python -m src.data.build --dataset cifake --root /data/cifake \\
        --out manifests/cifake.csv --split
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters import build_manifest, list_adapters
from .manifest import describe_records, write_manifest
from .schema import ManifestRecord
from .splitting import assign_splits, split_records

__all__ = ["generate_manifest", "generate_split_manifests", "main"]


def generate_manifest(
    root: str,
    dataset: Optional[str] = None,
    adapter: str = "folder",
    out_path: Optional[str] = None,
    relative_to: Optional[str] = None,
    split: bool = False,
    ratios: Any = None,
    seed: int = 0,
    stratify_keys: Optional[Sequence[str]] = ("label",),
    group_keys: Sequence[str] = ("source_id",),
    **adapter_kwargs: Any,
) -> List[ManifestRecord]:
    """Build (and optionally write) a standard manifest for a folder tree.

    Parameters
    ----------
    root:
        Dataset directory.
    dataset:
        Value for the ``dataset`` column; defaults to the adapter name.
    adapter:
        ``"folder"`` (generic), ``"cifake"``, ``"sid_set"``, ``"wildfake"`` --
        see :func:`~src.data.adapters.list_adapters`.
    out_path:
        Where to write ``image_path,label,source_id,dataset,generator``.
        ``None`` builds the records without writing.
    relative_to:
        Store paths relative to this directory (defaults to ``root`` when an
        ``out_path`` is given, keeping manifests portable).
    split:
        Also assign ``train``/``val``/``test`` into the ``split`` column.
    """
    dataset = dataset if dataset is not None else adapter
    records = build_manifest(adapter, root, dataset=dataset, **adapter_kwargs)
    if split:
        records = assign_splits(
            records,
            ratios=ratios,
            seed=seed,
            stratify_keys=stratify_keys,
            group_keys=group_keys,
        )
    if out_path:
        write_manifest(
            records,
            out_path,
            relative_to=relative_to if relative_to is not None else root,
        )
    return records


def generate_split_manifests(
    root: str,
    out_dir: str,
    dataset: Optional[str] = None,
    adapter: str = "folder",
    ratios: Any = None,
    seed: int = 0,
    stratify_keys: Optional[Sequence[str]] = ("label",),
    group_keys: Sequence[str] = ("source_id",),
    prefix: str = "",
    fmt: str = "csv",
    **adapter_kwargs: Any,
) -> "OrderedDict[str, str]":
    """Write one manifest file per split; returns ``{split: path}``.

    The split is leakage-safe by ``source_id`` and verified before writing.
    """
    dataset = dataset if dataset is not None else adapter
    records = build_manifest(adapter, root, dataset=dataset, **adapter_kwargs)
    splits = split_records(
        records,
        ratios=ratios,
        seed=seed,
        stratify_keys=stratify_keys,
        group_keys=group_keys,
        verify=True,
    )
    os.makedirs(out_dir, exist_ok=True)
    paths: "OrderedDict[str, str]" = OrderedDict()
    for name, members in splits.items():
        filename = "%s%s.%s" % (prefix, name, fmt.lstrip("."))
        path = os.path.join(out_dir, filename)
        write_manifest(
            [rec.with_fields(split=name) for rec in members], path, relative_to=root
        )
        paths[name] = path
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.build",
        description="Build a standard AIGC-detection manifest from a folder tree.",
    )
    parser.add_argument("--root", required=True, help="dataset directory")
    parser.add_argument(
        "--adapter", default="folder", choices=list_adapters(), help="layout adapter"
    )
    parser.add_argument("--dataset", default=None, help="value for the dataset column")
    parser.add_argument("--out", default=None, help="output manifest path")
    parser.add_argument("--out-dir", default=None, help="write one manifest per split here")
    parser.add_argument("--split", action="store_true", help="assign a split column")
    parser.add_argument(
        "--ratios", default=None, help="train,val,test e.g. 0.7,0.15,0.15"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-stratify", action="store_true", help="disable label stratification"
    )
    parser.add_argument(
        "--skip-unlabelled",
        action="store_true",
        help="ignore images in directories the class map does not cover",
    )
    args = parser.parse_args(argv)

    ratios = [float(x) for x in args.ratios.split(",")] if args.ratios else None
    stratify = None if args.no_stratify else ("label",)
    extra: Dict[str, Any] = {}
    if args.skip_unlabelled:
        extra["on_unlabelled"] = "skip"

    if args.out_dir:
        paths = generate_split_manifests(
            args.root,
            args.out_dir,
            dataset=args.dataset,
            adapter=args.adapter,
            ratios=ratios,
            seed=args.seed,
            stratify_keys=stratify,
            **extra
        )
        for name, path in paths.items():
            print("%-6s -> %s" % (name, path))
        return 0

    records = generate_manifest(
        args.root,
        dataset=args.dataset,
        adapter=args.adapter,
        out_path=args.out,
        split=args.split,
        ratios=ratios,
        seed=args.seed,
        stratify_keys=stratify,
        **extra
    )
    info = describe_records(records)
    print(
        "%d images | %d source_ids | real=%d aigc=%d | datasets=%s"
        % (
            info["n_images"],
            info["n_source_ids"],
            info["n_real"],
            info["n_aigc"],
            sorted(info["datasets"]),
        )
    )
    if args.out:
        print("written: %s" % args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
