#!/usr/bin/env python
"""Fetch SID_Set from the Hugging Face Hub and export it into the layout
``src.data.adapters.sid_set_adapter`` expects.

    python scripts/download_sid_set.py --shards 3          # ~3 GB smoke slice
    python scripts/download_sid_set.py --all               # full ~140 GB (GPU box)
    python scripts/download_sid_set.py --shards 3 --download-only
    python scripts/download_sid_set.py --export-only

The Hub ships 283 parquet shards (249 train / 34 validation, ~0.49 GB each)
whose ``image`` and ``mask`` columns hold encoded bytes.  The adapter wants
``<root>/<split>/{real,full_synthetic,tampered}/<img_id>.<ext>``, so this script
walks the parquet row groups and writes the *original* image bytes straight to
disk -- no decode/re-encode, so nothing is recompressed.

Masks mark the tampered regions but are not training samples, so they go to a
sibling tree (``--masks-dir``).  Anything under the adapter root that is not a
class folder would trip its ``on_unlabelled="raise"`` guard.

Exit codes follow ``src.data.audit_cli``: ``0`` clean, ``1`` a blocking
problem, ``2`` bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_OK = 0
_FAIL = 1
_USAGE = 2

REPO_ID = "saberzl/SID_Set"
DEFAULT_RAW_DIR = os.path.join("data", "raw", "SID_Set")
DEFAULT_OUT_DIR = os.path.join("data", "SID_set")
DEFAULT_MASKS_DIR = os.path.join("data", "SID_set_masks")

#: SID_Set's integer ``label`` column -> the class folder the adapter reads.
#: 1 and 2 both collapse to the binary AIGC label via ``SID_SET_CLASS_MAP``;
#: keeping them apart on disk preserves the distinction for later analysis.
LABEL_DIRS: Dict[int, str] = {0: "real", 1: "full_synthetic", 2: "tampered"}

#: Hub split name -> the directory name ``sid_set_adapter`` normalises to "val".
SPLIT_DIRS: Dict[str, str] = {"train": "train", "validation": "val"}

#: Magic-byte prefixes, so an exported file gets the extension matching the
#: bytes actually stored rather than an assumed one.
_MAGIC = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=3,
        help="shards to take from EACH split (default 3, ~0.49 GB each)",
    )
    parser.add_argument(
        "--all", action="store_true", help="every shard: the full ~140 GB dataset"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=sorted(SPLIT_DIRS),
        help="which Hub splits to pull (default: both)",
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="parquet download dir")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="exported image root")
    parser.add_argument(
        "--masks-dir",
        default=DEFAULT_MASKS_DIR,
        help="where tampered-region masks go; must stay outside --out",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="fetch parquet, skip the export"
    )
    parser.add_argument(
        "--export-only", action="store_true", help="export parquet already on disk"
    )
    parser.add_argument(
        "--no-masks", action="store_true", help="skip mask export (saves space)"
    )
    parser.add_argument(
        "--drop-parquet",
        action="store_true",
        help="delete each shard once exported, to reclaim space on a small disk",
    )
    parser.add_argument(
        "--report",
        default=os.path.join("reports", "sid_set_export.json"),
        help="where to write the export summary",
    )
    return parser


def _extension(payload: bytes, fallback: str = ".jpg") -> str:
    """Extension implied by ``payload``'s magic bytes."""
    for prefix, ext in _MAGIC:
        if payload.startswith(prefix):
            return ext
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return ".webp"
    return fallback


def _shard_names(split: str, limit: Optional[int]) -> List[str]:
    """Sorted repo paths of ``split``'s parquet shards, capped at ``limit``."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    shards = sorted(
        f for f in files if f.startswith("data/%s-" % split) and f.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError("no %s shards found in %s" % (split, REPO_ID))
    return shards if limit is None else shards[:limit]


def download_shards(names: Sequence[str], raw_dir: str) -> List[str]:
    """Download ``names`` into ``raw_dir``, reusing whatever is already there."""
    from huggingface_hub import hf_hub_download

    out: List[str] = []
    for index, name in enumerate(names, 1):
        print("  [%d/%d] %s" % (index, len(names), name), flush=True)
        out.append(hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=raw_dir))
    return out


def local_shards(raw_dir: str, split: str, limit: Optional[int]) -> List[str]:
    """Parquet shards for ``split`` already downloaded under ``raw_dir``."""
    directory = os.path.join(raw_dir, "data")
    if not os.path.isdir(directory):
        return []
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("%s-" % split) and name.endswith(".parquet")
    )
    return paths if limit is None else paths[:limit]


def _payload(cell: object) -> Optional[bytes]:
    """Bytes out of a Hub ``Image`` cell, which arrives as ``{bytes, path}``."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        return bytes(raw) if raw else None
    return None


def _unique(target: str, counts: Dict[str, int]) -> str:
    """``target``, or the next free ``__N`` variant if it is already taken.

    ``img_id`` repeats across classes by design -- a tampered image keeps the
    id of the real image it was edited from -- and those land in different
    class folders.  A repeat *within* one folder is a genuine duplicate row, so
    keep both rather than silently dropping one.
    """
    if not os.path.exists(target):
        return target
    base, ext = os.path.splitext(target)
    suffix = 1
    while os.path.exists("%s__%d%s" % (base, suffix, ext)):
        suffix += 1
    counts["duplicate_ids"] += 1
    return "%s__%d%s" % (base, suffix, ext)


def export_shard(
    path: str,
    split_dir: str,
    out_root: str,
    masks_root: Optional[str],
    counts: Dict[str, int],
) -> None:
    """Write one parquet shard's images (and masks) out as files."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["img_id", "image", "label"]
    if masks_root is not None and "mask" in parquet.schema_arrow.names:
        columns.append("mask")

    for group in range(parquet.num_row_groups):
        for row in parquet.read_row_group(group, columns=columns).to_pylist():
            payload = _payload(row.get("image"))
            label = row.get("label")
            if payload is None or label not in LABEL_DIRS:
                counts["skipped"] += 1
                continue
            class_dir = LABEL_DIRS[label]
            stem = str(row.get("img_id") or "").strip() or "unnamed"
            stem = stem.replace("/", "_").replace("\\", "_")

            directory = os.path.join(out_root, split_dir, class_dir)
            os.makedirs(directory, exist_ok=True)
            target = _unique(os.path.join(directory, stem + _extension(payload)), counts)
            with open(target, "wb") as handle:
                handle.write(payload)
            counts[class_dir] += 1
            counts["images"] += 1

            mask = _payload(row.get("mask")) if masks_root is not None else None
            if mask:
                mask_dir = os.path.join(masks_root, split_dir, class_dir)
                os.makedirs(mask_dir, exist_ok=True)
                mask_path = _unique(
                    os.path.join(mask_dir, stem + _extension(mask, ".png")), counts
                )
                with open(mask_path, "wb") as handle:
                    handle.write(mask)
                counts["masks"] += 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.download_only and args.export_only:
        parser.error("--download-only and --export-only are mutually exclusive")
    if args.shards < 1 and not args.all:
        parser.error("--shards must be >= 1 (or pass --all)")

    out_root = os.path.abspath(args.out)
    masks_root = None if args.no_masks else os.path.abspath(args.masks_dir)
    if masks_root is not None and (
        masks_root == out_root or masks_root.startswith(out_root + os.sep)
    ):
        parser.error(
            "--masks-dir must live outside --out: masks are not training samples, "
            "and the adapter raises on unlabelled files under its root"
        )

    limit = None if args.all else args.shards
    counts: Dict[str, int] = {
        "images": 0,
        "masks": 0,
        "skipped": 0,
        "duplicate_ids": 0,
        "real": 0,
        "full_synthetic": 0,
        "tampered": 0,
    }
    shard_total = 0

    for split in args.splits:
        split_dir = SPLIT_DIRS[split]
        if args.export_only:
            paths = local_shards(args.raw_dir, split, limit)
            if not paths:
                print("no local %s shards under %s" % (split, args.raw_dir))
                continue
        else:
            names = _shard_names(split, limit)
            print("%s: downloading %d shard(s)" % (split, len(names)), flush=True)
            paths = download_shards(names, args.raw_dir)

        if args.download_only:
            shard_total += len(paths)
            continue

        print("%s: exporting %d shard(s) -> %s" % (split, len(paths), out_root), flush=True)
        for index, path in enumerate(paths, 1):
            print("  [%d/%d] %s" % (index, len(paths), os.path.basename(path)), flush=True)
            export_shard(path, split_dir, out_root, masks_root, counts)
            shard_total += 1
            if args.drop_parquet:
                os.remove(path)

    summary = {
        "repo_id": REPO_ID,
        "splits": list(args.splits),
        "shards": shard_total,
        "shard_limit_per_split": limit,
        "out": out_root,
        "masks_dir": masks_root,
        "counts": counts,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.download_only and counts["images"] == 0:
        print("no images exported", file=sys.stderr)
        return _FAIL
    return _OK


if __name__ == "__main__":
    sys.exit(main())
