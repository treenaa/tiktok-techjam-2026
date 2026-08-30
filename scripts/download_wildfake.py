#!/usr/bin/env python
"""Fetch WildFake from ModelScope and export it into the layout
``src.data.adapters.wildfake_adapter`` expects.

    python scripts/download_wildfake.py --list                  # inventory + sizes
    python scripts/download_wildfake.py                         # ~9 GB default slice
    python scripts/download_wildfake.py --groups real_small DDIM Other_based
    python scripts/download_wildfake.py --all                   # ~1.29 TB -- see below

The whole dataset is ~1.29 TB, so ``--all`` is only sensible on a machine with
that much free disk; everything is selected by *group* instead, a group being
one or more of the archives ModelScope actually ships.

ModelScope stores the images as large zips whose members look like
``./Diffusion_based/DDIM/imgs_CC9K/<hash>.png``, while ``label_csv_files/``
carries the authoritative ``IsFake`` and ``Architecture`` for every image.  The
adapter wants ``<root>/{real,fake}/<architecture>/<model>/...`` with the first
directory below the class folder naming the generator, so this script joins the
two: the CSVs decide the class and generator folder, the zip supplies bytes.

Exit codes follow ``src.data.audit_cli``: ``0`` clean, ``1`` a blocking
problem, ``2`` bad usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_OK = 0
_FAIL = 1
_USAGE = 2

NAMESPACE = "hy2628982280"
DATASET = "WildFake"
_API = "https://modelscope.cn/api/v1/datasets/%s/%s/repo?Revision=master&FilePath=%%s" % (
    NAMESPACE,
    DATASET,
)

DEFAULT_RAW_DIR = os.path.join("data", "raw", "WildFake")
DEFAULT_OUT_DIR = os.path.join("data", "wildfake")

#: The 34 per-generator label files.  All of them are downloaded regardless of
#: which image groups are selected -- together they are ~380 MB and they are
#: what makes the class/generator mapping authoritative rather than guessed.
LABEL_CSVS = (
    "adm", "BigGAN", "dalle2", "dalle3", "ddim", "ddpm", "DF-GAN", "GALIP",
    "GigaGAN", "imagen", "MAE", "MAGE", "mjv4", "mjv5", "originsd",
    "personalizedSD_dreambooth", "personalizedSD_finetune", "real_afhq",
    "real_celebahq", "real_church", "real_coco", "real_ffhq", "real_imagenet",
    "real_laion5b", "real_wukong", "SDwithAdaptor_controlnet",
    "SDwithAdaptor_lora", "SDwithAdaptor_lycris", "sdxl", "starGAN",
    "styleGAN", "vqdm", "VQGAN", "VQVAE",
)

#: ``group -> (repo paths, approximate GB)``.  Sizes are what the ModelScope
#: tree API reports, so ``--list`` can price a selection before downloading.
GROUPS: Dict[str, Tuple[Tuple[str, ...], float]] = {
    "real_afhq": (("Images/Real/afhq.zip",), 0.45),
    "real_celebahq": (("Images/Real/celebahq.zip",), 0.35),
    "real_church": (("Images/Real/church.zip",), 1.16),
    "real_coco": (("Images/Real/coco.zip",), 2.35),
    "real_ffhq": (("Images/Real/ffhq.zip",), 0.82),
    "real_imagenet": (("Images/Real/imagenet.zip",), 1.38),
    "real_laion5b": (("Images/Real/laion5b.zip",), 24.80),
    "real_wukong": (("Images/Real/wukong.zip",), 0.0),
    "ADM": (("Images/Diffusion_based/ADM.zip",), 18.55),
    "DALLE": (("Images/Diffusion_based/DALLE.zip",), 25.59),
    "DDIM": (("Images/Diffusion_based/DDIM.zip",), 6.05),
    "DDPM": (("Images/Diffusion_based/DDPM.zip",), 8.14),
    "Imagen": (("Images/Diffusion_based/Imagen.zip",), 17.07),
    "VQDM": (("Images/Diffusion_based/VQDM.zip",), 17.38),
    "GAN_based": (("Images/GAN_based.zip",), 47.33),
    "Other_based": (("Images/Other_based.zip",), 13.34),
    "personalizedSD": (("Images/Diffusion_based/SD/personalizedSD.zip",), 48.70),
    "SDwithAdaptor": (("Images/Diffusion_based/SD/SDwithAdaptor.zip",), 42.00),
    "originalSD_typical": (
        tuple(
            "Images/Diffusion_based/SD/originalSD/Typical/part_%d.zip" % i
            for i in range(1, 4)
        ),
        119.28,
    ),
    "originalSD_advanced": (
        tuple(
            "Images/Diffusion_based/SD/originalSD/Advanced/part_%d.zip" % i
            for i in range(1, 8)
        ),
        324.10,
    ),
    "midjourney_typical": (
        tuple(
            "Images/Diffusion_based/Midjourney/Typical/part_%d.zip" % i
            for i in range(1, 5)
        ),
        196.26,
    ),
    "midjourney_advanced": (
        tuple(
            "Images/Diffusion_based/Midjourney/Advanced/part_%d.zip" % i
            for i in range(1, 8)
        ),
        372.42,
    ),
}

#: Convenience aliases for sets of groups.
ALIASES: Dict[str, Tuple[str, ...]] = {
    "real_small": ("real_afhq", "real_celebahq", "real_ffhq", "real_imagenet"),
    "real_all": tuple(g for g in GROUPS if g.startswith("real_")),
}

#: Small real corpora plus the cheapest fake archive: enough to wire up the
#: adapter and check the generator depth without committing to a 1.29 TB pull.
DEFAULT_GROUPS = ALIASES["real_small"] + ("DDIM",)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="archive groups to pull; see --list for names and sizes",
    )
    parser.add_argument("--all", action="store_true", help="every group (~1.29 TB)")
    parser.add_argument(
        "--list", action="store_true", help="print the inventory and exit"
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="download dir")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="exported image root")
    parser.add_argument(
        "--download-only", action="store_true", help="fetch archives, skip the export"
    )
    parser.add_argument(
        "--export-only", action="store_true", help="export archives already on disk"
    )
    parser.add_argument(
        "--drop-zip",
        action="store_true",
        help="delete each archive once exported, to reclaim space on a small disk",
    )
    parser.add_argument(
        "--report",
        default=os.path.join("reports", "wildfake_export.json"),
        help="where to write the export summary",
    )
    return parser


def resolve_groups(names: Iterable[str]) -> List[str]:
    """Expand aliases and reject unknown group names."""
    out: List[str] = []
    for name in names:
        for expanded in ALIASES.get(name, (name,)):
            if expanded not in GROUPS:
                raise KeyError(expanded)
            if expanded not in out:
                out.append(expanded)
    return out


def print_inventory() -> None:
    print("%-22s %10s  %s" % ("group", "size", "archives"))
    total = 0.0
    for name, (paths, size) in GROUPS.items():
        total += size
        print("%-22s %8.2f GB  %d" % (name, size, len(paths)))
    print("%-22s %8.2f GB" % ("TOTAL", total))
    print("\naliases:")
    for name, members in ALIASES.items():
        size = sum(GROUPS[m][1] for m in members)
        print("  %-20s %8.2f GB  %s" % (name, size, " ".join(members)))
    print("\ndefault: %s" % " ".join(DEFAULT_GROUPS))


def download(repo_path: str, target: str, attempts: int = 6) -> str:
    """Fetch one repo file to ``target``, skipping a complete existing copy.

    The archives run to several GB over a link that does drop, so a partial
    download is resumed with a ``Range`` request rather than restarted, and a
    network error is retried with a widening backoff.  Bytes land in a
    ``.part`` file and are renamed into place only once the transfer is
    complete, so an interrupted run never leaves a truncated archive looking
    like a finished one.
    """
    import time
    import urllib.error
    import urllib.request

    if os.path.exists(target) and os.path.getsize(target) > 0:
        print("    have %s" % os.path.basename(target), flush=True)
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"

    for attempt in range(1, attempts + 1):
        have = os.path.getsize(partial) if os.path.exists(partial) else 0
        request = urllib.request.Request(_API % repo_path)
        if have:
            request.add_header("Range", "bytes=%d-" % have)
        try:
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as error:
                # 416 means the range starts at or past EOF: the .part already
                # holds the whole file, so there is nothing left to fetch.
                if error.code == 416 and have:
                    os.replace(partial, target)
                    return target
                raise
            with response:
                # A server that ignores Range answers 200 with the whole file;
                # appending then would corrupt it, so start over instead.
                mode = "ab"
                if have and response.status != 206:
                    have, mode = 0, "wb"
                expected = response.headers.get("Content-Length")
                expected = have + int(expected) if expected else None
                with open(partial, mode) as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            written = os.path.getsize(partial)
            if expected is not None and written != expected:
                raise OSError(
                    "%s: got %d bytes, expected %d" % (repo_path, written, expected)
                )
            os.replace(partial, target)
            return target
        except (urllib.error.URLError, OSError) as error:
            if attempt == attempts:
                raise
            delay = min(60, 2 ** attempt)
            print(
                "    retry %d/%d in %ds (%s)" % (attempt, attempts - 1, delay, error),
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def download_labels(raw_dir: str) -> str:
    """Fetch every label CSV; returns the directory holding them."""
    directory = os.path.join(raw_dir, "labels")
    for name in LABEL_CSVS:
        download("label_csv_files/%s.csv" % name, os.path.join(directory, "%s.csv" % name))
    return directory


def _normalise(path: str) -> str:
    """A zip member or CSV ``Image_path`` reduced to one comparable form."""
    cleaned = path.replace("\\", "/").lstrip("./")
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    return cleaned


def _tail(path: str, count: int = 2) -> str:
    """The last ``count`` components of ``path``, as a cheap match key."""
    return "/".join(path.split("/")[-count:])


def load_index(labels_dir: str) -> Tuple[Dict[str, Tuple[int, str]], Dict[str, List[str]]]:
    """``normalised image path -> (label, architecture)`` from the label CSVs.

    ``IsFake`` is the label the binary task needs (real ``0`` / fake ``1``) and
    ``Architecture`` becomes the generator directory the adapter reads.  The
    second return value indexes the same keys by their trailing two components,
    which is what makes the suffix match in :func:`lookup` affordable.
    """
    index: Dict[str, Tuple[int, str]] = {}
    by_tail: Dict[str, List[str]] = {}
    csv.field_size_limit(1 << 24)
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(labels_dir, name), encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_path = row.get("Image_path")
                if not image_path:
                    continue
                try:
                    label = int(row.get("IsFake", ""))
                except (TypeError, ValueError):
                    continue
                architecture = (row.get("Architecture") or "unknown").strip()
                key = _normalise(image_path)
                index[key] = (label, architecture)
                by_tail.setdefault(_tail(key), []).append(key)
    return index, by_tail


def lookup(
    member: str,
    index: Dict[str, Tuple[int, str]],
    by_tail: Dict[str, List[str]],
) -> Optional[Tuple[int, str]]:
    """Find ``member``'s label and architecture, matching paths either way round.

    An archive member can sit *deeper* than the path the CSV records or, as
    with ``celebahq/...`` against ``Real/celebahq/...``, *shallower*.  Try the
    exact path, then CSV keys ending with the member, then progressively
    shorter member suffixes.  A tail that resolves to conflicting rows is
    reported as a miss rather than guessed at.
    """
    cleaned = _normalise(member)
    hit = index.get(cleaned)
    if hit is not None:
        return hit

    candidates = [k for k in by_tail.get(_tail(cleaned), ()) if k.endswith("/" + cleaned)]
    if candidates:
        values = {index[k] for k in candidates}
        return values.pop() if len(values) == 1 else None

    parts = cleaned.split("/")
    for start in range(1, len(parts)):
        hit = index.get("/".join(parts[start:]))
        if hit is not None:
            return hit
    return None


def export_archive(
    archive: str,
    index: Dict[str, Tuple[int, str]],
    by_tail: Dict[str, List[str]],
    out_root: str,
    counts: Dict[str, int],
    generators: Dict[str, int],
) -> None:
    """Extract one archive's images into ``{real,fake}/<architecture>/...``."""
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(IMAGE_EXTENSIONS):
                counts["non_image"] += 1
                continue
            hit = lookup(info.filename, index, by_tail)
            if hit is None:
                counts["unlabelled"] += 1
                continue
            label, architecture = hit
            class_dir = "fake" if label else "real"

            # Keep everything below the architecture so the adapter's
            # generator_depth=0 lands on <architecture> and deeper directories
            # stay available for a finer generator_depth later.
            tail = _normalise(info.filename).split("/")
            tail = tail[tail.index(architecture) + 1:] if architecture in tail else tail[-1:]
            relative = "/".join(tail) or os.path.basename(info.filename)

            target = os.path.join(out_root, class_dir, architecture, *relative.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                counts["already_present"] += 1
                continue
            with bundle.open(info) as source, open(target, "wb") as handle:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            counts[class_dir] += 1
            counts["images"] += 1
            generators[architecture] = generators.get(architecture, 0) + 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_inventory()
        return _OK
    if args.download_only and args.export_only:
        parser.error("--download-only and --export-only are mutually exclusive")

    try:
        groups = list(GROUPS) if args.all else resolve_groups(args.groups)
    except KeyError as error:
        parser.error("unknown group %s; run --list for the inventory" % error)

    planned = sum(GROUPS[g][1] for g in groups)
    print("groups: %s" % " ".join(groups))
    print("approximate download: %.2f GB" % planned, flush=True)

    raw_dir = os.path.abspath(args.raw_dir)
    out_root = os.path.abspath(args.out)

    print("fetching label CSVs", flush=True)
    labels_dir = download_labels(raw_dir)
    index, by_tail = load_index(labels_dir)
    print("labelled images known: %d" % len(index), flush=True)
    if not index:
        print("no label rows loaded", file=sys.stderr)
        return _FAIL

    counts: Dict[str, int] = {
        "images": 0,
        "real": 0,
        "fake": 0,
        "unlabelled": 0,
        "non_image": 0,
        "already_present": 0,
    }
    generators: Dict[str, int] = {}
    archives: List[str] = []

    for group in groups:
        for repo_path in GROUPS[group][0]:
            target = os.path.join(raw_dir, "zips", os.path.basename(repo_path))
            if args.export_only:
                if os.path.exists(target):
                    archives.append(target)
                continue
            print("  %s" % repo_path, flush=True)
            archives.append(download(repo_path, target))

    if args.download_only:
        print(json.dumps({"archives": len(archives), "gb": planned}, indent=2))
        return _OK

    for index_of, archive in enumerate(archives, 1):
        print(
            "exporting [%d/%d] %s" % (index_of, len(archives), os.path.basename(archive)),
            flush=True,
        )
        export_archive(archive, index, by_tail, out_root, counts, generators)
        if args.drop_zip:
            os.remove(archive)

    summary = {
        "source": "modelscope:%s/%s" % (NAMESPACE, DATASET),
        "groups": groups,
        "archives": len(archives),
        "approx_download_gb": planned,
        "out": out_root,
        "counts": counts,
        "generators": dict(sorted(generators.items())),
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if counts["images"] == 0:
        print("no images exported", file=sys.stderr)
        return _FAIL
    return _OK


if __name__ == "__main__":
    sys.exit(main())
