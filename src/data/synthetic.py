"""Tiny synthetic dataset for integration tests.

Lets the whole repo exercise ``dataset -> dataloader -> train -> evaluate``
without downloading SID_Set, CIFAKE or WildFake.  Everything is generated from
seeded numpy, so runs are reproducible and the footprint is a few hundred KB.

Usage from any owner's tests::

    from src.data.synthetic import make_synthetic_dataset

    bundle = make_synthetic_dataset(tmp_path)      # writes images + manifests
    train = ManifestDataset(bundle.train, preprocess=my_preprocess)

The two classes are *learnably* different (AIGC images carry a periodic,
over-smooth structure absent from the "real" ones), so a model that trains
correctly reaches well above chance -- an integration test can assert on that
without the signal being trivially perfect.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .manifest import write_manifest
from .schema import LABEL_AIGC, LABEL_REAL, ManifestRecord
from .splitting import split_records

__all__ = ["SyntheticBundle", "make_synthetic_images", "make_synthetic_dataset"]

#: Fake generator names -- deliberately meaningless, to prove no code path
#: depends on real generator names.
DEFAULT_GENERATORS: Tuple[str, ...] = ("gen_alpha", "gen_beta")


class SyntheticBundle(object):
    """Everything an integration test needs from :func:`make_synthetic_dataset`."""

    def __init__(
        self,
        root: str,
        records: List[ManifestRecord],
        splits: "OrderedDict[str, List[ManifestRecord]]",
        manifest_paths: "OrderedDict[str, str]",
    ):
        self.root = root
        self.records = records
        self.splits = splits
        self.manifest_paths = manifest_paths

    @property
    def train(self) -> List[ManifestRecord]:
        return self.splits["train"]

    @property
    def val(self) -> List[ManifestRecord]:
        return self.splits["val"]

    @property
    def test(self) -> List[ManifestRecord]:
        return self.splits["test"]

    @property
    def train_manifest(self) -> str:
        return self.manifest_paths["train"]

    @property
    def val_manifest(self) -> str:
        return self.manifest_paths["val"]

    @property
    def test_manifest(self) -> str:
        return self.manifest_paths["test"]

    def __repr__(self) -> str:
        return "SyntheticBundle(root=%r, n=%d, splits=%s)" % (
            self.root,
            len(self.records),
            {k: len(v) for k, v in self.splits.items()},
        )


def _real_image(rng: "np.random.Generator", size: Tuple[int, int]) -> "np.ndarray":
    """Natural-ish: broadband noise plus a smooth gradient."""
    h, w = size
    ys, xs = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            120 + 80 * np.sin(xs / max(w, 1) * 2.0 + rng.uniform(0, 6.28)),
            120 + 80 * np.cos(ys / max(h, 1) * 2.0 + rng.uniform(0, 6.28)),
            128 + 60 * ((xs + ys) / max(w + h, 1)),
        ],
        axis=-1,
    )
    return np.clip(base + rng.normal(0, 28, base.shape), 0, 255)


def _aigc_image(rng: "np.random.Generator", size: Tuple[int, int]) -> "np.ndarray":
    """Synthetic-ish: strong periodic grid, little high-frequency noise."""
    h, w = size
    ys, xs = np.mgrid[0:h, 0:w]
    period = rng.integers(3, 6)
    grid = 60 * np.sin(2 * np.pi * xs / period) * np.sin(2 * np.pi * ys / period)
    base = np.stack([140 + grid, 130 + grid, 150 + grid], axis=-1)
    return np.clip(base + rng.normal(0, 4, base.shape), 0, 255)


def make_synthetic_images(
    root: str,
    n_per_class: int = 12,
    size: Tuple[int, int] = (32, 32),
    n_views: int = 1,
    generators: Sequence[str] = DEFAULT_GENERATORS,
    seed: int = 0,
) -> List[ManifestRecord]:
    """Write a tiny real/AIGC tree and return its records.

    Layout::

        <root>/real/real_000.png
        <root>/fake/<generator>/fake_000.png

    ``n_views > 1`` additionally writes ``*_jpeg_70.png`` style derivatives that
    share their original's ``source_id`` -- useful for exercising leakage
    checks.
    """
    if n_per_class < 2:
        raise ValueError("n_per_class must be >= 2 so every split can be populated")
    root = os.path.abspath(str(root))
    rng = np.random.default_rng(seed)
    records: List[ManifestRecord] = []
    view_suffixes = ["", "_jpeg_70", "_blur_1.0", "_noise_0.05"][:n_views]

    for index in range(n_per_class):
        for label in (LABEL_REAL, LABEL_AIGC):
            if label == LABEL_REAL:
                array = _real_image(rng, size)
                generator = ""
                directory = os.path.join(root, "real")
                stem = "real_%03d" % index
            else:
                array = _aigc_image(rng, size)
                generator = generators[index % len(generators)]
                directory = os.path.join(root, "fake", generator)
                stem = "fake_%03d" % index
            os.makedirs(directory, exist_ok=True)
            image = Image.fromarray(array.astype(np.uint8), mode="RGB")

            for suffix in view_suffixes:
                path = os.path.join(directory, stem + suffix + ".png")
                # Derivative views are visibly degraded but keep the class signal.
                if suffix == "_jpeg_70":
                    view = image.filter(ImageFilter.GaussianBlur(0.4))
                elif suffix == "_blur_1.0":
                    view = image.filter(ImageFilter.GaussianBlur(1.0))
                elif suffix == "_noise_0.05":
                    noisy = np.clip(
                        np.asarray(image, dtype=np.float32) + rng.normal(0, 12, (size[0], size[1], 3)),
                        0,
                        255,
                    )
                    view = Image.fromarray(noisy.astype(np.uint8), mode="RGB")
                else:
                    view = image
                view.save(path)
                records.append(
                    ManifestRecord(
                        image_path=path,
                        label=label,
                        source_id="synthetic:%s" % stem,
                        dataset="synthetic",
                        generator=generator,
                    )
                )
    return records


def make_synthetic_dataset(
    root: str,
    n_per_class: int = 12,
    size: Tuple[int, int] = (32, 32),
    n_views: int = 1,
    generators: Sequence[str] = DEFAULT_GENERATORS,
    seed: int = 0,
    ratios: Any = (0.5, 0.25, 0.25),
    write_manifests: bool = True,
    manifest_dir: Optional[str] = None,
) -> SyntheticBundle:
    """Create a complete, leakage-safe synthetic dataset with manifests.

    Returns a :class:`SyntheticBundle` exposing ``.train`` / ``.val`` / ``.test``
    record lists and the paths of the written manifest files.

    Defaults produce 24 images across 3 splits -- small enough for a unit test,
    large enough that every split holds both classes.
    """
    root = os.path.abspath(str(root))
    records = make_synthetic_images(
        root, n_per_class=n_per_class, size=size, n_views=n_views,
        generators=generators, seed=seed,
    )
    splits = split_records(records, ratios=ratios, seed=seed, stratify_keys=("label",))

    manifest_paths: "OrderedDict[str, str]" = OrderedDict()
    if write_manifests:
        manifest_dir = manifest_dir or os.path.join(root, "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        for name, members in splits.items():
            path = os.path.join(manifest_dir, "%s.csv" % name)
            write_manifest([r.with_fields(split=name) for r in members], path)
            manifest_paths[name] = path
    return SyntheticBundle(root, records, splits, manifest_paths)
