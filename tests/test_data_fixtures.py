"""Shared helpers for the data-pipeline tests.

Builds small synthetic on-disk datasets so the adapters, splitter and
transforms can be exercised without downloading SID_Set/CIFAKE/WildFake.

Named ``test_data_fixtures`` to sit inside the data owner's test namespace; it
contains helpers plus one self-check test.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

IMAGE_SIZE = (64, 48)  # (width, height) -- deliberately non-square


def make_image(seed: int = 0, size=IMAGE_SIZE, mode: str = "RGB") -> "Image.Image":
    """Deterministic structured image: gradients + noise.

    Structure matters -- a flat image is unchanged by blur/JPEG, which would
    make the transform tests vacuous.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            (xs / max(w - 1, 1)) * 255,
            (ys / max(h - 1, 1)) * 255,
            ((xs + ys) % 32) * 8,
        ],
        axis=-1,
    )
    arr = np.clip(base + rng.normal(0, 24, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img if mode == "RGB" else img.convert(mode)


def write_image(path: str, seed: int = 0, size=IMAGE_SIZE) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_image(seed, size).save(path)
    return path


def build_cifake_tree(root, n_per_class: int = 6, size=(32, 32)) -> str:
    """``<root>/{train,test}/{REAL,FAKE}/<i>.png`` -- note filenames repeat."""
    root = str(root)
    seed = 0
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            for i in range(n_per_class):
                write_image(os.path.join(root, split, cls, "%04d.png" % i), seed, size)
                seed += 1
    return root


def build_wildfake_tree(root, n_per_leaf: int = 3) -> str:
    """``<root>/{real,fake/<arch>/<model>}/img*.png``."""
    root = str(root)
    seed = 100
    for i in range(n_per_leaf * 2):
        write_image(os.path.join(root, "real", "real_%03d.png" % i), seed)
        seed += 1
    for arch, model in (("gan", "stylegan2"), ("gan", "biggan"), ("diffusion", "sdxl")):
        for i in range(n_per_leaf):
            write_image(
                os.path.join(root, "fake", arch, model, "%s_%03d.png" % (model, i)), seed
            )
            seed += 1
    return root


def build_sid_set_tree(root, n_per_class: int = 4) -> str:
    """``<root>/{train,val,test}/{real,synthetic,tampered}/*.png``.

    Tampered files are named ``<stem>_tampered.png`` after the real image they
    were edited from, so ``tampered_share_real_source_id`` can pair them.
    """
    root = str(root)
    seed = 200
    for split in ("train", "val", "test"):
        for i in range(n_per_class):
            stem = "%s_img%03d" % (split, i)
            write_image(os.path.join(root, split, "real", stem + ".png"), seed)
            write_image(
                os.path.join(root, split, "tampered", stem + "_tampered.png"), seed + 1
            )
            write_image(
                os.path.join(root, split, "synthetic", "%s_syn%03d.png" % (split, i)),
                seed + 2,
            )
            seed += 3
    return root


def build_derivative_tree(root, n_originals: int = 8) -> str:
    """A dataset already containing transformed derivatives on disk.

    ``<root>/{real,fake}/img<i>.png`` plus ``img<i>_jpeg70.png``,
    ``img<i>_blur_1.0.png``, ``img<i>_noise_0.05.png``.  All four files of an
    original must share one ``source_id``.
    """
    root = str(root)
    seed = 300
    for i in range(n_originals):
        cls = "real" if i % 2 == 0 else "fake"
        stem = "img%03d" % i
        for suffix in ("", "_jpeg70", "_blur_1.0", "_noise_0.05"):
            write_image(os.path.join(root, cls, stem + suffix + ".png"), seed)
            seed += 1
    return root


def test_fixture_images_are_deterministic_and_structured():
    """Self-check: the synthetic images are reproducible and not flat."""
    a, b = make_image(3), make_image(3)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(make_image(3)), np.asarray(make_image(4)))
    assert np.asarray(a).std() > 20, "fixture image must carry structure"
    assert a.size == IMAGE_SIZE
