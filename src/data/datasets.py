"""Dataset objects.

``ManifestDataset``
    One record -> one sample dict with ``image``, ``label``, ``source_id``,
    ``image_path`` plus metadata.  Yields PIL images unless ``preprocess`` is
    given (see :mod:`src.data.preprocessing`).

``PairedViewDataset``
    One record -> ``{"clean", "augmented", "label", "source_id"}`` for
    consistency-style training.  The loss itself lives in the training module,
    not here.

``TransformedEvalDataset``
    A whole dataset viewed through one named competition transform -- the unit
    of the robustness evaluation grid.

``torch`` is imported lazily so the module can be used (and tested) without it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from PIL import Image

from .contract import MODE_PAIRED, MODE_STANDARD, validate_sample
from .schema import ManifestRecord, label_counts, validate_records
from .transforms import Identity, Transform, get_eval_transform

try:  # pragma: no cover - trivial import guard
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _TorchDataset = object
    _HAS_TORCH = False

__all__ = [
    "default_loader",
    "BaseImageDataset",
    "ManifestDataset",
    "PairedViewDataset",
    "TransformedEvalDataset",
    "build_eval_datasets",
]

TransformLike = Union[Callable[["Image.Image"], Any], str, None]


def default_loader(path: str) -> "Image.Image":
    """Load an image as RGB, fully decoded (safe to use with DataLoader workers)."""
    with open(path, "rb") as fh:
        img = Image.open(fh)
        img.load()
    return img if img.mode == "RGB" else img.convert("RGB")


def _resolve_transform(transform: TransformLike) -> Optional[Callable]:
    """Accept a callable, a registry name, or ``None``."""
    if transform is None:
        return None
    if isinstance(transform, str):
        return get_eval_transform(transform)
    if not callable(transform):
        raise TypeError("transform must be callable, a registry name, or None")
    return transform


def _transform_name(transform: Any) -> str:
    if transform is None:
        return "clean"
    return getattr(transform, "name", type(transform).__name__)


class BaseImageDataset(_TorchDataset):
    """Shared record handling: resolution, loading, label/metadata plumbing."""

    def __init__(
        self,
        records: Iterable[ManifestRecord],
        root: Optional[str] = None,
        preprocess: Optional[Callable[["Image.Image"], Any]] = None,
        loader: Callable[[str], "Image.Image"] = default_loader,
        validate: bool = True,
        check_paths_exist: bool = False,
    ):
        records = list(records)
        if validate:
            records = validate_records(
                records, check_paths_exist=check_paths_exist, root=root
            )
        elif not records:
            raise ValueError("empty record collection")
        self.records: List[ManifestRecord] = records
        self.root = root
        self.preprocess = preprocess
        self.loader = loader

    #: Contract mode this dataset emits (see :mod:`src.data.contract`).
    mode: str = MODE_STANDARD

    def __len__(self) -> int:
        return len(self.records)

    def validate_schema(self, index: int = 0, **kwargs: Any) -> Dict[str, Any]:
        """Assert one sample satisfies the documented contract; return it.

        Cheap pre-flight check for integration code::

            ManifestDataset(records, preprocess=pre).validate_schema()
        """
        sample = self[index]
        validate_sample(sample, mode=self.mode, **kwargs)
        return sample

    # -- helpers ----------------------------------------------------------
    def path_at(self, index: int) -> str:
        return self.records[index].resolve_path(self.root)

    def load_image(self, index: int) -> "Image.Image":
        """Raw PIL image, before any transform or preprocessing."""
        return self.loader(self.path_at(index))

    def _apply_preprocess(self, img: "Image.Image") -> Any:
        return self.preprocess(img) if self.preprocess is not None else img

    def _meta(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        return {
            "label": rec.label,
            "source_id": rec.source_id,
            "image_path": self.path_at(index),
            "dataset": rec.dataset,
            "generator": rec.generator,
            "index": index,
        }

    # -- introspection ----------------------------------------------------
    @property
    def labels(self) -> List[int]:
        return [rec.label for rec in self.records]

    @property
    def source_ids(self) -> List[str]:
        return [rec.source_id for rec in self.records]

    def label_counts(self) -> Dict[int, int]:
        return label_counts(self.records)

    def subset(self, indices: Sequence[int]) -> "BaseImageDataset":
        """A new dataset of the same type over ``indices`` (keeps settings)."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.records = [self.records[i] for i in indices]
        return clone

    def __repr__(self) -> str:
        counts = self.label_counts()
        return "%s(n=%d, real=%d, aigc=%d, sources=%d, preprocess=%r)" % (
            type(self).__name__,
            len(self.records),
            counts[0],
            counts[1],
            len(set(self.source_ids)),
            self.preprocess,
        )


class ManifestDataset(BaseImageDataset):
    """Single-view dataset -- emits the ``standard`` contract mode.

    Parameters
    ----------
    records:
        :class:`ManifestRecord` list (from an adapter or
        :func:`~src.data.manifest.read_manifest`).
    root:
        Base directory for relative ``image_path`` values.
    transform:
        Optional competition transform (callable or registry name) applied to
        the raw PIL image *before* ``preprocess``.
    preprocess:
        Model-aware preprocessing.  ``None`` yields PIL images.

    Returns per item::

        {"image", "label", "source_id", "image_path", "dataset", "generator",
         "index", "transform_name"}
    """

    def __init__(
        self,
        records: Iterable[ManifestRecord],
        root: Optional[str] = None,
        transform: TransformLike = None,
        preprocess: Optional[Callable[["Image.Image"], Any]] = None,
        loader: Callable[[str], "Image.Image"] = default_loader,
        validate: bool = True,
        check_paths_exist: bool = False,
        return_metadata: bool = True,
    ):
        super().__init__(
            records,
            root=root,
            preprocess=preprocess,
            loader=loader,
            validate=validate,
            check_paths_exist=check_paths_exist,
        )
        self.transform = _resolve_transform(transform)
        self.return_metadata = return_metadata

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        sample: Dict[str, Any] = {"image": self._apply_preprocess(img)}
        meta = self._meta(index)
        if self.return_metadata:
            sample.update(meta)
            sample["transform_name"] = _transform_name(self.transform)
        else:
            sample["label"] = meta["label"]
            sample["source_id"] = meta["source_id"]
            sample["image_path"] = meta["image_path"]
        return sample


class PairedViewDataset(BaseImageDataset):
    """Clean + transformed views of one image -- emits the ``paired`` mode.

    Parameters
    ----------
    augment:
        What produces the augmented view.  Either

        * a :class:`~src.data.transforms.RandomCompetitionTransform` (or any
          object with ``.sample()``) -- a fresh corruption per access, for
          training; or
        * a fixed callable / registry name -- the same corruption every time,
          for debugging and paired evaluation.

        Defaults to a ``RandomCompetitionTransform`` over all six families.
    clean_transform:
        Optional transform for the clean branch (usually ``None``).  Both
        branches always share the same source image, label and ``source_id``.
    same_preprocess:
        Both views go through ``preprocess``.  Set ``augmented_preprocess`` to
        differ.

    Returns per item::

        {"clean", "augmented", "label", "source_id", "image_path", "dataset",
         "generator", "index", "transform_name"}

    ``transform_name`` names the corruption actually applied, so training code can
    log or condition on it.  This class produces views only -- no loss.
    """

    def __init__(
        self,
        records: Iterable[ManifestRecord],
        root: Optional[str] = None,
        augment: Any = None,
        clean_transform: TransformLike = None,
        preprocess: Optional[Callable[["Image.Image"], Any]] = None,
        augmented_preprocess: Optional[Callable[["Image.Image"], Any]] = None,
        loader: Callable[[str], "Image.Image"] = default_loader,
        validate: bool = True,
        check_paths_exist: bool = False,
    ):
        super().__init__(
            records,
            root=root,
            preprocess=preprocess,
            loader=loader,
            validate=validate,
            check_paths_exist=check_paths_exist,
        )
        self.mode = MODE_PAIRED
        if augment is None:
            from .transforms import RandomCompetitionTransform

            augment = RandomCompetitionTransform()
        self.augment = augment if hasattr(augment, "sample") else _resolve_transform(augment)
        if self.augment is None:
            self.augment = Identity()
        self.clean_transform = _resolve_transform(clean_transform)
        self.augmented_preprocess = augmented_preprocess

    def _sample_transform(self) -> Callable:
        """Concrete transform for this access (fresh draw if stochastic)."""
        if hasattr(self.augment, "sample"):
            return self.augment.sample()
        return self.augment

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img = self.load_image(index)
        clean_img = self.clean_transform(img) if self.clean_transform is not None else img
        transform = self._sample_transform()
        aug_img = transform(img)

        aug_preprocess = self.augmented_preprocess or self.preprocess
        sample: Dict[str, Any] = {
            "clean": self._apply_preprocess(clean_img),
            "augmented": aug_preprocess(aug_img) if aug_preprocess is not None else aug_img,
        }
        sample.update(self._meta(index))
        sample["transform_name"] = _transform_name(transform)
        return sample


class TransformedEvalDataset(ManifestDataset):
    """A dataset viewed through exactly one named competition transform.

    Thin wrapper over :class:`ManifestDataset` that keeps ``transform_name``
    accessible for reporting the robustness grid.
    """

    def __init__(self, records: Iterable[ManifestRecord], transform_name: str = "clean", **kwargs: Any):
        kwargs["transform"] = get_eval_transform(transform_name)
        super().__init__(records, **kwargs)
        self.transform_name = transform_name

    def __repr__(self) -> str:
        return "TransformedEvalDataset(n=%d, transform=%r)" % (
            len(self.records),
            self.transform_name,
        )


def build_eval_datasets(
    records: Iterable[ManifestRecord],
    transform_names: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> "Dict[str, TransformedEvalDataset]":
    """``{transform_name: dataset}`` covering the whole evaluation suite.

    Every dataset holds the same records in the same order, so per-transform
    predictions line up row-by-row.
    """
    from .transforms import EVAL_TRANSFORM_NAMES

    records = list(records)
    names = list(EVAL_TRANSFORM_NAMES) if transform_names is None else list(transform_names)
    return {
        name: TransformedEvalDataset(records, transform_name=name, **kwargs) for name in names
    }
