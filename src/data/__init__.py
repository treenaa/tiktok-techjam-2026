"""Data pipeline for binary AIGC detection (real=0 / AIGC=1).

Typical flow::

    from src.data import build_manifest, split_records, write_manifest
    from src.data import ManifestDataset, PairedViewDataset, build_preprocess

    records = build_manifest("cifake", "/data/cifake")
    splits  = split_records(records, ratios=(0.7, 0.15, 0.15), seed=42)
    write_manifest(splits["train"], "manifests/cifake_train.csv")

    pre   = build_preprocess("ijepa", image_size=224)   # model-owned
    train = PairedViewDataset(splits["train"], preprocess=pre)
    val   = ManifestDataset(splits["val"], preprocess=pre)

See ``src/data/README.md`` for the full API notes.
"""

from __future__ import annotations

from .adapters import (
    ADAPTERS,
    DEFAULT_CLASS_MAP,
    IMAGE_EXTENSIONS,
    build_manifest,
    cifake_adapter,
    from_class_folders,
    from_folder,
    iter_image_paths,
    list_adapters,
    sid_set_adapter,
    wildfake_adapter,
)
from .datasets import (
    BaseImageDataset,
    ManifestDataset,
    PairedViewDataset,
    TransformedEvalDataset,
    build_eval_datasets,
    default_loader,
)
from .manifest import (
    describe_records,
    filter_records,
    merge_manifests,
    read_manifest,
    records_from_dataframe,
    records_to_dataframe,
    write_manifest,
)
from .preprocessing import (
    NORMALIZATION_PRESETS,
    ImagePreprocessing,
    build_preprocess,
    normalize,
    to_tensor,
)
from .schema import (
    LABEL_AIGC,
    LABEL_NAMES,
    LABEL_REAL,
    MANIFEST_COLUMNS,
    REQUIRED_COLUMNS,
    DataError,
    ManifestRecord,
    label_counts,
    source_ids,
    validate_records,
)
from .source_id import (
    canonical_source_id,
    make_source_id_fn,
    find_source_id_collisions,
    strip_transform_suffixes,
)
from .splitting import (
    DEFAULT_RATIOS,
    LeakageError,
    assert_no_path_overlap,
    assert_no_source_id_leakage,
    assign_splits,
    check_split_integrity,
    format_split_report,
    split_by_source_id,
    split_records,
    split_report,
)
from .transforms import (
    EVAL_TRANSFORM_NAMES,
    TRANSFORM_FAMILIES,
    TRANSFORM_REGISTRY,
    CenterCropResize,
    ColorJitter,
    Compose,
    GaussianBlur,
    GaussianNoise,
    Identity,
    JPEGCompression,
    RandomCompetitionTransform,
    ResizeRoundTrip,
    Transform,
    build_eval_suite,
    get_transform,
    list_transforms,
)

__version__ = "0.1.0"

__all__ = [
    # schema
    "ManifestRecord", "DataError", "LABEL_REAL", "LABEL_AIGC", "LABEL_NAMES",
    "MANIFEST_COLUMNS", "REQUIRED_COLUMNS", "validate_records", "label_counts",
    "source_ids",
    # source ids
    "make_source_id_fn", "canonical_source_id", "strip_transform_suffixes",
    "find_source_id_collisions",
    # adapters
    "build_manifest", "from_folder", "from_class_folders", "cifake_adapter",
    "sid_set_adapter", "wildfake_adapter", "ADAPTERS", "list_adapters",
    "IMAGE_EXTENSIONS", "DEFAULT_CLASS_MAP", "iter_image_paths",
    # manifest io
    "read_manifest", "write_manifest", "records_to_dataframe",
    "records_from_dataframe", "filter_records", "describe_records",
    "merge_manifests",
    # splitting
    "split_records", "split_by_source_id", "assign_splits", "DEFAULT_RATIOS",
    "LeakageError", "assert_no_source_id_leakage", "assert_no_path_overlap",
    "check_split_integrity", "split_report", "format_split_report",
    # transforms
    "Transform", "Identity", "Compose", "JPEGCompression", "GaussianBlur",
    "ResizeRoundTrip", "GaussianNoise", "ColorJitter", "CenterCropResize",
    "TRANSFORM_REGISTRY", "TRANSFORM_FAMILIES", "EVAL_TRANSFORM_NAMES",
    "get_transform", "list_transforms", "build_eval_suite",
    "RandomCompetitionTransform",
    # preprocessing
    "ImagePreprocessing", "build_preprocess", "NORMALIZATION_PRESETS",
    "to_tensor", "normalize",
    # datasets
    "BaseImageDataset", "ManifestDataset", "PairedViewDataset",
    "TransformedEvalDataset", "build_eval_datasets", "default_loader",
]
