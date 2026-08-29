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

from .audit import (
    ShortcutFinding,
    audit_shortcuts,
    encoding_shortcut,
    format_audit_report,
    generator_concentration,
    provenance_shortcut,
    resolution_shortcut,
)
from .build import generate_manifest, generate_split_manifests
from .config import DatasetConfigError, build_from_config, load_config
from .experiment import (
    COMPARABILITY_KEYS,
    ExperimentConfig,
    assert_comparable,
    comparability_report,
    load_experiment,
    save_experiment,
)
from .loading import (
    ON_ERROR_POLICIES,
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    list_images,
    load_image,
    make_loader,
    verify_images,
)
from .protected import (
    DEMO_SPLIT_NAMES,
    PROTECTED_DATASETS,
    ProtectedDataError,
    assert_not_trainable,
    classify_protected,
    find_protected_records,
    partition_protected,
    protected_report,
    register_protected_dataset,
)
from .seeding import dataloader_kwargs, make_generator, seed_everything, seed_worker
from .contract import (
    IMAGE_KEYS,
    MODE_PAIRED,
    MODE_STANDARD,
    MODES,
    PAIRED_OPTIONAL_KEYS,
    PAIRED_REQUIRED_KEYS,
    STANDARD_OPTIONAL_KEYS,
    STANDARD_REQUIRED_KEYS,
    SchemaError,
    all_keys,
    describe_contract,
    optional_keys,
    required_keys,
    validate_batch,
    validate_sample,
)
from .generators import (
    assert_field_disjoint,
    assert_generators_disjoint,
    list_field_values,
    split_by_field_holdout,
    filter_by_generator,
    generator_counts,
    group_by_generator,
    list_generators,
    partition_generators,
    split_by_generator_holdout,
)
from .synthetic import SyntheticBundle, make_synthetic_dataset, make_synthetic_images
from .validation import (
    ValidationReport,
    find_derivative_leakage,
    find_protected_data,
    find_forbidden_combinations,
    normalized_stem,
    validate_splits,
)
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
    OFFICIAL_TRANSFORM_NAMES,
    TRANSFORM_ALIASES,
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
    canonical_transform_name,
    describe_eval_transforms,
    get_eval_transform,
    get_transform,
    list_eval_transforms,
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
    # contract
    "MODE_STANDARD", "MODE_PAIRED", "MODES", "SchemaError", "validate_sample",
    "validate_batch", "required_keys", "optional_keys", "all_keys",
    "describe_contract", "IMAGE_KEYS", "STANDARD_REQUIRED_KEYS",
    "STANDARD_OPTIONAL_KEYS", "PAIRED_REQUIRED_KEYS", "PAIRED_OPTIONAL_KEYS",
    # split validation
    "validate_splits", "ValidationReport", "find_derivative_leakage",
    "find_forbidden_combinations", "normalized_stem", "find_protected_data",
    # protected / demonstration-only data (rule 11.B)
    "PROTECTED_DATASETS", "DEMO_SPLIT_NAMES", "ProtectedDataError",
    "classify_protected", "find_protected_records", "assert_not_trainable",
    "partition_protected", "protected_report", "register_protected_dataset",
    # shortcut auditing (rule 11.C)
    "audit_shortcuts", "format_audit_report", "ShortcutFinding",
    "provenance_shortcut", "resolution_shortcut", "encoding_shortcut",
    "generator_concentration",
    # reproducibility (rule 20.9)
    "seed_everything", "seed_worker", "make_generator", "dataloader_kwargs",
    # config-driven builds (rule 20.8)
    "build_from_config", "load_config", "DatasetConfigError",
    # experiment config / comparability (rule 21)
    "ExperimentConfig", "load_experiment", "save_experiment",
    "comparability_report", "assert_comparable", "COMPARABILITY_KEYS",
    # shared image loading (rules 17, 20.6)
    "load_image", "make_loader", "list_images", "verify_images",
    "ImageLoadError", "SUPPORTED_EXTENSIONS", "ON_ERROR_POLICIES",
    # generator-aware splitting
    "list_generators", "generator_counts", "group_by_generator",
    "filter_by_generator", "partition_generators", "split_by_generator_holdout",
    "assert_generators_disjoint", "list_field_values", "split_by_field_holdout",
    "assert_field_disjoint",
    # manifest generation
    "generate_manifest", "generate_split_manifests",
    # synthetic fixture
    "make_synthetic_dataset", "make_synthetic_images", "SyntheticBundle",
    # transforms
    "Transform", "Identity", "Compose", "JPEGCompression", "GaussianBlur",
    "ResizeRoundTrip", "GaussianNoise", "ColorJitter", "CenterCropResize",
    "TRANSFORM_REGISTRY", "TRANSFORM_FAMILIES", "TRANSFORM_ALIASES",
    "EVAL_TRANSFORM_NAMES", "OFFICIAL_TRANSFORM_NAMES", "get_eval_transform",
    "list_eval_transforms", "describe_eval_transforms",
    "canonical_transform_name", "get_transform", "list_transforms",
    "build_eval_suite",
    "RandomCompetitionTransform",
    # preprocessing
    "ImagePreprocessing", "build_preprocess", "NORMALIZATION_PRESETS",
    "to_tensor", "normalize",
    # datasets
    "BaseImageDataset", "ManifestDataset", "PairedViewDataset",
    "TransformedEvalDataset", "build_eval_datasets", "default_loader",
]
