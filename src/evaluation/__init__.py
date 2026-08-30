"""Reliable evaluation for clean and transformed AIGC detection."""

from .analysis import representative_errors, subgroup_metrics
from .metrics import (
    aggregate_stability,
    auroc,
    binary_classification_metrics,
    robustness_summary,
    score_stability,
)
from .reporting import (
    REPORT_SCHEMA_VERSION,
    build_report,
    write_metrics_csv,
    write_predictions,
    write_report,
)
from .runner import evaluate_grid, extract_logits, predict_dataset, resolve_device
from .types import EvaluationError, EvaluationRun, PredictionTable, RuntimeStats

__all__ = [
    "EvaluationError",
    "PredictionTable",
    "RuntimeStats",
    "EvaluationRun",
    "auroc",
    "binary_classification_metrics",
    "score_stability",
    "aggregate_stability",
    "robustness_summary",
    "subgroup_metrics",
    "representative_errors",
    "resolve_device",
    "extract_logits",
    "predict_dataset",
    "evaluate_grid",
    "REPORT_SCHEMA_VERSION",
    "build_report",
    "write_report",
    "write_metrics_csv",
    "write_predictions",
]
