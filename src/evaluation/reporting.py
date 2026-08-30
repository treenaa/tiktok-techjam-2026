"""Build and persist machine-readable robustness reports."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from src.data import describe_eval_transforms

from .analysis import representative_errors, subgroup_metrics
from .metrics import (
    aggregate_stability,
    binary_classification_metrics,
    robustness_summary,
    score_stability,
)
from .types import EvaluationError, EvaluationRun


REPORT_SCHEMA_VERSION = "1.0"


def build_report(
    run: EvaluationRun,
    *,
    threshold: float = 0.5,
    max_errors_per_type: int = 20,
    model_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the canonical JSON-serialisable evaluation report."""
    clean = run.predictions["clean"]
    transform_metrics: Dict[str, Dict[str, Any]] = {}
    stability: Dict[str, Dict[str, Any]] = {}
    for name, table in run.predictions.items():
        transform_metrics[name] = binary_classification_metrics(
            table.labels, table.probabilities, threshold
        )
        if name != "clean":
            stability[name] = score_stability(clean, table, threshold)
    clean_auc = transform_metrics["clean"]["auroc"]
    for name, values in transform_metrics.items():
        auc = values["auroc"]
        values["auroc_drop_from_clean"] = (
            float(clean_auc) - float(auc)
            if clean_auc is not None and auc is not None
            else None
        )

    known_specs = {item["name"]: item for item in describe_eval_transforms()}
    specs = []
    for name in run.predictions:
        spec = known_specs.get(name, {"name": name, "family": "custom", "params": {}, "severity": None})
        specs.append(spec)

    subgroups_by_transform = {
        name: {
            "dataset": subgroup_metrics(table, "dataset", threshold),
            "generator": subgroup_metrics(table, "generator", threshold),
        }
        for name, table in run.predictions.items()
    }
    errors_by_transform = {
        name: representative_errors(table, threshold, max_per_type=max_errors_per_type)
        for name, table in run.predictions.items()
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": {"label_0": "real", "label_1": "aigc", "score": "P(AIGC)"},
        "threshold": {
            "value": float(threshold),
            "selection": "provided_to_evaluator; must be selected on validation, never test",
        },
        "model": dict(model_info or {}),
        "n_samples": len(clean),
        "transforms": specs,
        "metrics_by_transform": transform_metrics,
        "stability_by_transform": stability,
        "stability_summary": aggregate_stability(stability),
        "robustness_summary": robustness_summary(transform_metrics),
        "subgroups_by_transform": subgroups_by_transform,
        "representative_errors_by_transform": errors_by_transform,
        "runtime_by_transform": {
            name: runtime.to_dict() for name, runtime in run.runtimes.items()
        },
        "runtime_semantics": {
            "total_seconds": "end-to-end DataLoader iteration, preprocessing, transfer, and inference",
            "model_batch_ms_fields": "model forward and sigmoid only; excludes DataLoader/preprocessing and transfer",
        },
        "metadata_coverage": {
            "dataset_nonempty": int(sum(bool(value) for value in clean.datasets)),
            "generator_nonempty_aigc": int(
                sum(bool(value) and label == 1 for value, label in zip(clean.generators, clean.labels))
            ),
            "n_aigc": int(sum(clean.labels)),
        },
    }


def write_report(report: Mapping[str, Any], path: str) -> str:
    """Write the complete report as key-sorted, readable JSON."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def write_metrics_csv(report: Mapping[str, Any], path: str) -> str:
    """Write the compact clean-vs-transform table used in the submission."""
    metrics = report.get("metrics_by_transform")
    stability = report.get("stability_by_transform", {})
    if not isinstance(metrics, Mapping) or "clean" not in metrics:
        raise EvaluationError("report has no valid metrics_by_transform")
    clean_auc = metrics["clean"].get("auroc")
    fields = [
        "transform",
        "n_samples",
        "auroc",
        "auroc_drop",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "mean_absolute_drift",
        "p95_absolute_drift",
        "class_flip_rate",
        "samples_per_second",
    ]
    runtime = report.get("runtime_by_transform", {})
    rows = []
    for name, values in metrics.items():
        auc = values.get("auroc")
        drift = stability.get(name, {})
        rows.append(
            {
                "transform": name,
                "n_samples": values.get("n_samples"),
                "auroc": auc,
                "auroc_drop": (
                    float(clean_auc) - float(auc)
                    if clean_auc is not None and auc is not None
                    else None
                ),
                "accuracy": values.get("accuracy"),
                "f1": values.get("f1"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "specificity": values.get("specificity"),
                "false_positive_rate": values.get("false_positive_rate"),
                "mean_absolute_drift": drift.get("mean_absolute_drift"),
                "p95_absolute_drift": drift.get("p95_absolute_drift"),
                "class_flip_rate": drift.get("class_flip_rate"),
                "samples_per_second": runtime.get(name, {}).get("samples_per_second"),
            }
        )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_predictions(run: EvaluationRun, output_dir: str) -> Dict[str, str]:
    """Persist per-image probabilities for auditability and later plots."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, table in run.predictions.items():
        path = os.path.join(output_dir, "%s.jsonl" % name)
        with open(path, "w", encoding="utf-8") as handle:
            for row in table.rows():
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        paths[name] = path
    return paths
