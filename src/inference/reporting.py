"""Competition-safe JSON output and separate diagnostics."""

from __future__ import annotations

import json
import numbers
import os
import tempfile
from typing import Any, Mapping, Sequence

from .artifact import InferenceError


def write_json_atomic(payload: Any, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".prediction-", suffix=".json", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def validate_competition_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if set(row) != {"image_path", "pred"}:
            raise InferenceError(
                "competition row %d must contain image_path and pred only" % index
            )
        if not isinstance(row["image_path"], str) or not row["image_path"]:
            raise InferenceError("competition row %d has invalid image_path" % index)
        if isinstance(row["pred"], bool) or not isinstance(row["pred"], numbers.Real):
            raise InferenceError("competition row %d pred must be a numeric probability" % index)
        probability = float(row["pred"])
        if not 0.0 <= probability <= 1.0:
            raise InferenceError("competition row %d pred is outside [0,1]" % index)


def write_competition_json(rows: Sequence[Mapping[str, Any]], path: str) -> str:
    rows = list(rows)
    validate_competition_rows(rows)
    return write_json_atomic(rows, path)
