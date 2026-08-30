from __future__ import annotations

import json

import pytest

from src.inference import InferenceError, validate_competition_rows, write_competition_json


def test_competition_json_has_exact_schema_and_no_nan(tmp_path):
    rows = [{"image_path": "images/a.jpg", "pred": 0.25}]
    path = write_competition_json(rows, str(tmp_path / "nested" / "predictions.json"))
    assert json.loads(open(path, encoding="utf-8").read()) == rows


@pytest.mark.parametrize(
    "row,match",
    [
        ({"image_path": "a.jpg", "pred": 0.2, "label": 0}, "only"),
        ({"image_path": "", "pred": 0.2}, "image_path"),
        ({"image_path": "a.jpg", "pred": -0.1}, "outside"),
        ({"image_path": "a.jpg", "pred": 1.1}, "outside"),
        ({"image_path": "a.jpg", "pred": "0.2"}, "numeric"),
    ],
)
def test_competition_schema_rejects_debug_fields_and_invalid_scores(row, match):
    with pytest.raises(InferenceError, match=match):
        validate_competition_rows([row])
