from __future__ import annotations

import json

import pytest

from src.data import ManifestRecord, build_eval_datasets, build_preprocess, write_manifest
from src.evaluation import (
    EvaluationError,
    build_report,
    evaluate_grid,
    extract_logits,
    predict_dataset,
    write_metrics_csv,
    write_predictions,
    write_report,
)

torch = pytest.importorskip("torch")


class TinyLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.saw_training_mode = None

    def forward(self, images):
        self.saw_training_mode = self.training
        return (images.mean(dim=(1, 2, 3)) - 0.5) * self.scale


class NaNModel(torch.nn.Module):
    def forward(self, images):
        return torch.full((len(images),), float("nan"), device=images.device)


def records(tmp_path):
    from PIL import Image

    values = [20, 60, 200, 240]
    labels = [0, 0, 1, 1]
    output = []
    for i, (value, label) in enumerate(zip(values, labels)):
        path = tmp_path / ("image-%d.png" % i)
        Image.new("RGB", (24, 24), (value, value, value)).save(path)
        output.append(
            ManifestRecord(
                image_path=str(path),
                label=label,
                source_id="source-%d" % i,
                dataset="synthetic",
                generator="toy" if label else "",
                split="test",
            )
        )
    return output


def test_predict_dataset_sets_eval_and_applies_sigmoid_once(tmp_path):
    grid = build_eval_datasets(
        records(tmp_path),
        transform_names=["clean"],
        preprocess=build_preprocess("none", image_size=16),
    )
    model = TinyLogitModel()
    model.train()
    table, runtime = predict_dataset(model, grid["clean"], batch_size=2, device="cpu")
    expected = torch.sigmoid((torch.tensor([20, 60, 200, 240]) / 255.0 - 0.5) * 2.0)
    assert table.probabilities == pytest.approx(expected.tolist())
    assert model.saw_training_mode is False
    assert runtime.n_batches == 2
    assert runtime.to_dict()["samples_per_second"] > 0


def test_grid_preserves_alignment_and_builds_serialisable_report(tmp_path):
    grid = build_eval_datasets(
        records(tmp_path),
        transform_names=["clean", "jpeg_30", "blur_2.0"],
        preprocess=build_preprocess("none", image_size=16),
    )
    run = evaluate_grid(TinyLogitModel(), grid, batch_size=2, device="cpu")
    assert list(run.predictions) == ["clean", "jpeg_30", "blur_2.0"]
    assert run.predictions["clean"].identity == run.predictions["jpeg_30"].identity
    report = build_report(run, threshold=0.5, model_info={"name": "tiny"})
    assert report["task"]["score"] == "P(AIGC)"
    assert report["metrics_by_transform"]["clean"]["auroc"] == pytest.approx(1.0)
    assert report["metrics_by_transform"]["clean"]["auroc_drop_from_clean"] == 0.0
    assert report["stability_summary"]["worst_drift_transform"] in {"jpeg_30", "blur_2.0"}
    assert report["subgroups_by_transform"]["clean"]["generator"]["toy"]["n_aigc"] == 2
    assert "jpeg_30" in report["representative_errors_by_transform"]
    json.dumps(report, allow_nan=False)


def test_report_outputs_are_auditable(tmp_path):
    grid = build_eval_datasets(
        records(tmp_path),
        transform_names=["clean", "jpeg_30"],
        preprocess=build_preprocess("none", image_size=16),
    )
    run = evaluate_grid(TinyLogitModel(), grid, batch_size=4, device="cpu")
    report = build_report(run)
    report_path = write_report(report, str(tmp_path / "out" / "report.json"))
    csv_path = write_metrics_csv(report, str(tmp_path / "out" / "robustness.csv"))
    prediction_paths = write_predictions(run, str(tmp_path / "out" / "predictions"))
    with open(report_path, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["schema_version"] == "1.0"
    assert "auroc_drop" in open(csv_path, encoding="utf-8").readline()
    assert set(prediction_paths) == {"clean", "jpeg_30"}
    with open(prediction_paths["clean"], encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    assert first["transform_name"] == "clean"
    assert set(first) >= {"image_path", "source_id", "label", "prob_aigc"}


def test_runner_rejects_nonfinite_logits(tmp_path):
    grid = build_eval_datasets(
        records(tmp_path),
        transform_names=["clean"],
        preprocess=build_preprocess("none", image_size=8),
    )
    with pytest.raises(EvaluationError, match="NaN"):
        predict_dataset(NaNModel(), grid["clean"], device="cpu")


def test_extract_logits_rejects_ambiguous_two_class_head():
    with pytest.raises(EvaluationError, match="binary detector"):
        extract_logits(torch.zeros(3, 2), expected_batch_size=3)
    assert extract_logits({"logits": torch.zeros(3, 1)}).shape == (3,)


def test_cli_runs_checkpoint_to_reports(tmp_path, monkeypatch):
    import evaluate as cli

    manifest = str(tmp_path / "test.csv")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_manifest(records(image_dir), manifest)
    checkpoint = str(tmp_path / "tiny.pt")
    torch.save({"model_state_dict": TinyLogitModel().state_dict()}, checkpoint)
    monkeypatch.setattr(cli, "_import_callable", lambda spec: TinyLogitModel)

    output_dir = tmp_path / "results"
    exit_code = cli.main(
        [
            "--manifest",
            manifest,
            "--model-factory",
            "fake.module:create_model",
            "--checkpoint",
            checkpoint,
            "--preprocess",
            "none",
            "--image-size",
            "16",
            "--transforms",
            "clean,jpeg_30",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--save-predictions",
        ]
    )
    assert exit_code == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["model"]["parameter_count"] == 1
    assert report["model"]["within_parameter_limit"] is True
    assert list(report["metrics_by_transform"]) == ["clean", "jpeg_30"]
    assert (output_dir / "robustness.csv").exists()
    assert (output_dir / "predictions" / "jpeg_30.jsonl").exists()
