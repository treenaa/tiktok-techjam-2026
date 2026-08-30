from __future__ import annotations

import json

import pytest
from PIL import Image

import predict as cli
from src.data import build_preprocess
from src.inference import LoadedArtifact

torch = pytest.importorskip("torch")


class MeanModel(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(1, 2, 3))


def artifact():
    return LoadedArtifact(
        model=MeanModel().eval(),
        preprocess=build_preprocess("none", image_size=12),
        device="cpu",
        threshold=0.5,
        threshold_source="validation",
        checkpoint_path="fake.pt",
        metadata={},
    )


def test_cli_writes_sorted_exact_json_errors_and_separate_diagnostics(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    nested = inputs / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (16, 16), (255, 255, 255)).save(inputs / "z.png")
    Image.new("RGB", (16, 16), (0, 0, 0)).save(nested / "a.jpg")
    (inputs / "bad.webp").write_text("corrupt", encoding="utf-8")
    monkeypatch.setattr(cli, "load_artifact", lambda *args, **kwargs: artifact())

    output = tmp_path / "predictions.json"
    diagnostics = tmp_path / "diagnostics.json"
    assert cli.main(
        [
            "--input", str(inputs),
            "--output", str(output),
            "--checkpoint", "fake.pt",
            "--device", "cpu",
            "--on-error", "skip",
            "--path-format", "input-relative",
            "--diagnostics-output", str(diagnostics),
        ]
    ) == 0
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows == [
        {"image_path": "nested/a.jpg", "pred": pytest.approx(0.5)},
        {"image_path": "z.png", "pred": pytest.approx(float(torch.sigmoid(torch.tensor(1.0))))},
    ]
    assert all(set(row) == {"image_path", "pred"} for row in rows)
    errors = json.loads((tmp_path / "predictions.errors.json").read_text(encoding="utf-8"))
    assert errors["n_unreadable"] == 1
    assert errors["unreadable"][0]["image_path"] == "bad.webp"
    diagnostic_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert len(diagnostic_payload["diagnostics"]) == 2
    assert "scores" in diagnostic_payload["diagnostics"][0]


def test_path_format_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = str(tmp_path / "images" / "a.png")
    assert cli.format_output_path(path, str(tmp_path / "images"), "relative") == "images/a.png"
    assert cli.format_output_path(path, str(tmp_path / "images"), "input-relative") == "a.png"
    assert cli.format_output_path(path, str(tmp_path / "images"), "absolute").endswith("/images/a.png")


def test_cli_supports_required_formats_and_nested_directories(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    nested = inputs / "nested"
    nested.mkdir(parents=True)
    for path in (
        inputs / "a.jpg",
        inputs / "b.jpeg",
        inputs / "c.png",
        nested / "d.webp",
    ):
        Image.new("RGB", (8, 8), (64, 64, 64)).save(path)
    monkeypatch.setattr(cli, "load_artifact", lambda *args, **kwargs: artifact())

    output = tmp_path / "predictions.json"
    assert cli.main(
        [
            "--input", str(inputs),
            "--output", str(output),
            "--checkpoint", "fake.pt",
            "--path-format", "input-relative",
        ]
    ) == 0

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["image_path"] for row in rows] == [
        "a.jpg",
        "b.jpeg",
        "c.png",
        "nested/d.webp",
    ]


def test_cli_empty_directory_fails_before_checkpoint_loading(tmp_path, monkeypatch):
    inputs = tmp_path / "empty"
    inputs.mkdir()
    output = tmp_path / "predictions.json"

    def unexpected_load(*args, **kwargs):
        raise AssertionError("checkpoint loading should not run for an empty directory")

    monkeypatch.setattr(cli, "load_artifact", unexpected_load)
    with pytest.raises(cli.InferenceError, match="no supported images"):
        cli.main(
            [
                "--input", str(inputs),
                "--output", str(output),
                "--checkpoint", "fake.pt",
            ]
        )
    assert not output.exists()
