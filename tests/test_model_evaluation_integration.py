from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from src.data import ManifestRecord, build_eval_datasets
from src.evaluation import build_report, evaluate_grid
from src.models import create_model, create_preprocess

torch = pytest.importorskip("torch")


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.conv = torch.nn.Conv2d(3, 6, 1)

    def forward(self, pixel_values):
        tokens = self.conv(pixel_values).flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1, keepdim=True)
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls, tokens), dim=1),
            pooler_output=cls[:, 0],
        )


def test_fusion_model_runs_through_shared_data_and_evaluation_contract(tmp_path):
    records = []
    for index, (value, label) in enumerate(((30, 0), (80, 0), (180, 1), (230, 1))):
        path = tmp_path / ("sample-%d.png" % index)
        Image.new("RGB", (28, 24), (value, value, value)).save(path)
        records.append(
            ManifestRecord(
                image_path=str(path),
                label=label,
                source_id="source-%d" % index,
                dataset="integration",
                generator="toy" if label else "",
                split="test",
            )
        )

    model = create_model(
        backbone="dinov2",
        architecture="fusion",
        backbone_model=TinyBackbone(),
        freeze_backbone=True,
        head_hidden_dim=8,
        head_dropout=0.0,
        forensic_dim=5,
        forensic_width=8,
        forensic_dropout=0.0,
    )
    preprocess = create_preprocess("dinov2", image_size=16)
    grid = build_eval_datasets(
        records,
        transform_names=["clean", "jpeg_30"],
        preprocess=preprocess,
    )
    run = evaluate_grid(model, grid, batch_size=2, device="cpu")
    report = build_report(run, threshold=0.5)

    assert run.predictions["clean"].probabilities.shape == (4,)
    assert run.predictions["clean"].identity == run.predictions["jpeg_30"].identity
    assert set(report["metrics_by_transform"]) == {"clean", "jpeg_30"}
    assert report["task"] == {"label_0": "real", "label_1": "aigc", "score": "P(AIGC)"}
