"""Deterministic batch inference and live robustness diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate

from src.data import canonical_transform_name, get_eval_transform, load_image, verify_images
from src.evaluation import extract_logits

from .artifact import InferenceError, LoadedArtifact


def _to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _forward(model: torch.nn.Module, inputs: Any) -> Any:
    return model(**inputs) if isinstance(inputs, Mapping) else model(inputs)


class _PathDataset(Dataset):
    def __init__(self, paths: Sequence[str], preprocess: Any, allow_truncated: bool) -> None:
        self.paths = list(paths)
        self.preprocess = preprocess
        self.allow_truncated = bool(allow_truncated)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        image = load_image(path, on_error="raise", allow_truncated=self.allow_truncated)
        return {"input": self.preprocess(image), "path": path}


@dataclass(frozen=True)
class Prediction:
    image_path: str
    probability_aigc: float

    def competition_row(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "image_path": output_path if output_path is not None else self.image_path,
            "pred": float(self.probability_aigc),
        }


@dataclass
class PredictionRun:
    predictions: List[Prediction]
    unreadable: List[Tuple[str, str]]
    total_seconds: float

    @property
    def samples_per_second(self) -> Optional[float]:
        if self.total_seconds <= 0:
            return None
        return len(self.predictions) / self.total_seconds


class Predictor:
    def __init__(
        self,
        artifact: LoadedArtifact,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        if batch_size < 1 or num_workers < 0:
            raise InferenceError("batch_size must be positive and num_workers non-negative")
        self.artifact = artifact
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def _probabilities(self, inputs: Any) -> List[float]:
        inputs = _to_device(inputs, self.artifact.device)
        batch_size = None
        if torch.is_tensor(inputs):
            batch_size = len(inputs)
        elif isinstance(inputs, Mapping):
            tensor_values = [value for value in inputs.values() if torch.is_tensor(value)]
            if tensor_values:
                batch_size = len(tensor_values[0])
        if batch_size is None:
            raise InferenceError("preprocessing must produce tensors or a mapping of tensors")
        with torch.inference_mode():
            output = _forward(self.artifact.model, inputs)
            logits = extract_logits(output, expected_batch_size=batch_size)
            probabilities = torch.sigmoid(logits)  # exactly once
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def predict_paths(
        self,
        paths: Sequence[str],
        *,
        on_error: str = "raise",
        allow_truncated: bool = False,
    ) -> PredictionRun:
        if on_error not in {"raise", "skip"}:
            raise InferenceError("on_error must be 'raise' or 'skip'; placeholder scores are forbidden")
        paths = list(paths)
        if not paths:
            raise InferenceError("no supported images were found")
        verification = verify_images(paths, allow_truncated=allow_truncated)
        unreadable = list(verification["unreadable"])
        if unreadable and on_error == "raise":
            examples = "; ".join("%s (%s)" % item for item in unreadable[:3])
            raise InferenceError("%d unreadable image(s): %s" % (len(unreadable), examples))
        readable = list(verification["readable"])
        if not readable:
            raise InferenceError("no readable images remain after verification")

        dataset = _PathDataset(readable, self.artifact.preprocess, allow_truncated)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.artifact.device.startswith("cuda"),
        )
        predictions: List[Prediction] = []
        start = time.perf_counter()
        for batch in loader:
            probabilities = self._probabilities(batch["input"])
            batch_paths = list(batch["path"])
            if len(probabilities) != len(batch_paths):
                raise InferenceError("prediction/path count mismatch")
            predictions.extend(
                Prediction(str(path), probability)
                for path, probability in zip(batch_paths, probabilities)
            )
        elapsed = time.perf_counter() - start
        if [prediction.image_path for prediction in predictions] != readable:
            raise InferenceError("inference changed deterministic input ordering")
        return PredictionRun(predictions, unreadable, elapsed)

    def predict_image(self, image: Image.Image) -> float:
        if not isinstance(image, Image.Image):
            raise InferenceError("predict_image expects PIL.Image")
        batch = default_collate([self.artifact.preprocess(image.convert("RGB"))])
        return self._probabilities(batch)[0]

    def diagnose_image(
        self,
        image: Image.Image,
        transform_names: Sequence[str],
    ) -> Dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise InferenceError("diagnose_image expects PIL.Image")
        names = [canonical_transform_name(name) for name in transform_names]
        if "clean" not in names:
            names.insert(0, "clean")
        names = list(dict.fromkeys(names))
        rgb = image.convert("RGB")
        processed = [self.artifact.preprocess(get_eval_transform(name)(rgb)) for name in names]
        probabilities = self._probabilities(default_collate(processed))
        scores = dict(zip(names, probabilities))
        clean = scores["clean"]
        transformed = [score for name, score in scores.items() if name != "clean"]
        drifts = [abs(clean - score) for score in transformed]
        classes = [score >= self.artifact.threshold for score in scores.values()]
        return {
            "probability_aigc": clean,
            "predicted_label": int(clean >= self.artifact.threshold),
            "prediction": "AI-generated" if clean >= self.artifact.threshold else "real",
            "threshold": self.artifact.threshold,
            "threshold_source": self.artifact.threshold_source,
            "scores": scores,
            "mean_absolute_drift": float(np.mean(drifts)) if drifts else 0.0,
            "max_absolute_drift": float(np.max(drifts)) if drifts else 0.0,
            "class_stable": len(set(classes)) == 1,
            "stability_note": "Prediction stability is not proof that the classification is correct.",
        }
