"""Epoch loops, validation-only model selection, and resumable orchestration."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

from src.data import MODE_PAIRED, MODE_STANDARD, validate_batch
from src.evaluation import binary_classification_metrics, resolve_device

from .checkpoints import (
    checkpoint_payload,
    read_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig, TrainingError
from .loaders import capture_loader_state, restore_loader_state
from .losses import RobustBinaryObjective
from .thresholds import select_threshold


@dataclass
class TrainingResult:
    history: List[Dict[str, Any]]
    best_epoch: int
    best_metric: float
    best_threshold: float
    best_checkpoint: str
    last_checkpoint: str
    stopped_early: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_threshold": self.best_threshold,
            "best_checkpoint": self.best_checkpoint,
            "last_checkpoint": self.last_checkpoint,
            "stopped_early": self.stopped_early,
        }


class EarlyStopping:
    def __init__(
        self,
        *,
        mode: str = "max",
        patience: Optional[int] = 3,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"max", "min"}:
            raise TrainingError("early stopping mode must be max or min")
        if patience is not None and patience < 0:
            raise TrainingError("early stopping patience cannot be negative")
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> tuple[bool, bool]:
        value = float(value)
        if not math.isfinite(value):
            raise TrainingError("early stopping metric is NaN or infinity")
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = (
            not improved
            and self.patience is not None
            and self.bad_epochs >= self.patience
        )
        return improved, should_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("mode") != self.mode or state.get("patience") != self.patience:
            raise TrainingError("resume checkpoint early-stopping policy does not match config")
        if float(state.get("min_delta", 0.0)) != self.min_delta:
            raise TrainingError("resume checkpoint early-stopping min_delta does not match config")
        self.best = state.get("best")
        self.bad_epochs = int(state.get("bad_epochs", 0))


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingError("model has no trainable parameters")
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.SGD(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> Optional[Any]:
    if config.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr,
    )


def _create_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda", enabled=True) if enabled else nullcontext()


def _ensure_logits(output: Any, batch_size: int) -> Tensor:
    if not torch.is_tensor(output):
        raise TrainingError("model must return a tensor of raw logits")
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or len(output) != batch_size:
        raise TrainingError(
            "model must return logits shaped (B,) or (B,1), got %s" % (tuple(output.shape),)
        )
    if not bool(torch.isfinite(output).all()):
        raise TrainingError("model returned NaN or infinite logits")
    return output


def _epoch_metrics(labels: List[int], probabilities: List[float], prefix: str) -> Dict[str, Any]:
    metrics = binary_classification_metrics(labels, probabilities, threshold=0.5)
    return {
        "%s_auroc" % prefix: metrics["auroc"],
        "%s_accuracy_at_0.5" % prefix: metrics["accuracy"],
        "%s_f1_at_0.5" % prefix: metrics["f1"],
    }


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    objective: RobustBinaryObjective,
    *,
    device: str,
    scaler: Any,
    amp: bool,
    max_grad_norm: Optional[float],
) -> Dict[str, Any]:
    model.train()
    total_samples = 0
    sums = {
        "loss": 0.0,
        "clean_classification_loss": 0.0,
        "augmented_classification_loss": 0.0,
        "consistency_loss": 0.0,
    }
    labels_all: List[int] = []
    clean_probabilities: List[float] = []
    augmented_probabilities: List[float] = []

    for batch in loader:
        paired = "clean" in batch or "augmented" in batch
        validate_batch(batch, mode=MODE_PAIRED if paired else MODE_STANDARD)
        labels = batch["label"].to(device=device, dtype=torch.float32)
        batch_size = int(labels.shape[0])
        clean_images = batch["clean" if paired else "image"].to(device)
        augmented_images = batch["augmented"].to(device) if paired else None

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            clean_logits = _ensure_logits(model(clean_images), batch_size)
            augmented_logits = (
                _ensure_logits(model(augmented_images), batch_size)
                if augmented_images is not None
                else None
            )
            losses = objective(clean_logits, labels, augmented_logits)
        scaler.scale(losses.total).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        detached = losses.detached()
        total_samples += batch_size
        for name in sums:
            value = detached[name]
            if value is not None:
                sums[name] += value * batch_size
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        clean_probabilities.extend(
            float(value) for value in torch.sigmoid(clean_logits.detach()).cpu().tolist()
        )
        if augmented_logits is not None:
            augmented_probabilities.extend(
                float(value) for value in torch.sigmoid(augmented_logits.detach()).cpu().tolist()
            )

    if total_samples == 0:
        raise TrainingError("training loader produced no samples")
    result: Dict[str, Any] = {
        "train_loss": sums["loss"] / total_samples,
        "train_clean_classification_loss": sums["clean_classification_loss"] / total_samples,
    }
    result.update(_epoch_metrics(labels_all, clean_probabilities, "train_clean"))
    if augmented_probabilities:
        result.update(
            {
                "train_augmented_classification_loss": sums["augmented_classification_loss"] / total_samples,
                "train_consistency_loss": sums["consistency_loss"] / total_samples,
                "train_mean_absolute_drift": float(
                    np.mean(np.abs(np.asarray(clean_probabilities) - np.asarray(augmented_probabilities)))
                ),
            }
        )
        result.update(_epoch_metrics(labels_all, augmented_probabilities, "train_augmented"))
    return result


def validate_one_epoch(
    model: nn.Module,
    loader: Any,
    objective: RobustBinaryObjective,
    *,
    device: str,
    threshold_metric: str,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    labels_all: List[int] = []
    probabilities: List[float] = []
    loss_sum = 0.0
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            validate_batch(batch, mode=MODE_STANDARD)
            labels = batch["label"].to(device=device, dtype=torch.float32)
            images = batch["image"].to(device)
            batch_size = int(labels.shape[0])
            with _autocast(device, amp):
                logits = _ensure_logits(model(images), batch_size)
                loss = objective.classification_loss(logits, labels)
            loss_sum += float(loss.detach()) * batch_size
            total_samples += batch_size
            labels_all.extend(int(value) for value in labels.cpu().tolist())
            probabilities.extend(float(value) for value in torch.sigmoid(logits).cpu().tolist())
    if total_samples == 0:
        raise TrainingError("validation loader produced no samples")
    selected = select_threshold(labels_all, probabilities, metric=threshold_metric)
    metrics = binary_classification_metrics(
        labels_all, probabilities, threshold=selected["threshold"]
    )
    return {
        "val_loss": loss_sum / total_samples,
        "val_auroc": metrics["auroc"],
        "val_accuracy": metrics["accuracy"],
        "val_f1": metrics["f1"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_specificity": metrics["specificity"],
        "val_false_positive_rate": metrics["false_positive_rate"],
        "val_threshold": selected["threshold"],
        "val_threshold_metric": selected["metric"],
        "val_threshold_metric_value": selected["metric_value"],
        "val_threshold_source": "validation",
    }


def _write_json_atomic(payload: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".history-", suffix=".json", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        output_dir: str,
        device: str = "auto",
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.output_dir = os.path.abspath(output_dir)
        self.device = resolve_device(device)
        if config.amp and not self.device.startswith("cuda"):
            raise TrainingError("amp=true currently requires a CUDA device")
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.to(self.device)
        self.objective = RobustBinaryObjective(
            clean_weight=config.clean_loss_weight,
            augmented_weight=config.augmented_loss_weight,
            consistency_weight=config.consistency_weight,
            positive_class_weight=config.positive_class_weight,
        ).to(self.device)
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.scaler = _create_scaler(config.amp)
        self.early_stopping = EarlyStopping(
            mode=config.early_stopping_mode,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        self.run_metadata = dict(run_metadata or {})

    def _check_resume_config(self, saved: Mapping[str, Any]) -> None:
        current = self.config.to_dict()
        saved = dict(saved)
        saved_epochs = saved.pop("epochs", None)
        current_epochs = current.pop("epochs")
        if saved != current:
            differences = sorted(
                key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
            )
            raise TrainingError("resume config differs at %s" % differences)
        if saved_epochs is not None and current_epochs < int(saved_epochs):
            raise TrainingError("resume epochs cannot be lower than the checkpoint config")
        if self.config.scheduler != "none" and current_epochs != saved_epochs:
            raise TrainingError("changing epochs on resume is incompatible with the configured scheduler")

    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        *,
        resume_from: Optional[str] = None,
    ) -> TrainingResult:
        history: List[Dict[str, Any]] = []
        start_epoch = 0
        best_epoch: Optional[int] = None
        best_threshold = 0.5
        if resume_from:
            before = read_checkpoint(resume_from)
            self._check_resume_config(before.get("training_config", {}))
            payload = restore_training_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            history = list(payload.get("history", []))
            start_epoch = int(payload["epoch"]) + 1
            best_epoch = payload.get("best_epoch")
            best_threshold = float(payload.get("best_threshold", payload.get("threshold", 0.5)))
            self.early_stopping.load_state_dict(payload.get("early_stopping_state", {}))
            restore_loader_state(train_loader, payload.get("loader_state", {}))

        best_path = os.path.join(self.output_dir, "best.pt")
        last_path = os.path.join(self.output_dir, "last.pt")
        stopped_early = False
        for epoch in range(start_epoch, self.config.epochs):
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.objective,
                device=self.device,
                scaler=self.scaler,
                amp=self.config.amp,
                max_grad_norm=self.config.max_grad_norm,
            )
            val_metrics = validate_one_epoch(
                self.model,
                validation_loader,
                self.objective,
                device=self.device,
                threshold_metric=self.config.threshold_metric,
                amp=self.config.amp,
            )
            row = {"epoch": epoch, "learning_rate": learning_rate, **train_metrics, **val_metrics}
            history.append(row)
            monitor = self.config.early_stopping_monitor
            if monitor not in row or row[monitor] is None:
                raise TrainingError("early stopping monitor %r is missing or undefined" % monitor)
            improved, should_stop = self.early_stopping.step(float(row[monitor]))
            if improved:
                best_epoch = epoch
                best_threshold = float(row["val_threshold"])
            if self.scheduler is not None:
                self.scheduler.step()

            payload = checkpoint_payload(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                history=history,
                training_config=self.config.to_dict(),
                threshold=float(row["val_threshold"]),
                best_metric=self.early_stopping.best,
                best_epoch=best_epoch,
                early_stopping_state=self.early_stopping.state_dict(),
                loader_state=capture_loader_state(train_loader),
                run_metadata=self.run_metadata,
            )
            payload["best_threshold"] = best_threshold
            if improved:
                save_checkpoint(payload, best_path)
            save_checkpoint(payload, last_path)
            _write_json_atomic({"history": history}, os.path.join(self.output_dir, "history.json"))
            if should_stop:
                stopped_early = True
                break

        if best_epoch is None or self.early_stopping.best is None:
            raise TrainingError("training completed without a valid validation metric")
        result = TrainingResult(
            history=history,
            best_epoch=int(best_epoch),
            best_metric=float(self.early_stopping.best),
            best_threshold=float(best_threshold),
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            stopped_early=stopped_early,
        )
        _write_json_atomic(result.to_dict(), os.path.join(self.output_dir, "training_summary.json"))
        return result
