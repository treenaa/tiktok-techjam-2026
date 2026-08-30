#!/usr/bin/env python3
"""Train one leakage-checked AIGC detector experiment."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from src.data import load_experiment, save_experiment, seed_everything
from src.models import canonical_backbone_name, get_backbone_spec, parameter_report
from src.training import TrainingConfig, TrainingError, Trainer, build_datasets, build_loaders


LEGACY_BACKBONES = {
    "dinov2_vitb14": "dinov2",
    "dino_vitb14": "dinov2",
    "clip_vit_b16": "clip",
    "clip_vit_b32": "clip",
    "ijepa_vith14": "ijepa",
}


def _import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise TrainingError("factory must be written as 'module.path:function_name'")
    module_name, attribute = spec.rsplit(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise TrainingError("could not import %r: %s" % (spec, exc)) from exc
    if not callable(value):
        raise TrainingError("imported object %r is not callable" % spec)
    return value


def translate_model_config(
    model_section: Mapping[str, Any], training_section: Mapping[str, Any]
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
    """Translate existing phase-1 YAML keys into Mateo's factory contract."""
    raw = dict(model_section)
    factory = raw.pop("factory", "src.models:create_model")
    preprocess_factory = raw.pop("preprocess_factory", "src.models:create_preprocess")
    backbone_raw = raw.get("backbone", "dinov2")
    backbone = LEGACY_BACKBONES.get(str(backbone_raw).lower(), backbone_raw)
    backbone = canonical_backbone_name(backbone)
    raw["backbone"] = backbone

    if "pretrained" in raw:
        pretrained = raw.pop("pretrained")
        if not isinstance(pretrained, str) or not pretrained:
            raise TrainingError("model.pretrained must be a non-empty checkpoint identifier")
        raw["model_id"] = pretrained
    forensic = raw.pop("forensic_branch", None)
    if forensic is not None and "architecture" not in raw:
        if not isinstance(forensic, bool):
            raise TrainingError("model.forensic_branch must be true or false")
        raw["architecture"] = "fusion" if forensic else "visual"
    head = raw.pop("head", None)
    if head == "linear" and "head_hidden_dim" not in raw:
        raw["head_hidden_dim"] = None
    elif head not in (None, "linear", "mlp"):
        raise TrainingError("model.head must be linear or mlp")
    image_size = int(raw.pop("image_size", get_backbone_spec(backbone).image_size))
    normalization = raw.pop("normalization", None)
    expected_name = "clip" if backbone == "clip" else "imagenet"
    if normalization is not None and str(normalization).lower() != expected_name:
        raise TrainingError(
            "model normalization %r conflicts with %s's required %s normalization"
            % (normalization, backbone, expected_name)
        )
    raw.setdefault("freeze_backbone", bool(training_section.get("freeze_backbone", True)))
    preprocess_kwargs = {"backbone": backbone, "image_size": image_size}
    return str(factory), raw, str(preprocess_factory), preprocess_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML or JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", help="last.pt or another training checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    experiment = load_experiment(args.config)
    training_config = TrainingConfig.from_mapping(experiment.training)
    seed_everything(
        experiment.seed,
        deterministic_torch=training_config.deterministic_torch,
    )
    splits = experiment.build_splits(validate=True)
    # Deliberately never construct a test dataset/loader here.
    train_records, validation_records = splits["train"], splits["val"]

    model_factory, model_kwargs, preprocess_factory, preprocess_kwargs = translate_model_config(
        experiment.model, experiment.training
    )
    model = _import_callable(model_factory)(**model_kwargs)
    preprocess = _import_callable(preprocess_factory)(**preprocess_kwargs)
    train_dataset, validation_dataset = build_datasets(
        train_records,
        validation_records,
        preprocess=preprocess,
        config=training_config,
        seed=experiment.seed,
    )
    train_loader, validation_loader = build_loaders(
        train_dataset,
        validation_dataset,
        training_config,
        seed=experiment.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    resolved_path = os.path.join(args.output_dir, "experiment.resolved.yaml")
    save_experiment(experiment, resolved_path)
    model_parameters = parameter_report(model)
    if not model_parameters["within_limit"]:
        raise TrainingError("model exceeds the <2B parameter competition limit")
    trainer = Trainer(
        model,
        training_config,
        output_dir=args.output_dir,
        device=args.device,
        run_metadata={
            "experiment_name": experiment.name,
            "experiment_fingerprint": experiment.fingerprint(),
            "experiment_config": os.path.abspath(args.config),
            "model_factory": model_factory,
            "model_kwargs": model_kwargs,
            "preprocess_factory": preprocess_factory,
            "preprocess_kwargs": preprocess_kwargs,
            "parameter_report": model_parameters,
        },
    )
    result = trainer.fit(
        train_loader,
        validation_loader,
        resume_from=args.resume,
    )
    print("Training complete: best epoch %d" % result.best_epoch)
    print("Best %s: %.6f" % (training_config.early_stopping_monitor, result.best_metric))
    print("Validation-selected threshold: %.6f" % result.best_threshold)
    print("Best checkpoint: %s" % result.best_checkpoint)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TrainingError, ValueError, KeyError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
