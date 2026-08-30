#!/usr/bin/env python3
"""Evaluate one checkpoint on clean and official robustness transformations.

Example::

    python evaluate.py \
      --manifest manifests/test.csv \
      --model-factory src.models.factory:create_model \
      --checkpoint checkpoints/best.pt \
      --preprocess ijepa \
      --output-dir results/final

The model factory must return a ``torch.nn.Module`` whose forward pass emits
one raw AIGC logit per image.  Threshold selection is deliberately absent:
provide the threshold previously selected on validation with ``--threshold``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional

from src.data import (
    build_eval_datasets,
    build_preprocess,
    canonical_transform_name,
    list_eval_transforms,
    read_manifest,
    seed_everything,
)
from src.evaluation import (
    EvaluationError,
    build_report,
    evaluate_grid,
    resolve_device,
    write_metrics_csv,
    write_predictions,
    write_report,
)


def _import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise EvaluationError("factory must be written as 'module.path:function_name'")
    module_name, attribute = spec.rsplit(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise EvaluationError("could not import factory %r: %s" % (spec, exc)) from exc
    if not callable(value):
        raise EvaluationError("imported object %r is not callable" % spec)
    return value


def _json_object(value: Optional[str], option: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        if os.path.isfile(value):
            with open(value, encoding="utf-8") as handle:
                parsed = json.load(handle)
        else:
            parsed = json.loads(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("%s must be a JSON object or path to one: %s" % (option, exc)) from exc
    if not isinstance(parsed, dict):
        raise EvaluationError("%s must resolve to a JSON object" % option)
    return parsed


def _state_dict(checkpoint: Any, requested_key: Optional[str]) -> Mapping[str, Any]:
    import torch

    if requested_key:
        if not isinstance(checkpoint, Mapping) or requested_key not in checkpoint:
            raise EvaluationError("checkpoint has no state-dict key %r" % requested_key)
        state = checkpoint[requested_key]
    elif isinstance(checkpoint, Mapping):
        state = None
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        if state is None and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        if state is None:
            raise EvaluationError(
                "checkpoint does not contain 'model_state_dict' or 'state_dict'; "
                "pass --state-dict-key explicitly"
            )
    else:
        raise EvaluationError("checkpoint must contain a state-dict mapping")
    if not isinstance(state, Mapping):
        raise EvaluationError("selected checkpoint state dict is not a mapping")
    return state


def load_model(
    factory_spec: str,
    factory_kwargs: Mapping[str, Any],
    checkpoint_path: str,
    *,
    state_dict_key: Optional[str] = None,
    strict: bool = True,
):
    """Construct the architecture and restore a weights-only checkpoint."""
    import torch

    model = _import_callable(factory_spec)(**dict(factory_kwargs))
    if not isinstance(model, torch.nn.Module):
        raise EvaluationError("model factory must return torch.nn.Module, got %r" % type(model))
    if not os.path.isfile(checkpoint_path):
        raise EvaluationError("checkpoint not found: %s" % checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(
        _state_dict(checkpoint, state_dict_key), strict=bool(strict)
    )
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "WARNING: non-strict checkpoint load: missing=%s unexpected=%s"
            % (incompatible.missing_keys, incompatible.unexpected_keys),
            file=sys.stderr,
        )
    return model


def _transforms(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list_eval_transforms()
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise EvaluationError("--transforms cannot be empty")
    names = [canonical_transform_name(item) for item in requested]
    if "clean" not in names:
        names.insert(0, "clean")
    return list(OrderedDict.fromkeys(names))


def _preprocess(args: argparse.Namespace):
    kwargs = _json_object(args.preprocess_kwargs, "--preprocess-kwargs")
    if args.preprocess_factory:
        return _import_callable(args.preprocess_factory)(**kwargs)
    if "image_size" in kwargs:
        raise EvaluationError("set image size with --image-size, not --preprocess-kwargs")
    return build_preprocess(args.preprocess, image_size=args.image_size, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV/JSON/JSONL evaluation manifest")
    parser.add_argument("--root", help="base directory for relative manifest image paths")
    parser.add_argument("--split", help="optional manifest split to select (normally test)")
    parser.add_argument("--model-factory", required=True, help="module:function returning nn.Module")
    parser.add_argument("--model-kwargs", help="JSON object or path passed to the model factory")
    parser.add_argument("--checkpoint", required=True, help="trained checkpoint")
    parser.add_argument("--state-dict-key", help="non-standard checkpoint key containing weights")
    parser.add_argument(
        "--non-strict-load", action="store_true", help="allow missing/unexpected checkpoint keys"
    )
    parser.add_argument("--preprocess", default="ijepa", help="shared preprocessing preset")
    parser.add_argument("--preprocess-factory", help="custom module:function preprocessing factory")
    parser.add_argument("--preprocess-kwargs", help="JSON object or path for preprocessing factory")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--transforms", default="all", help="all or comma-separated official names")
    parser.add_argument("--threshold", type=float, default=0.5, help="validation-selected threshold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-predictions", action="store_true", help="write one JSONL per transform")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise EvaluationError("--threshold must be in [0, 1]")
    seed_everything(args.seed, deterministic_torch=True)
    records = read_manifest(
        args.manifest,
        root=args.root,
        split=args.split,
        check_paths_exist=True,
    )
    populated_splits = sorted({record.split for record in records if record.split})
    if args.split is None and len(populated_splits) > 1:
        raise EvaluationError(
            "manifest contains multiple splits %s; select one with --split" % populated_splits
        )
    model_kwargs = _json_object(args.model_kwargs, "--model-kwargs")
    model = load_model(
        args.model_factory,
        model_kwargs,
        args.checkpoint,
        state_dict_key=args.state_dict_key,
        strict=not args.non_strict_load,
    )
    transform_names = _transforms(args.transforms)
    grid = build_eval_datasets(
        records,
        transform_names=transform_names,
        preprocess=_preprocess(args),
        check_paths_exist=True,
    )
    device = resolve_device(args.device)
    run = evaluate_grid(
        model,
        grid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    preprocessing_spec = getattr(model, "preprocessing_spec", None)
    if dataclasses.is_dataclass(preprocessing_spec):
        preprocessing_metadata = dataclasses.asdict(preprocessing_spec)
    else:
        preprocessing_metadata = None
    report = build_report(
        run,
        threshold=args.threshold,
        max_errors_per_type=args.max_errors,
        model_info={
            "factory": args.model_factory,
            "backbone": getattr(model, "backbone_name", None),
            "backbone_model_id": getattr(model, "backbone_model_id", None),
            "backbone_revision": getattr(model, "backbone_revision", None),
            "checkpoint": os.path.abspath(args.checkpoint),
            "parameter_count": int(parameter_count),
            "parameter_limit": 2_000_000_000,
            "within_parameter_limit": bool(parameter_count < 2_000_000_000),
            "device": device,
            "preprocess": args.preprocess_factory or args.preprocess,
            "preprocessing_spec": preprocessing_metadata,
            "image_size": args.image_size,
        },
    )
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = write_report(report, os.path.join(args.output_dir, "report.json"))
    csv_path = write_metrics_csv(report, os.path.join(args.output_dir, "robustness.csv"))
    if args.save_predictions:
        write_predictions(run, os.path.join(args.output_dir, "predictions"))

    summary = report["robustness_summary"]
    print("Evaluation complete: %d samples x %d transforms" % (len(records), len(grid)))
    print("Clean AUROC: %.6f" % summary["clean_auroc"])
    if summary["mean_transformed_auroc"] is not None:
        print("Mean transformed AUROC: %.6f" % summary["mean_transformed_auroc"])
        print(
            "Worst transformed AUROC: %.6f (%s)"
            % (summary["worst_case_transformed_auroc"], summary["worst_case_transform"])
        )
    print("Report: %s" % report_path)
    print("Table:  %s" % csv_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, ValueError, KeyError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
