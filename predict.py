#!/usr/bin/env python3
"""Predict P(AIGC) for every supported image in a directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from src.data import canonical_transform_name, list_images, load_image
from src.inference import (
    InferenceError,
    Predictor,
    load_artifact,
    write_competition_json,
    write_json_atomic,
)


DEFAULT_DIAGNOSTIC_TRANSFORMS = (
    "clean",
    "jpeg_30",
    "blur_2.0",
    "resize_0.25",
    "crop_0.80",
)


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
        raise InferenceError("%s must be a JSON object or path to one: %s" % (option, exc)) from exc
    if not isinstance(parsed, dict):
        raise InferenceError("%s must resolve to a JSON object" % option)
    return parsed


def format_output_path(path: str, input_dir: str, mode: str) -> str:
    if mode == "absolute":
        output = os.path.abspath(path)
    elif mode == "input-relative":
        output = os.path.relpath(os.path.abspath(path), os.path.abspath(input_dir))
    elif mode == "relative":
        output = os.path.relpath(os.path.abspath(path), os.getcwd())
    else:
        raise InferenceError("unknown path format %r" % mode)
    return output.replace(os.sep, "/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="directory containing images")
    parser.add_argument("--output", required=True, help="competition JSON output")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-factory", help="override checkpoint module:function")
    parser.add_argument("--model-kwargs", help="JSON object/path merged over checkpoint metadata")
    parser.add_argument("--preprocess-factory", help="override checkpoint preprocessing factory")
    parser.add_argument("--preprocess-kwargs", help="JSON object/path merged over checkpoint metadata")
    parser.add_argument("--threshold", type=float, help="display/diagnostic override; does not change pred")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--on-error", choices=("raise", "skip"), default="raise")
    parser.add_argument("--errors-output", help="separate unreadable-file report")
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--path-format",
        choices=("relative", "input-relative", "absolute"),
        default="relative",
    )
    parser.add_argument("--diagnostics-output", help="optional robustness JSON; never mixed into output")
    parser.add_argument(
        "--diagnostic-transforms",
        default=",".join(DEFAULT_DIAGNOSTIC_TRANSFORMS),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = list_images(args.input, recursive=not args.no_recursive)
    if not paths:
        raise InferenceError("no supported images were found in %s" % args.input)
    diagnostic_names = None
    if args.diagnostics_output:
        requested = [name.strip() for name in args.diagnostic_transforms.split(",") if name.strip()]
        diagnostic_names = [canonical_transform_name(name) for name in requested]
    artifact = load_artifact(
        args.checkpoint,
        model_factory=args.model_factory,
        model_kwargs=_json_object(args.model_kwargs, "--model-kwargs"),
        preprocess_factory=args.preprocess_factory,
        preprocess_kwargs=_json_object(args.preprocess_kwargs, "--preprocess-kwargs"),
        device=args.device,
        threshold=args.threshold,
    )
    predictor = Predictor(
        artifact,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    run = predictor.predict_paths(
        paths,
        on_error=args.on_error,
        allow_truncated=args.allow_truncated,
    )
    output_paths = {
        prediction.image_path: format_output_path(
            prediction.image_path, args.input, args.path_format
        )
        for prediction in run.predictions
    }
    rows = [
        prediction.competition_row(output_paths[prediction.image_path])
        for prediction in run.predictions
    ]
    diagnostic_payload = None
    if args.diagnostics_output:
        diagnostics = []
        for prediction in run.predictions:
            image = load_image(
                prediction.image_path,
                on_error="raise",
                allow_truncated=args.allow_truncated,
            )
            diagnostics.append(
                {
                    "image_path": output_paths[prediction.image_path],
                    **predictor.diagnose_image(image, diagnostic_names),
                }
            )
        diagnostic_payload = {
            "diagnostics": diagnostics,
            "note": "Stability measures consistency under transformations, not correctness.",
        }

    write_competition_json(rows, args.output)

    if run.unreadable:
        errors_path = args.errors_output or os.path.splitext(args.output)[0] + ".errors.json"
        write_json_atomic(
            {
                "n_unreadable": len(run.unreadable),
                "unreadable": [
                    {"image_path": format_output_path(path, args.input, args.path_format), "reason": reason}
                    for path, reason in run.unreadable
                ],
            },
            errors_path,
        )
        print("Unreadable-file report: %s" % errors_path)

    if args.diagnostics_output:
        write_json_atomic(diagnostic_payload, args.diagnostics_output)

    print("Predicted %d readable image(s)" % len(rows))
    if run.samples_per_second is not None:
        print("Throughput: %.2f images/s" % run.samples_per_second)
    print("Competition JSON: %s" % args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InferenceError, OSError, ValueError, KeyError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
