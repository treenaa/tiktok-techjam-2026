#!/usr/bin/env python
"""Verify a GPU instance before anyone starts a real training run.

    python scripts/gpu_check.py --config configs/gpu_check.yaml
    python scripts/gpu_check.py --config configs/gpu_check_smoke.yaml --allow-cpu
    python scripts/gpu_check.py --config configs/gpu_check.yaml --deterministic --strict

Exit codes follow ``src.data.audit_cli``: ``0`` clean, ``1`` a blocking
problem, ``2`` bad usage -- so this drops straight into CI or a pre-run gate.
``--strict`` promotes warnings (an OOM-prone config, a loose determinism knob)
to failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_OK = 0
_FAIL = 1
_USAGE = 2

DEFAULT_OUTPUT_DIR = os.path.join("reports", "gpu")


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="YAML/JSON GPU-check config")
    parser.add_argument("--device", default=None, help="override config device, e.g. cuda:0")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run the checks on CPU when no GPU is present (clearly labelled in the report)",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless a CUDA device is actually used",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic cuDNN/cuBLAS kernels for this run",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="subset of backbones to check, e.g. --backbones dinov2 clip",
    )
    parser.add_argument(
        "--architectures", nargs="+", default=None, choices=["visual", "fusion"]
    )
    parser.add_argument(
        "--backbone-source",
        default=None,
        choices=["stub", "pretrained"],
        help="'stub' avoids all downloads; 'pretrained' is what the real run uses",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="gpu_report", help="report filename stem")
    parser.add_argument("--no-report", action="store_true", help="do not write report files")
    parser.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    parser.add_argument("--quiet", action="store_true", help="suppress the text summary")
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures for the exit code"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="environment, smoke and determinism only (seconds instead of minutes)",
    )
    parser.add_argument(
        "--skip-determinism", action="store_true", help="skip the repeat-run comparison"
    )
    return parser


def apply_overrides(raw: dict, args: argparse.Namespace) -> dict:
    """Fold CLI flags into the loaded config mapping."""
    config = dict(raw)
    model = dict(config.get("model") or {})
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.deterministic:
        config["deterministic"] = True
    if args.allow_cpu:
        config["allow_cpu"] = True
        config["require_cuda"] = False
    if args.require_cuda:
        config["require_cuda"] = True
        config["allow_cpu"] = False
    if args.backbones:
        model["backbones"] = list(args.backbones)
    if args.architectures:
        model["architectures"] = list(args.architectures)
    if args.backbone_source:
        model["backbone_source"] = args.backbone_source
    if model:
        config["model"] = model
    return config


def _config_mapping(path: Optional[str]) -> dict:
    """Read the config file as a plain mapping so CLI flags can be folded in."""
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError("gpu config not found: %s" % path)
    text = open(path, encoding="utf-8").read()
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("gpu config root must be a mapping, got %r" % type(raw).__name__)
    return raw


def prepare_environment(deterministic: bool) -> List[str]:
    """Set process-level knobs that only work before CUDA initialises.

    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created, so
    setting it from Python after the first CUDA call silently does nothing.
    Setting it here, before ``torch`` is imported, is the only place it works.
    """
    notes: List[str] = []
    if deterministic and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        notes.append(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch, as deterministic "
            "cuBLAS reductions require"
        )
    return notes


def run(
    args: argparse.Namespace,
    *,
    benchmarks: bool,
    determinism: bool,
    smoke: bool,
    overrides: Optional[Callable[[Dict, argparse.Namespace], Dict]] = None,
) -> int:
    """Shared body for gpu_check.py and benchmark_gpu.py.

    ``overrides`` lets a wrapper script fold in its own flags; it defaults to
    :func:`apply_overrides`.
    """
    notes = prepare_environment(args.deterministic)

    # Imported after prepare_environment so the CUDA-time env vars take effect.
    from src.gpu import GpuCheckError, config_from_mapping, render_text, run_gpu_checks, write_report
    from src.gpu.config import GpuConfigError

    try:
        raw = _config_mapping(args.config)
        config = config_from_mapping((overrides or apply_overrides)(raw, args))
    except (GpuConfigError, FileNotFoundError, ValueError) as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return _USAGE

    try:
        report = run_gpu_checks(
            config,
            include_smoke=smoke,
            include_benchmarks=benchmarks,
            include_determinism=determinism,
        )
    except GpuCheckError as exc:
        # Deliberately not swallowed: print the full chained traceback so the
        # failing backbone, architecture and batch size stay attached.
        print("GPU check aborted: %s" % exc, file=sys.stderr)
        import traceback

        traceback.print_exc()
        return _FAIL

    report.notes.extend(notes)
    paths = None
    if not args.no_report:
        paths = write_report(
            report, args.output_dir, basename=args.basename, strict=args.strict
        )

    if args.json:
        print(json.dumps(report.to_dict(strict=args.strict), indent=2, sort_keys=True, default=str))
    elif not args.quiet:
        print(render_text(report, strict=args.strict))
    if paths and not args.quiet:
        print("report written to %s and %s" % (paths["json"], paths["text"]), file=sys.stderr)

    return _FAIL if report.status(strict=args.strict) == "fail" else _OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        args,
        benchmarks=not args.skip_benchmarks,
        determinism=not args.skip_determinism,
        smoke=True,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
