#!/usr/bin/env python
"""Benchmark throughput, VRAM and time-to-first-batch for each backbone.

    python scripts/benchmark_gpu.py --config configs/gpu_check.yaml
    python scripts/benchmark_gpu.py --config configs/gpu_check.yaml \
        --backbones dinov2 --batch-sizes 16 32 64 --modes train

Same exit codes as ``scripts/gpu_check.py``: ``0`` clean, ``1`` blocking,
``2`` bad usage. Configurations that run out of memory are reported as
findings, not crashes, so one sweep can answer "what is the largest batch this
device holds?". Any other CUDA error aborts with the failing backbone,
architecture, precision and batch size attached.

Smoke and determinism checks are skipped here; run ``gpu_check.py`` for those.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.gpu_check import apply_overrides, build_parser, run  # noqa: E402


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = build_parser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="override benchmark.batch_sizes, e.g. --batch-sizes 8 16 32 64",
    )
    parser.add_argument("--precisions", nargs="+", default=None, choices=["fp32", "amp"])
    parser.add_argument("--modes", nargs="+", default=None, choices=["train", "inference"])
    parser.add_argument(
        "--measure-steps", type=int, default=None, help="timed steps per configuration"
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None, help="override model.image_size")
    parser.set_defaults(basename="benchmark")
    return parser


def benchmark_overrides(raw: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Shared CLI flags plus the benchmark-only sweep flags."""
    config = apply_overrides(raw, args)
    benchmark = dict(config.get("benchmark") or {})
    model = dict(config.get("model") or {})
    if args.batch_sizes:
        benchmark["batch_sizes"] = list(args.batch_sizes)
    if args.precisions:
        benchmark["precisions"] = list(args.precisions)
    if args.modes:
        benchmark["modes"] = list(args.modes)
    if args.measure_steps is not None:
        benchmark["measure_steps"] = args.measure_steps
    if args.warmup_steps is not None:
        benchmark["warmup_steps"] = args.warmup_steps
    if args.image_size is not None:
        model["image_size"] = args.image_size
    if benchmark:
        config["benchmark"] = benchmark
    if model:
        config["model"] = model
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_benchmark_parser().parse_args(argv)
    return run(
        args,
        benchmarks=True,
        determinism=False,
        smoke=False,
        overrides=benchmark_overrides,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
