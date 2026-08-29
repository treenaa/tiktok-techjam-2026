"""Command-line data audit -- run this against the real corpus before training.

Four subcommands::

    python -m src.data.audit_cli splits   --config configs/baseline_clip.yaml
    python -m src.data.audit_cli splits   --train t.csv --val v.csv --test s.csv
    python -m src.data.audit_cli shortcut --config configs/baseline_clip.yaml
    python -m src.data.audit_cli verify   --input ./images
    python -m src.data.audit_cli compare  configs/baseline_*.yaml

Exit codes: ``0`` clean, ``1`` a blocking problem (leakage, protected data in
training, incomparable runs), ``2`` bad usage.  So it drops straight into CI.

Advisory shortcut findings do not fail the run by default -- ``--strict`` makes
critical findings blocking.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .audit import audit_shortcuts, format_audit_report
from .config import DatasetConfigError
from .schema import DataError
from .experiment import comparability_report, load_experiment
from .loading import list_images, verify_images
from .manifest import read_manifest
from .protected import protected_report
from .splitting import LeakageError, format_split_report
from .validation import validate_splits

__all__ = ["main"]

_OK = 0
_FAIL = 1
_USAGE = 2


def _emit(payload: Dict[str, Any], as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


def _splits_from_config(path: str):
    config = load_experiment(path) if _is_experiment(path) else None
    if config is not None:
        return config.build_splits(validate=False)
    from .config import build_from_config

    return build_from_config(path, validate=False)


def _is_experiment(path: str) -> bool:
    """An experiment config nests its dataset spec under ``data:``."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return False
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        try:
            import yaml

            raw = yaml.safe_load(text)
        except Exception:
            return False
    else:
        try:
            raw = json.loads(text)
        except ValueError:
            return False
    return isinstance(raw, dict) and "data" in raw


# --------------------------------------------------------------------------
def cmd_splits(args: argparse.Namespace) -> int:
    """Leakage gate: source_id overlap, paths, derivatives, protected data."""
    if args.config:
        splits = _splits_from_config(args.config)
        kwargs = {
            "train_manifest": splits.get("train"),
            "val_manifest": splits.get("val"),
            "test_manifest": splits.get("test"),
            "extra_splits": {k: v for k, v in splits.items()
                             if k not in ("train", "val", "test")} or None,
        }
    else:
        if not args.train:
            print("error: pass --config or at least --train", file=sys.stderr)
            return _USAGE
        kwargs = {
            "train_manifest": args.train,
            "val_manifest": args.val,
            "test_manifest": args.test,
        }

    report = validate_splits(
        check_paths_exist=args.check_files,
        raise_on_failure=False,
        **kwargs
    )
    payload = {
        "ok": report.ok,
        "stats": report.stats,
        "problems": {k: v for k, v in report.problems.items()},
    }
    _emit(payload, args.json, report.summary())
    if report.ok and not args.json:
        print("\n" + format_split_report(report.stats))
    return _OK if report.ok else _FAIL


def cmd_shortcut(args: argparse.Namespace) -> int:
    """Rule 11.C: how easily could a model cheat on this corpus?"""
    if args.config:
        splits = _splits_from_config(args.config)
        records = splits.get(args.split) or []
        if not records:
            print("error: split %r is empty or absent" % args.split, file=sys.stderr)
            return _USAGE
    elif args.manifest:
        records = read_manifest(args.manifest, root=args.root)
    else:
        print("error: pass --config or --manifest", file=sys.stderr)
        return _USAGE

    report = audit_shortcuts(
        records,
        root=args.root,
        inspect_files=not args.no_files,
        sample_size=args.sample,
        raise_on_critical=False,
    )
    report["protected"] = protected_report(records)
    _emit(report, args.json, format_audit_report(report))

    if not args.json and report["protected"]["n_protected"]:
        print("\n  NOTE: %d protected (demonstration-only) record(s) present"
              % report["protected"]["n_protected"])
    if args.strict and report["n_critical"]:
        return _FAIL
    return _OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Check every image in a directory or manifest is readable."""
    if args.input:
        paths = list_images(args.input, recursive=not args.flat)
    elif args.manifest:
        paths = [r.resolve_path(args.root) for r in read_manifest(args.manifest, root=args.root)]
    else:
        print("error: pass --input or --manifest", file=sys.stderr)
        return _USAGE

    report = verify_images(paths)
    payload = {
        "n_checked": report["n_checked"],
        "n_readable": report["n_readable"],
        "unreadable": [{"image_path": p, "reason": r} for p, r in report["unreadable"]],
    }
    text = ["%d/%d images readable" % (report["n_readable"], report["n_checked"])]
    for path, reason in report["unreadable"][:20]:
        text.append("  UNREADABLE %s -- %s" % (path, reason))
    if len(report["unreadable"]) > 20:
        text.append("  ... and %d more" % (len(report["unreadable"]) - 20))
    _emit(payload, args.json, "\n".join(text))
    return _OK if not report["unreadable"] else _FAIL


def cmd_compare(args: argparse.Namespace) -> int:
    """Rule 21: are these runs comparable (differing only in `model`)?"""
    paths: List[str] = []
    for pattern in args.configs:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    if len(paths) < 2:
        print("error: need at least two configs", file=sys.stderr)
        return _USAGE

    configs = [load_experiment(p) for p in paths]
    report = comparability_report(configs)

    lines = ["comparing %d runs: %s" % (len(configs), ", ".join(c.name for c in configs))]
    for name, fingerprint in report["fingerprints"].items():
        lines.append("  %-24s fingerprint %s" % (name, fingerprint))
    if report["comparable"]:
        lines.append("\nCOMPARABLE: runs differ only in `model` -- "
                     "a metric gap is attributable to the backbone.")
    else:
        lines.append("\nNOT COMPARABLE (rule 21) -- these differ outside `model`:")
        for pair, diffs in report["differences"].items():
            lines.append("  %s" % pair)
            lines.extend("    - %s" % d for d in diffs[:10])
            if len(diffs) > 10:
                lines.append("    ... and %d more" % (len(diffs) - 10))
    if report["identical_models"]:
        lines.append("\nWARNING: identical model sections (vacuous comparison): %s"
                     % report["identical_models"])

    _emit(report, args.json, "\n".join(lines))
    return _OK if report["comparable"] else _FAIL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.audit_cli",
        description="Audit data splits, shortcut risk, image readability and run comparability.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("splits", help="leakage + protected-data gate")
    p.add_argument("--config")
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--test")
    p.add_argument("--check-files", action="store_true", help="also verify files exist")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_splits)

    p = subparsers.add_parser("shortcut", help="dataset-shortcut audit (rule 11.C)")
    p.add_argument("--config")
    p.add_argument("--manifest")
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--no-files", action="store_true", help="metadata-only audit")
    p.add_argument("--strict", action="store_true", help="critical findings fail the run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_shortcut)

    p = subparsers.add_parser("verify", help="check images are readable")
    p.add_argument("--input", help="image directory")
    p.add_argument("--manifest")
    p.add_argument("--root", default=None)
    p.add_argument("--flat", action="store_true", help="do not recurse")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = subparsers.add_parser("compare", help="are runs comparable? (rule 21)")
    p.add_argument("configs", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return _USAGE
    try:
        return args.func(args)
    except (LeakageError, DatasetConfigError, DataError, OSError) as exc:
        # Expected, user-facing failures: report them cleanly rather than
        # dumping a traceback. Anything else is a bug and should propagate.
        print("error: %s" % exc, file=sys.stderr)
        return _FAIL


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
