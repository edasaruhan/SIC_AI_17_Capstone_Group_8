"""CLI entry point for offline evidence preparation, audits and reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import baselines, listwise, report, review, workspace
from .io import clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=workspace.DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--english", type=Path, default=Path("data/interim/reference.parquet"))
        sub.add_argument("--turkish", type=Path, default=Path("data/interim/turkish_raw.parquet"))
    commands.add_parser("sample")
    sub = commands.add_parser("review-check")
    sub.add_argument("--require-complete", action="store_true")
    sub = commands.add_parser("baselines")
    sub.add_argument("--tracks", nargs="+", choices=["en", "tr"], default=["en", "tr"])
    sub.add_argument(
        "--targets", nargs="+", choices=["y_top", "y_mention"], default=["y_top", "y_mention"]
    )
    sub = commands.add_parser("report")
    sub.add_argument("--brand", required=True)
    sub.add_argument("--domain", choices=["vpn", "cosmetics"], required=True)
    sub.add_argument("--language", choices=["en", "tr"], default="tr")
    for command in ("m3-smoke", "m3-train"):
        sub = commands.add_parser(command)
        sub.add_argument("--track", choices=["en", "tr"], default="tr")
        sub.add_argument("--category", choices=["vpn", "cosmetics"], default="vpn")
        sub.add_argument("--model-revision", required=True)
        sub.add_argument("--allow-download", action="store_true")
        sub.add_argument("--epochs", type=int, default=2)
        sub.add_argument("--max-length", type=int, default=128)
        sub.add_argument("--candidate-batch", type=int, default=4)
        sub.add_argument("--seed", type=int, default=7)
        sub.set_defaults(smoke=command == "m3-smoke")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = workspace.prepare(args.root, args.english, args.turkish)
        elif args.command == "validate":
            value = workspace.validate_inputs(args.english, args.turkish)
            if (args.root / "manifest.json").exists():
                workspace.verify(args.root)
        elif args.command == "sample":
            value = review.sample(args.root)
        elif args.command == "review-check":
            _, value = review.checked_annotations(args.root, require_complete=args.require_complete)
        elif args.command == "baselines":
            value = baselines.train_baselines(args.root, args.tracks, args.targets)
        elif args.command == "report":
            value = report.save_report(args.root, args.brand, args.domain, args.language)
        else:
            value = listwise.train(args.root, args)
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
