"""Command line entry point for the collection pipeline.

    PYTHONPATH=src uv run python -m collect plan   --run-id pilot-01
    PYTHONPATH=src uv run python -m collect run    --run-id pilot-01 --limit 20
    PYTHONPATH=src uv run python -m collect status --run-id pilot-01

``plan`` and ``status`` never call a provider, so they are safe to run at any
time. ``run`` spends money; it appends to ``<raw_dir>/<run-id>.jsonl`` and can
be interrupted and restarted with the same ``--run-id``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .config import CollectionSettings, ConfigError, load_collection_settings
from .monitor import render_daily_line, render_table, report_for_run
from .plan import build_layer_a_specs, build_layer_b_specs, load_design, summarize_plan
from .records import CallSpec
from .runner import run_collection
from .storage import raw_log_path, scan_raw_log


def build_specs(args: argparse.Namespace, settings: CollectionSettings) -> list[CallSpec]:
    """Load the frozen design and expand it into the calls this run will make."""

    design_dir = Path(args.design)
    design = load_design(
        queries_path=design_dir / "queries_tr.yaml",
        brands_path=design_dir / "brands_tr.yaml",
        protocols_path=design_dir / "protocols.yaml",
        variants_path=design_dir / "variants.yaml",
    )
    model_keys = args.models or [model.key for model in settings.models]
    for key in model_keys:
        settings.model(key)

    builder = build_layer_a_specs if args.layer == "A" else build_layer_b_specs
    specs = list(
        builder(
            design,
            run_id=args.run_id,
            protocol_name=args.protocol,
            model_keys=model_keys,
            seed=settings.seed,
        )
    )
    return specs[: args.limit] if args.limit else specs


def command_plan(args: argparse.Namespace) -> int:
    settings = load_collection_settings(args.config)
    specs = build_specs(args, settings)
    summary = summarize_plan(specs)
    summary["run_id"] = args.run_id
    summary["layer"] = args.layer
    summary["protocol"] = args.protocol
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.sample:
        print("\n--- örnek istem ---")
        print(specs[0].prompt if specs else "(plan boş)")
    return 0


def command_run(args: argparse.Namespace) -> int:
    settings = load_collection_settings(args.config)
    specs = build_specs(args, settings)
    if args.dry_run:
        print(json.dumps(summarize_plan(specs), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    summary = asyncio.run(run_collection(specs, settings, run_id=args.run_id))
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.failed == 0 and not summary.stopped_on_spend_cap else 1


def command_status(args: argparse.Namespace) -> int:
    settings = load_collection_settings(args.config)
    path = raw_log_path(settings.raw_dir, args.run_id)
    resume = scan_raw_log(path)
    status: dict[str, Any] = {
        "run_id": args.run_id,
        "log_path": str(path),
        "exists": path.exists(),
        "lines": resume.total_lines,
        "completed": len(resume.completed),
        "failed_pending_retry": len(resume.failed),
        "malformed_lines": list(resume.malformed_lines[:20]),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    settings = load_collection_settings(args.config)
    report = report_for_run(settings, args.run_id)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.format == "line":
        print(render_daily_line(report))
    else:
        print(render_daily_line(report))
        print()
        print(render_table(report))
    return 1 if report["spend_cap_exceeded"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collect", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect.yaml"),
        help="Collection runtime configuration (default: configs/collect.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for the run (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_design_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--run-id", required=True, help="Identifies and resumes a run")
        subparser.add_argument(
            "--design",
            type=Path,
            default=Path("configs"),
            help="Directory holding the frozen design files (default: configs)",
        )
        subparser.add_argument("--layer", choices=("A", "B"), default="A")
        subparser.add_argument("--protocol", default="alpha")
        subparser.add_argument(
            "--models",
            type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
            help="Comma-separated model keys (default: every configured model)",
        )
        subparser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Only plan or run the first N calls; use for the pilot",
        )

    plan_parser = subparsers.add_parser("plan", help="Show the call plan without spending")
    add_design_arguments(plan_parser)
    plan_parser.add_argument("--sample", action="store_true", help="Also print one rendered prompt")
    plan_parser.set_defaults(handler=command_plan)

    run_parser = subparsers.add_parser("run", help="Collect the plan, resuming if interrupted")
    add_design_arguments(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Plan only; make no calls")
    run_parser.set_defaults(handler=command_run)

    status_parser = subparsers.add_parser("status", help="Report what a run has already collected")
    status_parser.add_argument("--run-id", required=True)
    status_parser.set_defaults(handler=command_status)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Daily watch table: rate, errors, parse success, spend"
    )
    monitor_parser.add_argument("--run-id", required=True)
    monitor_parser.add_argument(
        "--format",
        choices=("table", "line", "json"),
        default="table",
        help="'line' prints the one-line summary for the group message",
    )
    monitor_parser.set_defaults(handler=command_monitor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    try:
        handler = args.handler
        return int(handler(args))
    except ConfigError as error:
        print(f"Yapılandırma hatası: {error}")
        return 2
    except KeyboardInterrupt:
        print("\nDurduruldu. Aynı --run-id ile yeniden başlatınca kaldığı yerden devam eder.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
