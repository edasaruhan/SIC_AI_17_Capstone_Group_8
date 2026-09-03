"""Command-line entry point. Subcommands are registered by their owning modules."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import load_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bias-eval")
    parser.add_argument("--config", default="configs/evaluation/suite.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the frozen 400-cell plan without API calls")
    subparsers.add_parser("preflight", help="Check credentials and model-catalog connectivity")
    search_test = subparsers.add_parser("search-test", help="Spend one Serper credit on a test")
    search_test.add_argument("query", nargs="?", default="2026 en iyi VPN Türkiye")
    run = subparsers.add_parser(
        "run", help="Collect generation responses, resuming completed cells"
    )
    run.add_argument("--pilot", action="store_true", help="Run the 16 production pilot cells")
    subparsers.add_parser("status", help="Summarize raw generation progress")
    subparsers.add_parser("judge", help="Judge successful generation records with Cerebras")
    subparsers.add_parser("judge-status", help="Summarize current and stale judgments")
    subparsers.add_parser("export", help="Write reference-compatible Parquet files")
    subparsers.add_parser("validate", help="Require the complete 400 + 400-row result")
    report = subparsers.add_parser("report", help="Compare Turkish VPN with English reference")
    report.add_argument("--reference", default="data/interim/reference.parquet")
    report.add_argument("--output", default="reports/tr_vpn_language_comparison.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        from .planner import render_plan

        print(render_plan(args.config))
        return 0
    config = load_suite(args.config)
    if args.command == "preflight":
        from .preflight import check_connections

        result = asyncio.run(check_connections(config))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if all(value.get("ok") for value in result.values()) else 1
    if args.command == "search-test":
        import httpx

        from .search import SerperSearch

        async def run_search_test() -> dict[str, object]:
            async with httpx.AsyncClient(timeout=30.0) as http:
                hit = await SerperSearch(config.search, config.search_cache_dir, http).search(
                    args.query
                )
                return {
                    "query": hit.query,
                    "source": hit.source,
                    "organic_results": len(hit.results.get("organic") or []),
                }

        print(json.dumps(asyncio.run(run_search_test()), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        from .runner import run_suite

        result = asyncio.run(run_suite(config, pilot=args.pilot))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["errors"] == 0 and result["blocked"] == 0 else 1
    if args.command == "status":
        from .storage import status_summary

        print(
            json.dumps(
                status_summary(config.raw_dir, config.expected_rows),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "judge":
        from .judge import judge_suite

        result = asyncio.run(judge_suite(config))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["errors"] == 0 and result["blocked"] == 0 else 1
    if args.command == "judge-status":
        from .judge import judge_status

        print(json.dumps(judge_status(config), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "export":
        from .exporter import export_dataset

        print(json.dumps(export_dataset(config), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        from .validation import validate_dataset, validation_exit_code

        result = validate_dataset(config)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return validation_exit_code(result)
    if args.command == "report":
        from .report import build_report

        result = build_report(
            config.processed_dir / "vpn" / "train.parquet",
            Path(args.reference),
            Path(args.output),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 2
