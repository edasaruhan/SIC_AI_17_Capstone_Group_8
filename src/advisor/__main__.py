"""CLI for the advisor graph. Paid work needs `--yes` and stays inside `--max-calls`.

    PYTHONPATH=src uv run --with langgraph python -m advisor \
        --brand "Windscribe" --sector vpn --language tr --yes

A run writes its receipts under ``data/processed/advisor/<run id>/steps`` and its
report beside them. Rerunning the same brand reads the cache instead of paying again.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from brand_demo.workflow import Receipts
from evidence_eval.io import digest, write_json, write_text

from . import measure, nodes
from .clients import AdvisorClient, require_keys

OUTPUT = Path("data/processed/advisor")


def plan_id(args: argparse.Namespace) -> str:
    return digest(
        {
            "brand": args.brand,
            "sector": args.sector,
            "language": args.language,
            "queries": args.queries,
            "reps": args.reps,
            "model": nodes.MODEL,
        }
    )[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", required=True)
    parser.add_argument("--sector", required=True, choices=measure.SECTORS)
    parser.add_argument("--language", default="tr", choices=("tr", "en"))
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıları onayla")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    calls = args.queries * (1 + 2 * args.reps)
    folder = args.output / plan_id(args)
    print(
        f"plan={folder.name} tahmini çağrı={calls} (arama {args.queries} + asistan {calls - args.queries})"
    )
    if not args.yes:
        print("Ücretli çağrılar için --yes gerekli; hiçbir çağrı yapılmadı.")
        return 1
    if calls > args.max_calls:
        print(f"Tahmini {calls} çağrı, --max-calls={args.max_calls} sınırının üstünde.")
        return 1

    require_keys()
    folder.mkdir(parents=True, exist_ok=True)

    async def run() -> dict:
        from .graph import build

        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as http:
            receipts = Receipts(folder, AdvisorClient(http), args.retry_failed)
            runtime = nodes.Runtime(
                receipts=receipts,
                reps=args.reps,
                n_queries=args.queries,
                concurrency=args.concurrency,
            )
            graph = build(runtime, args.max_calls)
            return await graph.ainvoke(
                {
                    "brand": args.brand,
                    "sector": args.sector,
                    "language": args.language,
                    "max_calls": args.max_calls,
                }
            )

    state = asyncio.run(run())
    write_text(folder / "report.md", state["report"])
    write_json(
        folder / "run.json",
        {
            "brand": args.brand,
            "sector": args.sector,
            "language": args.language,
            "model": nodes.MODEL,
            "diagnosis": state.get("diagnosis"),
            "measures": state.get("measures"),
            "queries": state.get("queries"),
            "notes": state.get("notes"),
        },
    )
    print(state["report"])
    print(f"\nwrote {folder}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
