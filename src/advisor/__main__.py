"""CLI for the advisor graph. Paid work needs `--yes` and stays inside `--max-calls`.

    PYTHONPATH=src uv run --with langgraph python -m advisor \
        --brand "Garanti BBVA" --sector "bankacılık" --language tr --yes

Any sector works. Competitors are extracted from the search results; correct them with
``--add-rival`` / ``--drop-rival`` and rerun -- cached calls are not paid for again.
Train the transferable model once first: ``make advisor-train``.

A run writes its receipts under ``data/processed/advisor/<run id>/steps`` and its
report beside them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from brand_demo.workflow import Receipts
from evidence_eval.io import digest, write_json, write_text

from . import features, model, nodes
from .clients import AdvisorClient, require_keys

OUTPUT = Path("data/processed/advisor")


def plan_id(args: argparse.Namespace) -> str:
    """Corrections change the candidate set, so they change the run identity."""
    return digest(
        {
            "brand": args.brand,
            "aliases": sorted(args.brand_alias),
            "sector": args.sector,
            "language": args.language,
            "queries": args.queries,
            "reps": args.reps,
            "model": nodes.MODEL,
            # Every setting that shapes a paid payload belongs to the run identity, or a
            # changed setting collides with cached receipts in the same folder.
            "temperature": nodes.TEMPERATURE,
            "max_tokens": nodes.MAX_TOKENS,
        }
    )[:16]


def estimated_calls(args: argparse.Namespace) -> int:
    recorded = nodes.recorded_queries(args.sector, args.language, args.queries)
    generation = 0 if recorded else 1
    return generation + args.queries * (1 + 2 * args.reps) + 1  # +1 candidate extraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", required=True)
    parser.add_argument("--sector", required=True, help="Serbest metin, ör. 'bankacılık'")
    parser.add_argument("--language", default="tr", choices=("tr", "en"))
    parser.add_argument(
        "--brand-alias", action="append", default=[], help="Markanın diğer yazılışı"
    )
    parser.add_argument(
        "--add-rival", action="append", default=[], help="Çıkarımın kaçırdığı rakip"
    )
    parser.add_argument("--drop-rival", action="append", default=[], help="Yanlış çıkarılan ad")
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıları onayla")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    calls = estimated_calls(args)
    folder = args.output / plan_id(args)
    print(f"plan={folder.name} sektör={args.sector!r} tahmini çağrı={calls}")
    try:
        boosters = model.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Öğrenilmiş sinyal modeli yüklenemedi: {exc}")
        return 1
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
            runtime = nodes.Runtime(
                receipts=Receipts(folder, AdvisorClient(http), args.retry_failed),
                boosters=boosters,
                rules=features.load_rules(),
                brand_aliases=args.brand_alias,
                add_rivals=args.add_rival,
                drop_rivals=args.drop_rival,
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
            key: state.get(key)
            for key in (
                "brand",
                "sector",
                "language",
                "curated",
                "queries",
                "candidates",
                "diagnosis",
                "measures",
                "scores",
                "signals",
                "notes",
            )
        }
        | {"model": nodes.MODEL, "corrections": {"add": args.add_rival, "drop": args.drop_rival}},
    )
    print(state["report"])
    print(f"\nwrote {folder}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
