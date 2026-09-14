"""CLI for the advisor graph. Paid work needs `--yes` and stays inside `--max-calls`.

    PYTHONPATH=src uv run --with langgraph python -m advisor \
        --brand "Garanti BBVA" --sector "bankacılık" --language tr --yes

Any sector works. Competitors are extracted from the search results; correct them with
``--add-rival`` / ``--drop-rival`` and rerun -- cached calls are not paid for again.
Train the transferable model once first: ``make advisor-train``. For the interface:
``make advisor-ui``.

A run writes its receipts under ``data/processed/advisor/<run id>/steps`` and its
report beside them. The run itself lives in ``service.py``, shared with the interface.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import model, service
from .clients import require_keys


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
    parser.add_argument("--output", type=Path, default=service.OUTPUT)
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıları onayla")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--description", default="", help="Ürün açıklaması (denetim için)")
    parser.add_argument("--description-file", type=Path, help="Ürün açıklaması dosyası")
    parser.add_argument(
        "--audit-ai", action="store_true", help="Açıklamayı Gemini ile sınıflandır (+1 çağrı)"
    )
    parser.add_argument(
        "--second-assistant", action="store_true", help="gpt-oss-120b (Cerebras) ile de ölç"
    )
    args = parser.parse_args(argv)
    description = (
        args.description_file.read_text(encoding="utf-8")
        if args.description_file
        else args.description
    )

    options = service.Options(
        brand=args.brand,
        sector=args.sector,
        language=args.language,
        brand_aliases=tuple(args.brand_alias),
        add_rivals=tuple(args.add_rival),
        drop_rivals=tuple(args.drop_rival),
        queries=args.queries,
        reps=args.reps,
        max_calls=args.max_calls,
        concurrency=args.concurrency,
        retry_failed=args.retry_failed,
        output=args.output,
        description=description,
        audit_ai=args.audit_ai,
        assistants=("gemini", "cerebras") if args.second_assistant else ("gemini",),
    )
    calls = service.estimated_calls(options)
    print(f"plan={service.run_id(options)} sektör={args.sector!r} tahmini çağrı={calls}")
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

    require_keys(options.assistants)
    final = asyncio.run(service.run(options, boosters))
    print(final["report"])
    print(f"\nwrote {service.run_folder(options)}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
