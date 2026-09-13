"""CLI for the description experiment. Paid calls need `--yes`.

    make lab-plan       # design and call count, no API call
    make lab-pilot      # a few calls to check parsing and truncation (PAID)
    make lab-run        # the full plan, resumable from receipts (PAID)
    make lab-analyze    # effects and report from completed calls, no API call

Receipts live under ``data/processed/description_lab/<plan id>/steps``. The pilot's
calls are keys of the full plan, so the run reuses them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evidence_eval.io import write_json

from . import analysis, design

OUTPUT = Path("data/processed/description_lab")
REPORT = Path("reports/description_lab")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "pilot", "run", "analyze"])
    parser.add_argument("--reps", type=int, default=design.REPS)
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıları onayla")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    folder = args.output / design.plan_id()
    jobs = design.plan_jobs(args.reps)
    pilot = design.pilot_jobs()
    print(
        json.dumps(
            {"id": design.plan_id(), "calls": len(jobs), "pilot": len(pilot), "reps": args.reps},
            ensure_ascii=False,
        )
    )

    if args.command == "plan":
        print("API çağrısı yapılmadı.")
        return 0

    if args.command in {"pilot", "run"}:
        if not args.yes:
            print("Ücretli Gemini çağrıları için --yes gerekli; hiçbir çağrı yapılmadı.")
            return 1
        import httpx

        from visibility.intervention import collect
        from visibility.llm import GeminiClient, load_key

        load_key()
        folder.mkdir(parents=True, exist_ok=True)
        chosen = pilot if args.command == "pilot" else jobs

        async def execute() -> list[dict]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as http:
                return await collect(
                    chosen,
                    folder,
                    GeminiClient(http),
                    concurrency=args.concurrency,
                    retry_failed=args.retry_failed,
                )

        failures = asyncio.run(execute())
        write_json(folder / f"failures_{args.command}.json", failures)
        print(f"{len(chosen) - len(failures)}/{len(chosen)} çağrı tamam; hata: {len(failures)}")

    table = analysis.outcomes(folder, jobs)
    if table.empty:
        print("Tamamlanmış çağrı yok.")
        return 1
    if args.command == "analyze":
        REPORT.mkdir(parents=True, exist_ok=True)
        identity = analysis.identity_effects(table)
        content = analysis.content_effects(table)
        position = analysis.position_effects(table)
        table.to_csv(REPORT / "outcomes.csv", index=False)
        identity.to_csv(REPORT / "identity_effects.csv", index=False)
        content.to_csv(REPORT / "content_effects.csv", index=False)
        position.to_csv(REPORT / "position_effects.csv", index=False)
        (REPORT / "README.md").write_text(
            analysis.render(table, identity, content, position, planned=len(jobs)),
            encoding="utf-8",
        )
        print(f"wrote {REPORT}/README.md")
    else:
        print(table.groupby(["category", "identity", "variant"])["first"].mean().round(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
