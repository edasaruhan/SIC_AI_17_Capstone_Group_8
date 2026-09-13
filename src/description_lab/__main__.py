"""CLI for the description experiment. Paid calls need `--yes`.

    make lab-plan       # design and call count, no API call
    make lab-pilot      # a few calls to check parsing and truncation (PAID)
    make lab-run        # the full plan, resumable from receipts (PAID)
    make lab-analyze    # effects and report from completed calls, no API call

Pass ``LAB_ARGS="--assistant cerebras --round 2"`` to choose the assistant and round.
Receipts live under ``data/processed/description_lab/<plan id>/steps``; each assistant
and round has its own plan id. The pilot's calls are keys of the full plan, so the run
reuses them. Reports go to ``reports/description_lab/<assistant>/round<n>``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evidence_eval.io import write_json

from . import analysis, clients, design, round2, round2_report

OUTPUT = Path("data/processed/description_lab")
REPORT = Path("reports/description_lab")


def _client(assistant: str, http):
    from visibility.llm import GeminiClient, load_key

    if assistant == "gemini":
        load_key()
        return GeminiClient(http)
    load_key(clients.KEY_ENV)
    return clients.CerebrasClient(http)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "pilot", "run", "analyze"])
    parser.add_argument("--assistant", choices=sorted(design.ASSISTANTS), default="gemini")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    parser.add_argument("--reps", type=int, default=design.REPS)
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıları onayla")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    if args.round == 1:
        plan = design.plan_id(args.assistant)
        jobs = design.plan_jobs(args.reps, args.assistant)
        pilot = design.pilot_jobs(args.assistant)
    else:
        plan = round2.plan_id(args.assistant)
        jobs = round2.plan_jobs(args.assistant)
        pilot = jobs[:12]
    folder = args.output / plan
    print(
        json.dumps(
            {
                "assistant": args.assistant,
                "round": args.round,
                "id": plan,
                "calls": len(jobs),
                "pilot": len(pilot),
            },
            ensure_ascii=False,
        )
    )

    if args.command == "plan":
        print("API çağrısı yapılmadı.")
        return 0

    if args.command in {"pilot", "run"}:
        if not args.yes:
            print("Ücretli çağrılar için --yes gerekli; hiçbir çağrı yapılmadı.")
            return 1
        import httpx

        folder.mkdir(parents=True, exist_ok=True)
        chosen = pilot if args.command == "pilot" else jobs

        async def execute() -> list[dict]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as http:
                return await clients.collect(
                    chosen,
                    folder,
                    _client(args.assistant, http),
                    args.assistant,
                    concurrency=args.concurrency,
                    retry_failed=args.retry_failed,
                )

        failures = asyncio.run(execute())
        write_json(folder / f"failures_{args.command}.json", failures)
        print(f"{len(chosen) - len(failures)}/{len(chosen)} çağrı tamam; hata: {len(failures)}")

    report = REPORT / args.assistant / f"round{args.round}"
    if args.round == 1:
        table = analysis.outcomes(folder, jobs)
        if table.empty:
            print("Tamamlanmış çağrı yok.")
            return 1
        if args.command != "analyze":
            print(table.groupby(["category", "identity", "variant"])["first"].mean().round(2))
            return 0
        report.mkdir(parents=True, exist_ok=True)
        identity = analysis.identity_effects(table)
        content = analysis.content_effects(table)
        position = analysis.position_effects(table)
        table.to_csv(report / "outcomes.csv", index=False)
        identity.to_csv(report / "identity_effects.csv", index=False)
        content.to_csv(report / "content_effects.csv", index=False)
        position.to_csv(report / "position_effects.csv", index=False)
        analysis.pick_share(table).to_csv(report / "pick_position.csv", index=False)
        text = analysis.render(
            table, identity, content, position, planned=len(jobs), assistant=args.assistant
        )
    else:
        table = round2_report.outcomes(folder, jobs)
        if table.empty:
            print("Tamamlanmış çağrı yok.")
            return 1
        if args.command != "analyze":
            print(table.groupby(["block", "category"])["picked_sentence"].value_counts())
            return 0
        report.mkdir(parents=True, exist_ok=True)
        first_round = analysis.outcomes(
            args.output / design.plan_id(args.assistant),
            design.plan_jobs(design.REPS, args.assistant),
        )
        shares = round2_report.win_shares(table)
        filler = round2_report.filler_contrast(table, first_round)
        cross = round2_report.cross_shares(table)
        echo = round2_report.echo_rates(table)
        table.to_csv(report / "outcomes.csv", index=False)
        shares.to_csv(report / "win_shares.csv", index=False)
        filler.to_csv(report / "filler_contrast.csv", index=False)
        cross.to_csv(report / "cross_shares.csv", index=False)
        echo.to_csv(report / "fabricated_claim.csv", index=False)
        text = round2_report.render(
            table, shares, filler, cross, echo, assistant=args.assistant, planned=len(jobs)
        )
    (report / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {report}/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
