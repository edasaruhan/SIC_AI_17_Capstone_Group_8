"""Daily collection monitoring (S2-3).

Answers the four questions on the watch checklist from one pass over the raw
log: is the call rate on plan, has the error rate risen and for which model,
is spend under the cap, and is the share of unparseable responses normal.

Reading is the only thing this module does to ``data/raw`` - the capture files
are never rewritten.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parse.protocol import parse_response

from .config import CollectionSettings
from .storage import raw_log_path


@dataclass
class ModelStats:
    """One model's numbers for one run."""

    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    parsed: int = 0
    unparsed: int = 0
    spend_usd: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)
    model_versions: set[str] = field(default_factory=set)

    @property
    def error_rate(self) -> float:
        return self.failed / self.calls if self.calls else 0.0

    @property
    def parse_rate(self) -> float:
        attempted = self.parsed + self.unparsed
        return self.parsed / attempted if attempted else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "error_rate": round(self.error_rate, 4),
            "parsed": self.parsed,
            "unparsed": self.unparsed,
            "parse_rate": round(self.parse_rate, 4),
            "spend_usd": round(self.spend_usd, 4),
            "errors": dict(sorted(self.errors.items())),
            "model_versions": sorted(self.model_versions),
        }


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every readable capture row; unreadable lines are skipped here and
    counted by :func:`build_report`, never dropped without a number."""

    log_path = Path(path)
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                yield {"__malformed__": True}
                continue
            if isinstance(payload, dict):
                yield payload


def build_report(
    path: str | Path,
    *,
    run_id: str,
    spend_cap_usd: float | None = None,
) -> dict[str, Any]:
    """Summarise one run's capture file, overall and per model and per day."""

    by_model: dict[str, ModelStats] = defaultdict(ModelStats)
    by_day: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "spend_usd": 0.0})
    malformed = 0
    latest_day = ""

    for record in iter_records(path):
        if record.get("__malformed__"):
            malformed += 1
            continue

        model_key = str(record.get("model_key", "unknown"))
        stats = by_model[model_key]
        stats.calls += 1

        day = str(record.get("started_at", ""))[:10]
        latest_day = max(latest_day, day)
        by_day[day]["calls"] += 1

        if record.get("status") == "ok":
            stats.succeeded += 1
            stats.spend_usd += float(record.get("cost_usd") or 0.0)
            by_day[day]["spend_usd"] += float(record.get("cost_usd") or 0.0)
            version = record.get("model_version")
            if version:
                stats.model_versions.add(str(version))
            result = parse_response(
                record.get("response_text"),
                protocol=str(record.get("protocol", "alpha")),
                candidates=tuple(record.get("candidates") or ()),
            )
            if result.succeeded:
                stats.parsed += 1
            else:
                stats.unparsed += 1
                key = f"parse:{result.failure_reason}"
                stats.errors[key] = stats.errors.get(key, 0) + 1
        else:
            stats.failed += 1
            key = str(record.get("error_type") or "unknown")
            stats.errors[key] = stats.errors.get(key, 0) + 1

    totals = ModelStats()
    for stats in by_model.values():
        totals.calls += stats.calls
        totals.succeeded += stats.succeeded
        totals.failed += stats.failed
        totals.parsed += stats.parsed
        totals.unparsed += stats.unparsed
        totals.spend_usd += stats.spend_usd

    today = by_day.get(latest_day, {"calls": 0.0, "spend_usd": 0.0})
    return {
        "run_id": run_id,
        "log_path": str(path),
        "malformed_lines": malformed,
        "totals": totals.as_dict(),
        "by_model": {model: stats.as_dict() for model, stats in sorted(by_model.items())},
        "by_day": {
            day: {"calls": int(values["calls"]), "spend_usd": round(values["spend_usd"], 4)}
            for day, values in sorted(by_day.items())
        },
        "latest_day": latest_day,
        "latest_day_spend_usd": round(float(today["spend_usd"]), 4),
        "spend_cap_usd": spend_cap_usd,
        "spend_cap_exceeded": (
            spend_cap_usd is not None and float(today["spend_usd"]) >= spend_cap_usd
        ),
    }


def render_table(report: dict[str, Any]) -> str:
    """Markdown table for the repository; one row per model."""

    header = (
        "| Model | Çağrı | Hata | Hata oranı | Ayrıştırma | Harcama (USD) |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {model} | {stats['calls']} | {stats['failed']} | "
        f"{stats['error_rate']:.1%} | {stats['parse_rate']:.1%} | {stats['spend_usd']:.2f} |"
        for model, stats in report["by_model"].items()
    ]
    totals = report["totals"]
    rows.append(
        f"| **toplam** | **{totals['calls']}** | **{totals['failed']}** | "
        f"**{totals['error_rate']:.1%}** | **{totals['parse_rate']:.1%}** | "
        f"**{totals['spend_usd']:.2f}** |"
    )
    return "\n".join([header, *rows])


def render_daily_line(report: dict[str, Any]) -> str:
    """The one-line summary posted to the group each day."""

    totals = report["totals"]
    cap = report.get("spend_cap_usd")
    cap_note = f"/{cap:.0f}" if cap else ""
    warning = " ⚠ TAVAN AŞILDI" if report.get("spend_cap_exceeded") else ""
    malformed = report.get("malformed_lines", 0)
    malformed_note = f", {malformed} bozuk satır" if malformed else ""
    return (
        f"{report['run_id']} · {totals['calls']} çağrı · "
        f"hata {totals['error_rate']:.1%} · ayrıştırma {totals['parse_rate']:.1%} · "
        f"bugün ${report['latest_day_spend_usd']:.2f}{cap_note}{warning}{malformed_note}"
    )


def report_for_run(settings: CollectionSettings, run_id: str) -> dict[str, Any]:
    return build_report(
        raw_log_path(settings.raw_dir, run_id),
        run_id=run_id,
        spend_cap_usd=settings.daily_spend_cap_usd,
    )
