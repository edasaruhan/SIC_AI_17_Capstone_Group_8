"""Deterministic measurement — the part that must not be a model's opinion.

Every number the advisor later quotes is produced here, by matching the curated brand
registry against text: which brands an answer names, which one it names first, whether
the brand appears in the retrieved results and how high. Asking an LLM "is this brand
visible?" would be circular and unreproducible, and the concept note rejects it
explicitly; this module is why we can refuse to.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from urllib.parse import urlparse

from evidence_eval.evidence import spans
from modeling.brands import load_registry

from .state import Observation

# Sectors the curated registry covers; the advisor refuses others rather than
# inventing a brand list on the fly.
SECTORS = ("vpn", "hosting", "travel", "editors", "cosmetics")
# Read-only: this file is part of the evidence manifest's hash contract.
SOURCE_DOMAINS = Path("configs/modeling/source_domains_v2.json")


@cache
def vendor_hosts() -> frozenset[str]:
    """Every brand's own official domain, from the source taxonomy."""
    if not SOURCE_DOMAINS.exists():
        return frozenset()
    official = json.loads(SOURCE_DOMAINS.read_text(encoding="utf-8")).get("official", {})
    return frozenset(domain.casefold() for domains in official.values() for domain in domains)


def is_vendor_site(link: str) -> bool:
    """A page on some brand's own site. A rival's homepage is not a place to get listed."""
    host = urlparse(link or "").netloc.casefold().removeprefix("www.")
    return any(host == d or host.endswith(f".{d}") for d in vendor_hosts())


def named_brands(text: str, sector: str) -> list[str]:
    """Brands named in an answer, in the order they first appear."""
    seen: dict[str, int] = {}
    for start, _, brand in spans(text, load_registry(sector)):
        seen.setdefault(brand, start)
    return sorted(seen, key=lambda brand: seen[brand])


def retrieval_presence(
    brand: str, results: list[dict], sector: str
) -> tuple[bool, int | None, int]:
    """Does the brand appear in the retrieved results, how high, and in how many?"""
    positions = []
    for index, result in enumerate(results, 1):
        blob = f"{result.get('title', '')}\n{result.get('snippet', '')}"
        if brand in named_brands(blob, sector) or _own_domain(brand, result.get("link", "")):
            positions.append(index)
    return bool(positions), (min(positions) if positions else None), len(positions)


def _own_domain(brand: str, link: str) -> bool:
    """A result on the brand's own site counts as presence even without a name match."""
    host = urlparse(link or "").netloc.casefold()
    stem = "".join(ch for ch in brand.casefold() if ch.isalnum())
    return bool(stem) and stem in host.replace(".", "").replace("-", "")


def observe(
    *,
    brand: str,
    sector: str,
    query: str,
    condition: str,
    rep: int,
    answer: str,
    results: list[dict],
) -> Observation:
    """One measurement. ``first`` is the top-recommendation proxy used throughout."""
    named = named_brands(answer, sector)
    present, best, count = retrieval_presence(brand, results, sector)
    return {
        "query": query,
        "condition": condition,
        "rep": rep,
        "mentioned": brand in named,
        "first": bool(named) and named[0] == brand,
        "named_brands": named,
        "in_search_results": present,
        "best_position": best,
        "n_results_mentioning": count,
    }


def rates(observations: list[Observation]) -> dict:
    """Mention and first-named rates per condition, plus the retrieval picture."""

    def share(rows: list[Observation], key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0

    off = [o for o in observations if o["condition"] == "search_off"]
    on = [o for o in observations if o["condition"] == "search_on"]
    present = [o for o in on if o["in_search_results"]]
    positions = [o["best_position"] for o in present if o["best_position"]]
    return {
        "n_observations": len(observations),
        "mention_off": share(off, "mentioned"),
        "mention_on": share(on, "mentioned"),
        "first_on": share(on, "first"),
        "retrieval_presence": share(on, "in_search_results"),
        "best_position_median": sorted(positions)[len(positions) // 2] if positions else None,
        "queries_with_presence": len({o["query"] for o in present}),
        "queries": len({o["query"] for o in observations}),
    }


def rival_brands(observations: list[Observation], brand: str, limit: int = 5) -> list[str]:
    """Brands the assistant named most often instead of this one."""
    counts: dict[str, int] = {}
    for row in observations:
        for other in row["named_brands"]:
            if other != brand:
                counts[other] = counts.get(other, 0) + 1
    return [b for b, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def outreach_targets(
    search: list[dict], brand: str, rivals: list[str], sector: str, limit: int = 6
) -> list[dict]:
    """Independent pages that already name your rivals and do not name you.

    Vendor sites are excluded: a rival's own homepage names the rival, but no brand can
    ask to be listed there.

    This is the concrete form of the one recommendation the controlled test measured:
    inclusion in an independent comparison that already lists the competition.
    """
    rows = []
    for page in search:
        for result in page.get("results", []):
            blob = f"{result.get('title', '')}\n{result.get('snippet', '')}"
            named = set(named_brands(blob, sector))
            hits = sorted(named & set(rivals))
            link = result.get("link", "")
            if (
                hits
                and brand not in named
                and not _own_domain(brand, link)
                and not is_vendor_site(link)
            ):
                rows.append(
                    {
                        "domain": urlparse(result.get("link", "")).netloc,
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "position": result.get("position"),
                        "rivals": hits,
                        "query": page.get("query", ""),
                    }
                )
    rows.sort(key=lambda row: (-len(row["rivals"]), row["position"] or 99))
    seen: set[str] = set()
    unique = []
    for row in rows:
        if row["domain"] not in seen:
            seen.add(row["domain"])
            unique.append(row)
    return unique[:limit]
