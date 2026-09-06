"""Deterministic response and search-trace features."""

from __future__ import annotations

import re
from typing import Any

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
HEADER = re.compile(r"(?m)^#{1,6}\s+")
TABLE = re.compile(r"(?m)^\s*\|.+\|\s*$")
NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+")
BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
COMPARISON = ("karşılaştır", "versus", " vs ", "hangisi", "en iyi", "compare", "best")
CURRENTNESS = ("güncel", "şu anda", "bugün", "son", "current", "latest", "now")


def deterministic_features(record: dict[str, Any]) -> dict[str, Any]:
    response = str(record.get("final_response") or "")
    calls = record.get("tool_calls") or []
    searches = record.get("search_results") or []
    search_queries = [
        str(item.get("arguments", {}).get("query") or "")
        for item in calls
        if isinstance(item, dict)
    ]
    search_queries = [query for query in search_queries if query]
    lowered_queries = [f" {query.casefold()} " for query in search_queries]
    rounds = {int(item.get("round", 0)) for item in calls if isinstance(item, dict)}
    organic_count = 0
    for search in searches:
        results = search.get("results") if isinstance(search, dict) else None
        if isinstance(results, dict):
            organic_count += len(results.get("organic") or [])
    brand_terms = (
        "vpn",
        "nordvpn",
        "expressvpn",
        "proton",
        "mullvad",
        "surfshark",
        "marka",
        "brand",
        "kozmetik",
    )
    return {
        "response_length_chars": len(response),
        "response_length_words": len(response.split()),
        "has_markdown_headers": bool(HEADER.search(response)),
        "has_markdown_table": len(TABLE.findall(response)) >= 2,
        "has_numbered_list": bool(NUMBERED.search(response)),
        "has_bullet_list": bool(BULLET.search(response)),
        "num_searches": len(calls),
        "search_queries": search_queries,
        "num_search_rounds": len(rounds),
        "num_search_results_returned": organic_count,
        "query_contains_year": any(YEAR.search(query) for query in search_queries),
        "query_contains_comparison_terms": any(
            term in query for query in lowered_queries for term in COMPARISON
        ),
        "query_contains_brand_names": any(
            term in query for query in lowered_queries for term in brand_terms
        ),
        "query_contains_currentness_terms": any(
            term in query for query in lowered_queries for term in CURRENTNESS
        ),
    }
