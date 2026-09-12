"""Preserve all retrieved results; name matches are not proof of support."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from modeling.brands import BrandRegistry, comparison_key, load_registry

from .io import digest, read_json, safe_url

SOURCE_RULES = Path("configs/modeling/source_domains.json")


def host(link: str) -> str:
    return (urlsplit(link).hostname or "").lower() if safe_url(link) else ""


def source_type(link: str, brand: str | None, rules: dict) -> str:
    hostname = host(link)

    def matches(domains: list[str]) -> bool:
        return any(hostname == d or hostname.endswith("." + d) for d in domains)

    if brand and matches(rules["official"].get(brand, [])):
        return "official"
    for kind in ("retailer", "forum", "affiliate", "editorial"):
        if matches(rules[kind]):
            return kind
    return "unknown"


def spans(text: str, registry: BrandRegistry) -> list[tuple[int, int, str]]:
    """Match aliases with word boundaries, I/İ/ı folding and punctuation variants.

    Offsets refer to the untouched input; replacing a match never changes claims.
    Ambiguous short aliases require exact case rather than matching common words.
    """
    matches = []
    for brand, forms in registry.surface_forms.items():
        for form in forms:
            tokens = re.findall(r"[^\W_]+", form, re.UNICODE)
            if not tokens:
                continue
            pattern = r"[^\w]*".join(re.escape(t) for t in tokens)
            flags = 0 if len(comparison_key(form)) < 4 else re.IGNORECASE
            for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text, flags):
                matches.append((match.start(), match.end(), brand))
    chosen = []
    end = -1
    for match in sorted(set(matches), key=lambda m: (m[0], -(m[1] - m[0]), m[2])):
        if match[0] >= end:
            chosen.append(match)
            end = match[1]
    return chosen


def mask_text(text: str, target: str, registry: BrandRegistry) -> str:
    other_ids: dict[str, int] = {}
    pieces, offset = [], 0
    for start, end, brand in spans(text, registry):
        pieces.append(text[offset:start])
        if brand == target:
            pieces.append("[TARGET_BRAND]")
        else:
            other_ids.setdefault(brand, len(other_ids) + 1)
            pieces.append(f"[OTHER_BRAND_{other_ids[brand]}]")
        offset = end
    pieces.append(text[offset:])
    return "".join(pieces)


def retrieved_sources(frame: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    active_rules: dict = rules if rules is not None else read_json(SOURCE_RULES)
    rows = []
    for record in frame.to_dict("records"):
        registry = load_registry(record["category"])
        for index, result in enumerate(record["organic"]):
            title, snippet = str(result.get("title") or ""), str(result.get("snippet") or "")
            link = str(result.get("link") or "")
            brands = sorted({b for _, _, b in spans(title + "\n" + snippet, registry)})
            rows.append(
                {
                    "source_id": digest([record["record_id"], index, result])[:24],
                    "record_id": record["record_id"],
                    "language": record["language"],
                    "category": record["category"],
                    "query_id": record["query_id"],
                    "search_query": result.get("search_query"),
                    "search_round": result.get("search_round"),
                    "result_index": index,
                    "position": result.get("position"),
                    "title": title,
                    "snippet": snippet,
                    "link": link,
                    "domain": host(link),
                    "matched_brands": brands,
                    "source_type": source_type(link, None, active_rules),
                    "exposure": "recorded_search_snippet",
                    "support": "not_reviewed",
                }
            )
    columns = [
        "source_id",
        "record_id",
        "language",
        "category",
        "query_id",
        "search_query",
        "search_round",
        "result_index",
        "position",
        "title",
        "snippet",
        "link",
        "domain",
        "matched_brands",
        "source_type",
        "exposure",
        "support",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def associations(sources: pd.DataFrame) -> pd.DataFrame:
    rules = read_json(SOURCE_RULES)
    rows = []
    for row in sources.to_dict("records"):
        for brand in row["matched_brands"]:
            rows.append(
                {**row, "brand": brand, "source_type": source_type(row["link"], brand, rules)}
            )
    return pd.DataFrame(rows).reindex(columns=[*sources.columns, "brand"])
