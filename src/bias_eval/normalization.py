"""Conservative brand normalization without automatic fuzzy merges."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

VPN_ALIASES = {
    "nord vpn": "nordvpn",
    "proton vpn": "protonvpn",
    "express vpn": "expressvpn",
    "surf shark": "surfshark",
    "private internet access pia": "private internet access",
    "pia": "private internet access",
}


def turkish_casefold(value: str) -> str:
    return value.translate(str.maketrans({"I": "ı", "İ": "i"})).casefold()


def normalized_brand(value: str, *, category: str) -> str:
    value = value.replace("™", "").replace("®", "")
    value = unicodedata.normalize("NFKC", value)
    value = turkish_casefold(value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = " ".join(value.split())
    if category == "vpn":
        return VPN_ALIASES.get(value, value)
    return value


def normalize_judgment_brands(judgment: dict[str, Any], category: str) -> dict[str, Any]:
    core = judgment.get("core")
    if not isinstance(core, dict):
        return judgment
    for key in ("first_mentioned_brand", "top_recommendation"):
        if isinstance(core.get(key), str):
            core[key] = normalized_brand(core[key], category=category)
    brands = core.get("all_brands_mentioned")
    if isinstance(brands, list):
        core["all_brands_mentioned"] = [
            normalized_brand(item, category=category) for item in brands if isinstance(item, str)
        ]
    mentions = core.get("brand_mentions")
    if isinstance(mentions, list):
        for mention in mentions:
            if isinstance(mention, dict) and isinstance(mention.get("name"), str):
                mention["name"] = normalized_brand(mention["name"], category=category)
    return judgment


def alias_candidates(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[str, set[str]] = defaultdict(set)
    for item in judgments:
        category = str(item.get("category") or "")
        core = item.get("judgment", {}).get("core", {})
        for brand in core.get("all_brands_mentioned", []):
            if isinstance(brand, str) and brand:
                observed[category].add(brand)
    candidates: list[dict[str, Any]] = []
    for category, names in sorted(observed.items()):
        ordered = sorted(names)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                score = SequenceMatcher(None, left, right).ratio()
                if 0.82 <= score < 1.0:
                    candidates.append(
                        {
                            "category": category,
                            "left": left,
                            "right": right,
                            "similarity": round(score, 3),
                            "action": "manual_review",
                        }
                    )
    return candidates
