"""Expand responses into (response, candidate brand) rows.

Neither corpus gave the assistant a brand list -- that is the design rule the
whole project rests on, so there is no candidate set to read off the prompt. The
action space is instead the curated registry for the category, restricted to
brands that appear at least once in that language's corpus. The restriction uses
brand presence only, never the label of the row being scored, and the priors that
could leak are computed per fold in ``features.py``.

Two targets are produced for every pair:

``y_mention``  the brand is named anywhere in the answer -- row-level AI Share of
               Voice, dense enough to train on in every category.
``y_top``      the brand is the answer's single top recommendation -- the task the
               concept note defines, available only for decided responses.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .brands import BrandRegistry, comparison_key, load_registry
from .features import load_lexicon, snippet_features

# Surface forms that are ordinary words in English or Turkish. Matching these
# case-insensitively in snippet text produces false hits ("render the page",
# "going to Rome"), so they are additionally required to appear capitalised.
AMBIGUOUS_FORMS = frozenset(
    {
        "render",
        "cursor",
        "going",
        "note",
        "dove",
        "essence",
        "vim",
        "zed",
        "warp",
        "helix",
        "kate",
        "nova",
        "atom",
        "idle",
        "rider",
        "eclipse",
        "pia",
        "mac",
        "booking",
        "hopper",
        "wix",
        "anua",
        "abib",
        "purito",
        "pastel",
        "kiehls",
    }
)

_I_FORMS = {"I": "i", "İ": "i", "ı": "i", "i": "i"}
_NON_ALNUM = re.compile(r"[^0-9a-zçğöşü]+")

EDITORIAL = frozenset(
    {
        "pcmag.com",
        "cnet.com",
        "wired.com",
        "techradar.com",
        "tomsguide.com",
        "nytimes.com",
        "theverge.com",
        "zdnet.com",
        "forbes.com",
        "engadget.com",
        "webtekno.com",
        "chip.com.tr",
        "hurriyet.com.tr",
        "milliyet.com.tr",
        "shiftdelete.net",
        "technopat.net",
        "wirecutter.com",
    }
)
FORUM = frozenset(
    {
        "reddit.com",
        "quora.com",
        "stackoverflow.com",
        "stackexchange.com",
        "news.ycombinator.com",
        "eksisozluk.com",
        "github.com",
        "medium.com",
        "dev.to",
    }
)
AFFILIATE = frozenset(
    {
        "vpnmentor.com",
        "top10vpn.com",
        "safetydetectives.com",
        "cybernews.com",
        "comparitech.com",
        "vpnranks.com",
        "bestvpn.com",
        "wizcase.com",
        "hostadvice.com",
        "websitetooltester.com",
        "hepsiburada.com",
        "trendyol.com",
        "n11.com",
        "amazon.com",
        "gratis.com",
    }
)


def text_key(text: str) -> str:
    """Normalise free text to space-delimited tokens using the brand i-folding."""
    folded = "".join(_I_FORMS.get(char, char) for char in unicodedata.normalize("NFKC", text))
    return f" {_NON_ALNUM.sub(' ', folded.casefold()).strip()} "


def _match_forms(registry: BrandRegistry, brand: str) -> tuple[list[str], list[str]]:
    """Split a brand's surface forms into unambiguous and ambiguous matchers."""
    forms = [form for form, display in registry.display_by_key.items() if display == brand]
    plain: list[str] = []
    ambiguous: list[str] = []
    for key in forms:
        if key in AMBIGUOUS_FORMS or len(key) < 4:
            ambiguous.append(key)
        else:
            plain.append(key)
    return plain, ambiguous


def _mentions(brand_forms: tuple[list[str], list[str]], raw: str, squashed: str) -> bool:
    plain, ambiguous = brand_forms
    for key in plain:
        if key in squashed:
            return True
    for key in ambiguous:
        pattern = rf"\b{re.escape(key[:1].upper())}{re.escape(key[1:])}\b"
        if re.search(pattern, raw):
            return True
    return False


def build_universe(frame: pd.DataFrame, category: str) -> list[str]:
    """Curated brands for a category that occur at least once in this corpus."""
    registry = load_registry(category)
    subset = frame[frame["category"] == category]
    observed: set[str] = set()
    for brands in subset["brands_mentioned"]:
        observed.update(brands)
    for brand in subset["top_recommendation"]:
        if brand:
            observed.add(brand)
    return sorted(observed & set(registry.brands))


def _source_type(
    link: str,
    brand_keys: Sequence[str],
    editorial: frozenset[str],
    forum: frozenset[str],
    affiliate: frozenset[str],
) -> str:
    host = re.sub(r"^https?://", "", link or "").split("/")[0].casefold()
    host_key = comparison_key(host)
    if any(key and key in host_key for key in brand_keys):
        return "official"
    if any(domain in host for domain in affiliate):
        return "affiliate"
    if any(domain in host for domain in forum):
        return "forum"
    if any(domain in host for domain in editorial):
        return "editorial"
    return "other"


EVIDENCE_COLUMNS = [
    "record_id",
    "language",
    "category",
    "query_id",
    "query_text",
    "model_id",
    "condition",
    "run_index",
    "brand",
    "search_query",
    "search_round",
    "position",
    "link",
    "domain",
    "source_type",
    "title",
    "snippet",
    "brand_in_title",
    "y_mention",
    "y_top",
]


def _domain(link: str) -> str:
    return re.sub(r"^https?://", "", link or "").split("/")[0].casefold()


def build_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (response, brand, retrieved result) with the source kept intact.

    ``build`` reduces the retrieved evidence to per-brand counts, which is what a
    model needs and the wrong shape for anything that has to cite a source. This
    table keeps the search query that surfaced each page, its round, rank, URL,
    domain and snippet text, so a later stage can point at the pages that carried
    a brand into the answer rather than only report that some existed.

    Long format: a brand appearing in six results gets six rows. Join back to the
    feature table on ``(record_id, brand)``.
    """
    languages = set(frame["language"])
    if len(languages) != 1:
        raise ValueError(f"build_evidence() handles one language at a time, got {sorted(languages)}")
    universes = {c: build_universe(frame, c) for c in sorted(frame["category"].unique())}
    registries = {c: load_registry(c) for c in universes}
    forms = {
        (c, brand): _match_forms(registries[c], brand)
        for c, brands in universes.items()
        for brand in brands
    }
    brand_keys = {
        (c, brand): [k for k, d in registries[c].display_by_key.items() if d == brand]
        for c, brands in universes.items()
        for brand in brands
    }

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        category = str(record["category"])
        mentioned = set(record["brands_mentioned"])
        top = record["top_recommendation"]
        for item in record["organic"]:
            title = str(item.get("title") or "")
            snippet = str(item.get("snippet") or "")
            link = str(item.get("link") or "")
            raw = f"{title} {snippet}"
            squashed = text_key(raw).replace(" ", "")
            title_squashed = text_key(title).replace(" ", "")
            for brand in universes[category]:
                key = (category, brand)
                if not _mentions(forms[key], raw, squashed):
                    continue
                rows.append(
                    {
                        "record_id": record["record_id"],
                        "language": record["language"],
                        "category": category,
                        "query_id": record["query_id"],
                        "query_text": record["query_text"],
                        "model_id": record["model_id"],
                        "condition": record["condition"],
                        "run_index": record["run_index"],
                        "brand": brand,
                        "search_query": item.get("search_query"),
                        "search_round": item.get("search_round"),
                        "position": int(item.get("position") or 0),
                        "link": link,
                        "domain": _domain(link),
                        "source_type": _source_type(
                            link, brand_keys[key], EDITORIAL, FORUM, AFFILIATE
                        ),
                        "title": title,
                        "snippet": snippet,
                        "brand_in_title": int(_mentions(forms[key], title, title_squashed)),
                        "y_mention": int(brand in mentioned),
                        "y_top": int(brand == top) if top else 0,
                    }
                )
    # reindex rather than the columns= argument, so an empty result still carries
    # the full schema instead of coming back with no columns at all.
    return pd.DataFrame(rows).reindex(columns=EVIDENCE_COLUMNS)


def brand_snippets(frame: pd.DataFrame, *, mask: bool = False) -> pd.DataFrame:
    """The retrieved text supporting each candidate brand, for the M3 encoder.

    ``mask=True`` replaces every registry surface form -- not only the candidate's
    own -- with a single ``[BRAND]`` token. That is the ablation: the snippets
    keep their evidence, claims and specificity, and lose only brand identity.
    The candidate a row is about survives the masking anyway, because the text is
    already selected as "the results that mention this brand".
    """
    universes = {c: build_universe(frame, c) for c in sorted(frame["category"].unique())}
    registries = {c: load_registry(c) for c in universes}
    forms = {
        (c, brand): _match_forms(registries[c], brand)
        for c, brands in universes.items()
        for brand in brands
    }
    # Longest first, so "Proton VPN" is masked before the shorter "Proton".
    all_forms = {
        c: sorted(
            {
                form
                for brand in registries[c].brands
                for form in registries[c].surface_forms.get(brand, ())
                if len(form) >= 4
            },
            key=len,
            reverse=True,
        )
        for c in universes
    }

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        category = str(record["category"])
        prepared = [
            (raw, text_key(raw))
            for raw in (
                f"{item.get('title') or ''} {item.get('snippet') or ''}"
                for item in record["organic"]
            )
        ]
        for brand in universes[category]:
            key = (category, brand)
            hits = [
                raw
                for raw, normalised in prepared
                if _mentions(forms[key], raw, normalised.replace(" ", ""))
            ]
            text = " ".join(hits)
            if mask:
                for form in all_forms[category]:
                    text = re.sub(re.escape(form), " [BRAND] ", text, flags=re.IGNORECASE)
                text = re.sub(r"\s+", " ", text).strip()
            rows.append(
                {
                    "record_id": record["record_id"],
                    "brand": brand,
                    "query_text": record["query_text"],
                    "snippet_text": text,
                }
            )
    return pd.DataFrame(rows)


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand every response into one row per candidate brand."""
    languages = set(frame["language"])
    if len(languages) != 1:
        raise ValueError(f"build() handles one language at a time, got {sorted(languages)}")
    lexicon = load_lexicon(next(iter(languages)))
    universes = {c: build_universe(frame, c) for c in sorted(frame["category"].unique())}
    registries = {c: load_registry(c) for c in universes}
    forms = {
        (c, brand): _match_forms(registries[c], brand)
        for c, brands in universes.items()
        for brand in brands
    }
    brand_keys = {
        (c, brand): [k for k, d in registries[c].display_by_key.items() if d == brand]
        for c, brands in universes.items()
        for brand in brands
    }

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        category = str(record["category"])
        mentioned = set(record["brands_mentioned"])
        top = record["top_recommendation"]
        prepared = []
        for item in record["organic"]:
            raw = f"{item.get('title') or ''} {item.get('snippet') or ''}"
            prepared.append(
                (
                    raw,
                    text_key(raw),
                    str(item.get("link") or ""),
                    int(item.get("position") or 0),
                )
            )
        for brand in universes[category]:
            key = (category, brand)
            hits = [
                (position, link, raw, normalised)
                for raw, normalised, link, position in prepared
                if _mentions(forms[key], raw, normalised.replace(" ", ""))
            ]
            positions = [position for position, _, _, _ in hits if position > 0]
            types = [
                _source_type(link, brand_keys[key], EDITORIAL, FORUM, AFFILIATE)
                for _, link, _, _ in hits
            ]
            language_features = snippet_features(
                [raw for _, _, raw, _ in hits],
                [normalised for _, _, _, normalised in hits],
                lexicon,
            )
            rows.append(
                {
                    "record_id": record["record_id"],
                    "language": record["language"],
                    "category": category,
                    "query_id": record["query_id"],
                    "model_id": record["model_id"],
                    "condition": record["condition"],
                    "run_index": record["run_index"],
                    "brand": brand,
                    "y_mention": int(brand in mentioned),
                    "y_top": int(brand == top) if top else 0,
                    "response_decided": int(bool(top)),
                    "n_results_mentioning": len(hits),
                    "in_search_results": int(bool(hits)),
                    "best_position": min(positions) if positions else 0,
                    "n_official": types.count("official"),
                    "n_editorial": types.count("editorial"),
                    "n_affiliate": types.count("affiliate"),
                    "n_forum": types.count("forum"),
                    "n_other": types.count("other"),
                    **language_features,
                }
            )
    return pd.DataFrame(rows)
