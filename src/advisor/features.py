"""Live search results in, the model's training columns out.

The invariant model only transfers if the features it sees at run time are built
exactly the way its training features were. So this module reuses the training code
rather than re-implementing it: ``evidence_v2.source_type`` classifies each result,
``snippet_features`` with the language lexicon summarises the text, ``finalise`` and
``add_relative`` turn raw counts into within-answer relative signals.

One retrieval context (one query's result list) plays the role of one recorded
response, and every candidate brand gets a row in it, as in ``pair_table``.
"""

from __future__ import annotations

import pandas as pd

from evidence_eval.evidence import spans
from evidence_eval.io import read_json
from modeling.brands import BrandRegistry
from modeling.features import finalise, load_lexicon, snippet_features
from modeling.pairs import text_key
from visibility import evidence_v2
from visibility import generalization as g

KINDS = ("official", "editorial", "affiliate", "forum", "retailer", "unknown")


def load_rules() -> dict:
    return read_json(evidence_v2.RULES)


def mentions(result: dict, registry: BrandRegistry) -> set[str]:
    return {brand for _, _, brand in spans(_text(result), registry)}


def _text(result: dict) -> str:
    return f"{result.get('title', '')} {result.get('snippet', '')}"


def candidate_rows(
    *,
    record_id: str,
    query: str,
    sector: str,
    language: str,
    results: list[dict],
    registry: BrandRegistry,
    rules: dict,
) -> list[dict]:
    """One row per candidate brand for one retrieval context, as ``pair_table`` builds it."""
    lexicon = load_lexicon(language)
    named = [mentions(result, registry) for result in results]
    rows = []
    for brand in registry.brands:
        hits = [
            (index, result) for index, result in enumerate(results, 1) if brand in named[index - 1]
        ]
        texts = [_text(result) for _, result in hits]
        types = [
            evidence_v2.source_type(result.get("link", ""), brand, rules) for _, result in hits
        ]
        rows.append(
            {
                "record_id": record_id,
                "query_id": query,
                "category": sector,
                "language": language,
                "brand": brand,
                "n_results_mentioning": len(hits),
                "in_search_results": int(bool(hits)),
                "best_position": min((index for index, _ in hits), default=0),
                **{f"n_{kind}": types.count(kind) for kind in KINDS},
                **snippet_features(texts, [text_key(t) for t in texts], lexicon),
            }
        )
    return rows


def live_frame(
    pages: list[dict], registry: BrandRegistry, rules: dict, *, sector: str, language: str
) -> pd.DataFrame:
    """Relative features for every (query, candidate) pair across the run's searches."""
    rows, per_response = [], {}
    for index, page in enumerate(pages):
        record_id = f"live_{index}"
        results = page.get("results", [])
        per_response[record_id] = len(results)
        rows += candidate_rows(
            record_id=record_id,
            query=str(page.get("query", "")),
            sector=sector,
            language=language,
            results=results,
            registry=registry,
            rules=rules,
        )
    if not rows:
        return pd.DataFrame()
    return g.add_relative(finalise(pd.DataFrame(rows)), pd.Series(per_response))
