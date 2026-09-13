"""What the learned model says about this brand, against its rivals in the same results.

Measurement (``measure.py``) says what the assistant *did*. This says what the
transferable signal *predicts* from the retrieval context alone, and which of the
signals that carried across sectors the brand lags on. The comparison is always
within the run: the same queries, the same result lists, the same candidates -- the
unit M2-Invariant was built on.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

# The signals the generalization report found stable across sectors, in plain words.
# Content language is deliberately absent: it did not transfer.
SIGNALS = {
    "in_search_results": "Arama sonuçlarında görünmek",
    "volume_share": "Sonuçların ne kadarında anılmak",
    "volume_vs_leader": "Lidere göre kaç sonuçta anılmak",
    "is_top_retrieved": "En üstte çıkan aday olmak",
    "position_rank_pct": "Rakiplere göre sıra",
}
LAG_RATIO = 0.75


def per_brand(scored: pd.DataFrame) -> pd.DataFrame:
    columns = ["score_mention", "score_top", *SIGNALS]
    return cast(pd.DataFrame, scored.groupby("brand")[columns].mean())


def comparison_set(table: pd.DataFrame, brand: str, rivals: list[str], limit: int = 5) -> list[str]:
    """Rivals the assistant actually named; if none, the model's highest-scoring others."""
    named = [r for r in rivals if r in table.index and r != brand][:limit]
    if named:
        return named
    others = table.drop(index=brand, errors="ignore").sort_values("score_mention", ascending=False)
    return [str(b) for b in others.index[:limit]]


def brand_scores(scored: pd.DataFrame, brand: str, rivals: list[str]) -> dict:
    """Predicted probabilities for the brand, the rivals' mean, and the brand's rank."""
    if scored.empty or brand not in set(scored["brand"]):
        return {}
    table = per_brand(scored)
    compared = comparison_set(table, brand, rivals)
    ranking = table["score_mention"].rank(ascending=False, method="min")
    return {
        "score_mention": float(table.at[brand, "score_mention"]),
        "score_top": float(table.at[brand, "score_top"]),
        "rival_score_mention": (
            float(table.loc[compared, "score_mention"].mean()) if compared else None
        ),
        "rival_score_top": float(table.loc[compared, "score_top"].mean()) if compared else None,
        "rank": int(ranking.at[brand]),
        "candidates": int(len(table)),
        "compared_with": compared,
    }


def lagging_signals(scored: pd.DataFrame, brand: str, rivals: list[str]) -> list[dict]:
    """Stable signals on which the brand falls clearly behind the rivals it is compared with."""
    if scored.empty or brand not in set(scored["brand"]):
        return []
    table = per_brand(scored)
    compared = comparison_set(table, brand, rivals)
    if not compared:
        return []
    rows = []
    for feature, label in SIGNALS.items():
        mine = float(table.at[brand, feature])
        theirs = float(table.loc[compared, feature].mean())
        rows.append(
            {
                "signal": feature,
                "label": label,
                "brand": mine,
                "rivals": theirs,
                "lagging": theirs > 0 and mine < LAG_RATIO * theirs,
            }
        )
    rows.sort(key=lambda row: (not row["lagging"], row["brand"] - row["rivals"]))
    return rows
