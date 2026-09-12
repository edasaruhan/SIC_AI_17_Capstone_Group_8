import pandas as pd

from visibility import brand_profile as bp


def _pairs() -> pd.DataFrame:
    """Leader L is always retrieved and named; small brand S rarely."""
    rows = []
    for record in range(10):
        for condition in ("search_off", "search_on"):
            on = condition == "search_on"
            for brand, named, retrieved in (
                ("L", 1, 1),
                ("R", record < 6, 1),
                ("S", record < 2, 0),
            ):
                rows.append(
                    {
                        "record_id": f"{condition}{record}",
                        "category": "vpn",
                        "condition": condition,
                        "brand": brand,
                        "y_mention": int(named),
                        "y_top": int(brand == "L"),
                        "in_search_results": int(on and retrieved),
                        "n_results_mentioning": 3 if on and retrieved else 0,
                        "best_position": 1 if on and retrieved and brand == "L" else 0,
                        **{f"n_{kind}": 0 for kind in bp.SOURCE_KINDS},
                    }
                )
    frame = pd.DataFrame(rows)
    frame.loc[frame["in_search_results"] == 1, "n_editorial"] = 2
    frame.loc[frame["in_search_results"] == 1, "n_forum"] = 1
    return frame


def _evidence() -> pd.DataFrame:
    rows = [
        ("pcmag.com", "editorial", ["L", "R"]),
        ("reddit.com", "forum", ["L", "R", "S"]),
        ("top10vpn.com", "affiliate", ["L"]),
        ("nordvpn.com", "vendor_other", ["L", "R"]),
    ]
    out = []
    for i, (domain, kind, brands) in enumerate(rows):
        for brand in brands:
            out.append(
                {
                    "category": "vpn",
                    "record_id": f"search_on{i}",
                    "source_id": f"s{i}",
                    "domain": domain,
                    "source_type": kind,
                    "brand": brand,
                    "title": f"{domain} page",
                    "link": f"https://{domain}/x",
                }
            )
    return pd.DataFrame(out)


def test_leaderboard_ranks_brands_by_retrieval_on_mention() -> None:
    board = bp.leaderboard(_pairs(), "vpn")
    assert list(board.index) == ["L", "R", "S"]
    assert board.at["S", "mention_on"] == 0.2 and board.at["L", "rank_on"] == 1
    assert bp.winners(board, "S", k=2) == ["L", "R"]


def test_outreach_targets_are_independent_pages_naming_rivals_but_not_the_brand() -> None:
    targets = bp.outreach_targets(_evidence(), "vpn", "S", ["L", "R"])
    # reddit names S already; top10vpn names one rival only; nordvpn is a vendor site.
    assert list(targets["domain"]) == ["pcmag.com"]
    assert targets.iloc[0]["rivals"] == "L, R"


def test_recommendations_follow_gaps_and_carry_the_tested_effect() -> None:
    pairs = _pairs()
    profile = bp.retrieval_profile(pairs, "vpn")
    gap = bp.gaps(profile, "S", ["L", "R"])
    assert gap.set_index("metric").at["presence", "brand"] == 0.0
    effects = pd.DataFrame(
        [
            {
                "scope": "ALL",
                "arm": "neutral_p5",
                "baseline": "control",
                "metric": "mentioned",
                "effect": 0.2,
                "effect_lo": 0.1,
                "effect_hi": 0.3,
                "cells": 30,
            }
        ]
    )
    recs = bp.recommendations(
        gap, bp.outreach_targets(_evidence(), "vpn", "S", ["L", "R"]), effects
    )
    titles = [r["title"] for r in recs]
    assert titles[0].startswith("Bağımsız inceleme")
    assert "+20 puan artırdı" in recs[0]["verdict"]
    assert recs[0]["targets"][0]["domain"] == "pcmag.com"
    # Untested advice says so instead of inventing an effect.
    assert "henüz ölçülmedi" in recs[-1]["verdict"]


def test_leader_gets_no_gap_advice_only_the_language_caution() -> None:
    pairs = _pairs()
    gap = bp.gaps(bp.retrieval_profile(pairs, "vpn"), "L", ["R", "S"])
    recs = bp.recommendations(gap, pd.DataFrame(), None)
    assert [r["title"] for r in recs] == [
        "Üstünlük dilini ('en iyi', '#1') tek başına strateji yapma"
    ]
