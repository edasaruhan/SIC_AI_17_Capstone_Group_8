import pandas as pd
import pytest

from visibility import intervention as iv

RESULTS = [
    {"title": f"t{i}", "snippet": f"s{i}", "link": f"https://site{i}.com/x"} for i in range(10)
]


def test_control_keeps_the_recorded_context_and_renumbers_positions() -> None:
    control = iv.arm_results(RESULTS, None, "vpn", "control")
    assert [r["title"] for r in control] == [r["title"] for r in RESULTS]
    assert [r["position"] for r in control] == list(range(1, 11))
    with pytest.raises(ValueError):
        iv.arm_results(RESULTS, None, "vpn", "neutral_p5")


def test_arms_change_exactly_one_thing_about_the_inserted_page() -> None:
    p5 = iv.arm_results(RESULTS, "IVPN", "vpn", "neutral_p5")
    p1 = iv.arm_results(RESULTS, "IVPN", "vpn", "neutral_p1")
    sup = iv.arm_results(RESULTS, "IVPN", "vpn", "superlative_p5")
    two = iv.arm_results(RESULTS, "IVPN", "vpn", "neutral_p5_p8")
    inserted = {k: v for k, v in p5[4].items() if k != "position"}
    assert "IVPN" in inserted["snippet"] and inserted["link"].startswith("https://independent")
    assert {k: v for k, v in p1[0].items() if k != "position"} == inserted
    assert sup[4]["link"] == inserted["link"] and sup[4]["snippet"] != inserted["snippet"]
    assert "best" in sup[4]["snippet"] and "best" not in inserted["snippet"]
    assert two[4]["snippet"] == two[7]["snippet"] and two[4]["link"] != two[7]["link"]
    assert (len(p5), len(two)) == (11, 12)
    assert [r["position"] for r in two] == list(range(1, 13))
    assert all(".example/" in r["link"] for r in (p5[4], p1[0], sup[4], two[4], two[7]))


def test_plan_shares_the_control_arm_and_has_unique_keys() -> None:
    contexts = [
        {"sector": "vpn", "query_id": f"vpn_0{i}", "query_text": "q", "results": RESULTS}
        for i in range(1, 6)
    ]
    jobs = iv.plan_jobs(contexts, {"vpn": ["IVPN", "Private Internet Access"]}, reps=8)
    assert len(jobs) == 5 * 8 * (1 + 2 * 4)
    assert len({j["key"] for j in jobs}) == len(jobs)
    assert sum(j["arm"] == "control" for j in jobs) == 5 * 8


def test_outcome_detects_mention_and_first_named_registry_brand() -> None:
    text = "I'd pick Mullvad first; IVPN is also solid."
    assert iv.outcome(text, "vpn", "IVPN") == {"mentioned": 1, "first": 0}
    assert iv.outcome(text, "vpn", "Mullvad") == {"mentioned": 1, "first": 1}
    assert iv.outcome("No names here.", "vpn", "IVPN") == {"mentioned": 0, "first": 0}


def _rates(sector: str, query: str, brand: str, relevance: float, presence: float) -> list[dict]:
    base = {"category": sector, "query_id": query, "brand": brand}
    off = [
        {**base, "condition": "search_off", "y_mention": int(i < relevance * 100)}
        for i in range(100)
    ]
    on = [
        {**base, "condition": "search_on", "in_search_results": int(i < presence * 100)}
        for i in range(100)
    ]
    return off + on


def test_targets_are_query_relevant_brands_that_retrieval_rarely_surfaces() -> None:
    rows = []
    for brand, relevance, presence in (
        ("NordVPN", 0.9, 0.9),  # dominant: outside the band
        ("IVPN", 0.3, 0.4),
        ("Private Internet Access", 0.1, 0.3),
        ("Mullvad", 0.2, 0.8),
        ("AirVPN", 0.01, 0.0),  # irrelevant: the assistant never names it
    ):
        rows += _rates("vpn", "vpn_01", brand, relevance, presence)
    # Relevance on a query outside the first five must not count.
    rows += _rates("vpn", "vpn_09", "Windscribe", 0.3, 0.0)
    for query in ("vpn_02", "vpn_03", "vpn_04", "vpn_05"):
        rows += _rates("vpn", query, "NordVPN", 0.9, 0.9)
    for sector in ("hosting", "travel"):
        for brand in ("A", "B"):
            rows += _rates(sector, f"{sector}_01", brand, 0.2, 0.0)
    targets = iv.select_targets(pd.DataFrame(rows))
    assert targets["vpn"] == ["Private Internet Access", "IVPN"]


def test_effects_pair_arms_within_cells() -> None:
    rows = []
    for query in ("q1", "q2", "q3"):
        for rep in range(4):
            rows.append(
                {
                    "sector": "vpn",
                    "query_id": query,
                    "target": "IVPN",
                    "arm": "control",
                    "rep": rep,
                    "mentioned": 0,
                    "first": 0,
                }
            )
            rows.append(
                {
                    "sector": "vpn",
                    "query_id": query,
                    "target": "IVPN",
                    "arm": "neutral_p5",
                    "rep": rep,
                    "mentioned": 1,
                    "first": int(rep < 2),
                }
            )
    result = iv.effects(pd.DataFrame(rows), n_resamples=200)
    presence = result[(result.scope == "vpn") & (result.arm == "neutral_p5")].set_index("metric")
    assert presence.loc["mentioned", "effect"] == 1.0
    assert presence.loc["mentioned", "effect_lo"] == 1.0
    assert presence.loc["first", "effect"] == pytest.approx(0.5)
    assert set(result.scope) == {"ALL", "vpn"}
