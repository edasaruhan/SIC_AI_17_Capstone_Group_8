import pandas as pd
import pytest

from advisor import advise, measure, render
from advisor.state import AdvisorState
from visibility import ethics


def _results(*brands_per_result: str) -> list[dict]:
    return [
        {
            "title": f"{brand} incelemesi",
            "snippet": f"{brand} hakkında.",
            "link": f"https://x{i}.example",
            "position": i,
        }
        for i, brand in enumerate(brands_per_result, 1)
    ]


def test_named_brands_keeps_first_appearance_order():
    text = "Önce Mullvad, sonra NordVPN, tekrar Mullvad."
    assert measure.named_brands(text, "vpn") == ["Mullvad", "NordVPN"]


def test_retrieval_presence_reports_best_position_and_count():
    results = _results("NordVPN", "Windscribe", "Windscribe")
    present, best, count = measure.retrieval_presence("Windscribe", results, "vpn")
    assert (present, best, count) == (True, 2, 2)
    assert measure.retrieval_presence("Mullvad", results, "vpn") == (False, None, 0)


def test_own_domain_counts_as_presence_without_a_name_match():
    results = [
        {
            "title": "Ana sayfa",
            "snippet": "Hızlı.",
            "link": "https://windscribe.com/",
            "position": 1,
        }
    ]
    assert measure.retrieval_presence("Windscribe", results, "vpn")[0] is True


def test_observe_marks_first_named_brand_only_when_it_leads():
    row = measure.observe(
        brand="Windscribe",
        sector="vpn",
        query="q",
        condition="search_on",
        rep=0,
        answer="NordVPN ve Windscribe iyi seçenekler.",
        results=_results("Windscribe"),
    )
    assert row["mentioned"] is True
    assert row["first"] is False
    assert row["in_search_results"] is True


def test_rates_separate_the_two_conditions():
    rows = [
        measure.observe(
            brand="Mullvad",
            sector="vpn",
            query="q",
            condition=c,
            rep=0,
            answer=answer,
            results=_results("Mullvad"),
        )
        for c, answer in (("search_off", "NordVPN iyidir."), ("search_on", "Mullvad iyidir."))
    ]
    m = measure.rates(rows)
    assert m["mention_off"] == 0.0
    assert m["mention_on"] == 1.0
    assert m["first_on"] == 1.0
    assert m["retrieval_presence"] == 1.0
    assert m["queries"] == 1


def test_outreach_targets_are_pages_that_name_rivals_but_not_you():
    search = [{"query": "en iyi vpn", "results": _results("NordVPN", "Windscribe")}]
    targets = measure.outreach_targets(search, "Windscribe", ["NordVPN"], "vpn")
    assert [t["rivals"] for t in targets] == [["NordVPN"]]
    assert all("Windscribe" not in t["title"] for t in targets)


@pytest.mark.parametrize(
    "measures,expected",
    [
        ({}, "thin"),
        ({"n_observations": 8, "retrieval_presence": 0.1}, "absent"),
        ({"n_observations": 8, "retrieval_presence": 1.0, "best_position_median": 8}, "low_rank"),
        (
            {
                "n_observations": 8,
                "retrieval_presence": 1.0,
                "best_position_median": 2,
                "mention_on": 0.8,
                "first_on": 0.0,
            },
            "ceiling",
        ),
        (
            {
                "n_observations": 8,
                "retrieval_presence": 1.0,
                "best_position_median": 1,
                "mention_on": 0.9,
                "first_on": 0.5,
            },
            "leader",
        ),
    ],
)
def test_diagnose_routes_each_situation(measures, expected):
    assert advise.diagnose(measures) == expected


def test_absent_brand_is_not_told_to_improve_a_rank_it_does_not_have():
    recs = advise.recommendations("absent", {"retrieval_presence": 0.0}, [], None)
    titles = [r["title"] for r in recs]
    assert any("karşılaştırma sayfalarına gir" in t for t in titles)
    assert not any("üst sıraya çık" in t for t in titles)


def test_ceiling_brand_is_told_the_truth_about_first_place():
    recs = advise.recommendations(
        "ceiling", {"retrieval_presence": 1.0, "best_position_median": 2}, [], None
    )
    ceiling = [r for r in recs if "Birinci sıraya" in r["title"]]
    assert ceiling and "sıfır etki" in ceiling[0]["verdict"]


def test_recommendations_quote_the_measured_effect_when_it_exists():
    effects = pd.DataFrame(
        [
            {
                "scope": "ALL",
                "arm": "neutral_p5",
                "baseline": "control",
                "metric": "mentioned",
                "effect": 0.312,
                "effect_lo": 0.192,
                "effect_hi": 0.442,
                "cells": 30,
            }
        ]
    )
    recs = advise.recommendations("absent", {"retrieval_presence": 0.0}, [], effects)
    assert "+31 puan" in recs[0]["verdict"]


def test_every_recommendation_passes_the_ethics_filter():
    for diagnosis in advise.DIAGNOSES:
        recs = advise.recommendations(
            diagnosis, {"retrieval_presence": 0.5, "best_position_median": 4}, [], None
        )
        for rec in recs:
            assert ethics.explain(f"{rec['title']} {rec['why']} {rec['action']}") == []


def test_report_states_the_diagnosis_and_the_limits():
    state: AdvisorState = {
        "brand": "Windscribe",
        "sector": "vpn",
        "language": "tr",
        "max_calls": 20,
        "measures": {
            "queries": 3,
            "n_observations": 12,
            "mention_off": 0.0,
            "mention_on": 0.5,
            "first_on": 0.0,
            "retrieval_presence": 1.0,
            "best_position_median": 3,
            "queries_with_presence": 3,
        },
        "diagnosis": "ceiling",
        "rivals": ["NordVPN"],
        "recommendations": advise.recommendations(
            "ceiling", {"retrieval_presence": 1.0, "best_position_median": 3}, [], None
        ),
        "notes": ["3 sorgu donmuş korpustan alındı."],
    }
    text = render.report(state)
    assert "Anılıyorsun ama asla ilk değilsin" in text
    assert "| İlk anılan marka olma | %0 |" in text
    assert "Bu rapor ne söylemiyor" in text
    assert "Gemini 3.5 Flash Lite" in text


def test_outreach_targets_exclude_rival_vendor_sites():
    """A rival's own homepage names the rival, but nobody can get listed there."""
    search = [
        {
            "query": "en iyi vpn",
            "results": [
                {
                    "title": "NordVPN resmi",
                    "snippet": "NordVPN ile güvenli bağlan.",
                    "link": "https://nordvpn.com/tr/",
                    "position": 1,
                },
                {
                    "title": "En iyi VPN'ler",
                    "snippet": "NordVPN ve Mullvad karşılaştırması.",
                    "link": "https://www.donanimhaber.com/vpn",
                    "position": 2,
                },
            ],
        }
    ]
    targets = measure.outreach_targets(search, "Windscribe", ["NordVPN", "Mullvad"], "vpn")
    assert [t["domain"] for t in targets] == ["www.donanimhaber.com"]


def test_vendor_site_matches_subdomains_but_not_lookalikes():
    assert measure.is_vendor_site("https://nordvpn.com/pricing")
    assert measure.is_vendor_site("https://support.nordvpn.com/x")
    assert not measure.is_vendor_site("https://notnordvpn.com.example/")
