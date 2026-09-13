import json

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from advisor import advise, candidates, features, measure, model, render, signals
from advisor.state import AdvisorState
from visibility import ethics

# A sector nobody curated: the advisor must work from the run's own registry.
SECTOR = "genel-test"


def _registry(brand: str, *rivals: str, aliases: tuple[str, ...] = ()):
    return candidates.build_registry(brand, aliases, rivals, SECTOR)


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


# --- measurement --------------------------------------------------------------


def test_named_brands_keeps_first_appearance_order():
    registry = _registry("Mullvad", "NordVPN")
    text = "Önce Mullvad, sonra NordVPN, tekrar Mullvad."
    assert measure.named_brands(text, registry) == ["Mullvad", "NordVPN"]


def test_retrieval_presence_reports_best_position_and_count():
    registry = _registry("Windscribe", "NordVPN", "Mullvad")
    results = _results("NordVPN", "Windscribe", "Windscribe")
    present, best, count = measure.retrieval_presence("Windscribe", results, registry)
    assert (present, best, count) == (True, 2, 2)
    assert measure.retrieval_presence("Mullvad", results, registry) == (False, None, 0)


def test_own_domain_counts_as_presence_without_a_name_match():
    registry = _registry("Windscribe")
    results = [
        {
            "title": "Ana sayfa",
            "snippet": "Hızlı.",
            "link": "https://windscribe.com/",
            "position": 1,
        }
    ]
    assert measure.retrieval_presence("Windscribe", results, registry)[0] is True


def test_observe_marks_first_named_brand_only_when_it_leads():
    row = measure.observe(
        brand="Windscribe",
        registry=_registry("Windscribe", "NordVPN"),
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
    registry = _registry("Mullvad", "NordVPN")
    rows = [
        measure.observe(
            brand="Mullvad",
            registry=registry,
            query="q",
            condition=c,
            rep=0,
            answer=answer,
            results=_results("Mullvad"),
        )
        for c, answer in (("search_off", "NordVPN iyidir."), ("search_on", "Mullvad iyidir."))
    ]
    m = measure.rates(rows)
    assert (m["mention_off"], m["mention_on"], m["first_on"]) == (0.0, 1.0, 1.0)
    assert m["retrieval_presence"] == 1.0
    assert m["queries"] == 1


def test_outreach_targets_are_pages_that_name_rivals_but_not_you():
    registry = _registry("Windscribe", "NordVPN")
    search = [{"query": "en iyi vpn", "results": _results("NordVPN", "Windscribe")}]
    targets = measure.outreach_targets(search, "Windscribe", ["NordVPN"], registry)
    assert [t["rivals"] for t in targets] == [["NordVPN"]]


def test_outreach_targets_exclude_rival_vendor_sites():
    """A rival's own homepage names the rival, but nobody can get listed there."""
    registry = _registry("Windscribe", "NordVPN", "Mullvad")
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
    targets = measure.outreach_targets(search, "Windscribe", ["NordVPN", "Mullvad"], registry)
    assert [t["domain"] for t in targets] == ["www.donanimhaber.com"]


def test_vendor_site_matches_subdomains_but_not_lookalikes():
    assert measure.is_vendor_site("https://nordvpn.com/pricing")
    assert measure.is_vendor_site("https://support.nordvpn.com/x")
    assert not measure.is_vendor_site("https://notnordvpn.com.example/")


# --- candidates: any sector, extracted then corrected -------------------------


def test_extraction_keeps_only_names_that_occur_in_the_text():
    corpus = "Garanti BBVA ve Akbank karşılaştırması. Yapı Kredi kampanyası."
    reply = json.dumps({"brands": ["Garanti BBVA", "Akbank", "UydurmaBank", "akbank"]})
    assert candidates.parse_extraction(reply, corpus) == ["Garanti BBVA", "Akbank"]


def test_extraction_accepts_fenced_json_and_rejects_a_bad_schema():
    corpus = "Akbank ve Yapı Kredi"
    fenced = '```json\n{"brands": ["Akbank"]}\n```'
    assert candidates.parse_extraction(fenced, corpus) == ["Akbank"]
    with pytest.raises(ValueError):
        candidates.parse_extraction('{"names": ["Akbank"]}', corpus)


def test_user_corrections_win_over_extraction():
    merged = candidates.apply_corrections(["Akbank", "Yanlış Ad"], add=["QNB"], drop=["yanlış ad"])
    assert merged == ["Akbank", "QNB"]


def test_uncurated_registry_contains_only_the_run_brands_and_aliases():
    registry = candidates.build_registry("Garanti BBVA", ["Garanti"], ["Akbank"], "bankacılık")
    assert registry.brands == ["Akbank", "Garanti BBVA"]
    assert measure.named_brands("Garanti ile Akbank", registry) == ["Garanti BBVA", "Akbank"]


def test_curated_sector_merges_hand_checked_aliases():
    registry = candidates.build_registry("Windscribe", [], ["YeniVPN"], "vpn")
    assert {"Mullvad", "NordVPN", "YeniVPN", "Windscribe"} <= set(registry.brands)


# --- features: the training columns, built from live results ------------------


def test_live_frame_builds_every_invariant_feature():
    registry = _registry("Windscribe", "NordVPN", "Acme")
    pages = [{"query": "q", "results": _results("NordVPN", "Windscribe", "NordVPN")}]
    frame = features.live_frame(
        pages, registry, features.load_rules(), sector=SECTOR, language="tr"
    )
    assert set(model.FEATURES) <= set(frame.columns)
    by_brand = frame.set_index("brand")
    assert by_brand.at["NordVPN", "n_results_mentioning"] == 2
    assert by_brand.at["Acme", "in_search_results"] == 0
    assert by_brand.at["NordVPN", "is_top_retrieved"] == 1.0


def test_a_rivals_own_site_is_not_counted_as_official_for_another_brand():
    registry = _registry("Windscribe", "NordVPN")
    page = {
        "query": "q",
        "results": [
            {
                "title": "NordVPN ve Windscribe",
                "snippet": "Karşılaştırma.",
                "link": "https://nordvpn.com/blog",
                "position": 1,
            }
        ],
    }
    frame = features.live_frame(
        [page], registry, features.load_rules(), sector=SECTOR, language="tr"
    ).set_index("brand")
    assert frame.at["NordVPN", "n_official"] == 1
    assert frame.at["Windscribe", "n_official"] == 0  # vendor_other, as in training


# --- the saved model ------------------------------------------------------------


def _boosters() -> dict:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((200, len(model.FEATURES))), columns=pd.Index(model.FEATURES))
    labels = (x["volume_share"] > 0.5).astype(int)
    fitted = {}
    for target in ("y_mention", "y_top"):
        clf = LGBMClassifier(n_estimators=10, min_child_samples=5, verbosity=-1)
        clf.fit(x, labels)
        fitted[target] = clf.booster_
    return fitted


def test_score_attaches_probabilities_for_both_targets():
    frame = pd.DataFrame(
        np.random.default_rng(1).random((5, len(model.FEATURES))), columns=pd.Index(model.FEATURES)
    )
    scored = model.score(_boosters(), frame)
    for column in ("score_mention", "score_top"):
        assert scored[column].between(0, 1).all()


def test_load_refuses_a_missing_or_tampered_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path)
    for target in ("y_mention", "y_top"):
        (tmp_path / f"invariant_{target}.txt").write_text("değişmiş", encoding="utf-8")
    manifest = {
        "features": model.FEATURES,
        "files": {f"invariant_{t}.txt": "0" * 64 for t in ("y_mention", "y_top")},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        model.load(tmp_path)


# --- signals ----------------------------------------------------------------------


def _scored() -> pd.DataFrame:
    rows = []
    for brand, mention, share, top in (
        ("Biz", 0.2, 0.1, 0.0),
        ("A", 0.6, 0.5, 1.0),
        ("B", 0.5, 0.4, 0.0),
    ):
        rows.append(
            {
                "brand": brand,
                "score_mention": mention,
                "score_top": mention / 2,
                "in_search_results": 1.0,
                "volume_share": share,
                "volume_vs_leader": share * 2,
                "is_top_retrieved": top,
                "position_rank_pct": share,
            }
        )
    return pd.DataFrame(rows)


def test_brand_scores_rank_the_brand_against_all_candidates():
    scores = signals.brand_scores(_scored(), "Biz", ["A", "B"])
    assert scores["rank"] == 3 and scores["candidates"] == 3
    assert scores["compared_with"] == ["A", "B"]
    assert scores["rival_score_mention"] == pytest.approx(0.55)


def test_lagging_signals_flag_what_the_brand_trails_on():
    rows = {row["signal"]: row for row in signals.lagging_signals(_scored(), "Biz", ["A", "B"])}
    assert rows["volume_share"]["lagging"] is True
    assert rows["in_search_results"]["lagging"] is False


# --- diagnosis and advice -----------------------------------------------------------


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


# --- report ------------------------------------------------------------------------


def _state(curated: bool) -> AdvisorState:
    return {
        "brand": "Garanti BBVA",
        "sector": "vpn" if curated else "bankacılık",
        "language": "tr",
        "max_calls": 20,
        "curated": curated,
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
        "scores": {
            "score_mention": 0.4,
            "score_top": 0.1,
            "rival_score_mention": 0.6,
            "rival_score_top": 0.3,
            "rank": 3,
            "candidates": 6,
            "compared_with": ["Akbank"],
        },
        "signals": [
            {
                "signal": "volume_share",
                "label": "Sonuçların ne kadarında anılmak",
                "brand": 0.1,
                "rivals": 0.4,
                "lagging": True,
            }
        ],
        "candidates": ["Akbank", "Garanti BBVA"],
        "diagnosis": "ceiling",
        "rivals": ["Akbank"],
        "recommendations": advise.recommendations(
            "ceiling", {"retrieval_presence": 1.0, "best_position_median": 3}, [], None
        ),
        "notes": ["not"],
    }


def test_report_states_the_diagnosis_the_learned_signal_and_the_limits():
    text = render.report(_state(curated=True))
    assert "Anılıyorsun ama asla ilk değilsin" in text
    assert "| İlk anılan marka olma | %0 |" in text
    assert "Öğrenilmiş sinyal" in text and "**geride**" in text
    assert "Bu rapor ne söylemiyor" in text


def test_report_on_an_unseen_sector_says_the_signal_was_transferred():
    text = render.report(_state(curated=False))
    assert "eğitim verisinde **yok**" in text
    assert "kayıtlı veri setinde yok" in text
    assert "--add-rival" in text


# --- resilience ----------------------------------------------------------------


class _Receipts:
    """Stand-in for the receipt wrapper: one answer is refused, the rest succeed."""

    def __init__(self, failing: str | None = None):
        self.failing = failing

    async def call(self, key, service, payload, validator=None):
        if key == self.failing:
            raise ValueError("gemini yanıtı eksik/geçersiz/kesilmiş")
        return {"text": "Windscribe iyi bir seçenek, NordVPN de var."}


def _interrogate_state() -> AdvisorState:
    return {
        "brand": "Windscribe",
        "sector": SECTOR,
        "language": "tr",
        "max_calls": 10,
        "queries": ["q"],
        "search": [{"query": "q", "results": _results("Windscribe")}],
    }


def test_interrogate_keeps_the_answers_it_got_when_one_is_refused():
    import asyncio

    from advisor import nodes

    runtime = nodes.Runtime(receipts=_Receipts("ask_0_search_on_r0"), reps=1)  # type: ignore[arg-type]
    runtime.registry = _registry("Windscribe", "NordVPN")
    out = asyncio.run(nodes.make_interrogate(runtime, 10)(_interrogate_state()))
    assert len(out["observations"]) == 1
    assert any("alınamadı" in note for note in out["notes"])


def test_interrogate_still_aborts_when_the_budget_runs_out():
    import asyncio

    from advisor import nodes

    runtime = nodes.Runtime(receipts=_Receipts(), reps=1)  # type: ignore[arg-type]
    runtime.registry = _registry("Windscribe", "NordVPN")
    with pytest.raises(nodes.BudgetExceeded):
        asyncio.run(nodes.make_interrogate(runtime, 1)(_interrogate_state()))


# --- queries and honest diagnosis ----------------------------------------------


def test_query_prompt_asks_for_brand_recommendations_in_the_recorded_style(monkeypatch):
    from advisor import nodes

    monkeypatch.setattr(
        nodes, "style_examples", lambda language, limit=3: ["İyi bir VPN önerir misin?"]
    )
    prompt = nodes.query_payload("bankacılık", "tr", 3)["messages"][0]["content"]
    assert "marka tavsiyesi" in prompt
    assert "İyi bir VPN önerir misin?" in prompt
    assert "Türkçe" in prompt and "bankacılık" in prompt


def test_style_examples_are_real_recorded_queries():
    from advisor import nodes

    if not (nodes.CORPUS / "responses_tr.json").exists():
        pytest.skip("donmuş korpus bu ortamda yok")
    recorded = {q for s in candidates.CURATED for q in nodes.recorded_queries(s, "tr", 5)}
    examples = nodes.style_examples("tr")
    assert examples and set(examples) <= recorded


def test_analyse_refuses_to_diagnose_when_no_market_surfaced():
    """The first banking run: generic questions, no brands anywhere -> not 'absent'."""
    import asyncio

    from advisor import nodes

    runtime = nodes.Runtime(receipts=_Receipts())  # type: ignore[arg-type]
    runtime.registry = _registry("Garanti BBVA", "Akbank")
    observation = measure.observe(
        brand="Garanti BBVA",
        registry=runtime.registry,
        query="tasarruf",
        condition="search_on",
        rep=0,
        answer="Bütçenizi yönetmek için harcamalarınızı takip edin.",
        results=[],
    )
    state: AdvisorState = {
        "brand": "Garanti BBVA",
        "sector": SECTOR,
        "language": "tr",
        "max_calls": 10,
        "queries": ["tasarruf"],
        "search": [],
        "observations": [observation],
    }
    out = asyncio.run(nodes.make_analyse(runtime)(state))
    assert out["diagnosis"] == "thin"
    assert any("Teşhis konmadı" in note for note in out["notes"])


# --- domains: whose site is this, in any sector ------------------------------------


def test_host_label_handles_country_second_level_domains():
    from advisor import domains

    assert domains.host_label("https://www.isbank.com.tr/kredi") == "isbank"
    assert domains.host_label("https://support.nordvpn.com/x") == "nordvpn"
    assert domains.host_label("https://www.hangikredi.com/") == "hangikredi"


def test_turkish_brand_names_match_their_ascii_domains():
    from advisor import domains

    assert domains.owned_by("https://www.isbank.com.tr/", ["Akbank", "İş Bankası"]) == "İş Bankası"
    assert (
        domains.owned_by("https://www.ziraatbank.com.tr/", ["Ziraat Bankası"]) == "Ziraat Bankası"
    )
    assert domains.owned_by("https://www.qnb.com.tr/", ["QNB"]) == "QNB"
    assert domains.owned_by("https://www.hangikredi.com/", ["Akbank", "İş Bankası"]) is None


def test_outreach_skips_platforms_and_rival_sites_in_an_uncurated_sector():
    registry = candidates.build_registry("Garanti BBVA", [], ["İş Bankası", "Akbank"], "bankacılık")
    search = [
        {
            "query": "hangi banka",
            "results": [
                {
                    "title": "Akbank uygulaması",
                    "snippet": "Akbank mobil.",
                    "link": "https://play.google.com/store/apps/details?id=akbank",
                    "position": 1,
                },
                {
                    "title": "İş Bankası kredi",
                    "snippet": "İş Bankası faiz oranları.",
                    "link": "https://www.isbank.com.tr/kredi",
                    "position": 2,
                },
                {
                    "title": "Kredi karşılaştırma",
                    "snippet": "Akbank ve İş Bankası oranları.",
                    "link": "https://www.hangikredi.com/kredi",
                    "position": 3,
                },
            ],
        }
    ]
    targets = measure.outreach_targets(search, "Garanti BBVA", ["İş Bankası", "Akbank"], registry)
    assert [t["domain"] for t in targets] == ["www.hangikredi.com"]


def test_extraction_accepts_a_brand_seen_only_as_a_domain():
    corpus = "qnb.com.tr — İhtiyaç Kredisi Hesaplama — Faiz oranları."
    reply = json.dumps({"brands": ["QNB", "UydurmaBank"]})
    assert candidates.parse_extraction(reply, corpus, ["qnb"]) == ["QNB"]


def test_query_prompt_keeps_questions_inside_the_sector(monkeypatch):
    from advisor import nodes

    monkeypatch.setattr(nodes, "style_examples", lambda language, limit=3: ["örnek?"])
    prompt = nodes.query_payload("bankacılık", "tr", 3)["messages"][0]["content"]
    assert "başka bir kategoriye" in prompt


# --- service: the one way both the CLI and the interface run the advisor ----------------


def test_rival_corrections_reuse_the_run_folder_but_aliases_do_not():
    from advisor import service

    base = service.Options(brand="Garanti BBVA", sector="bankacılık")
    corrected = service.Options(brand="Garanti BBVA", sector="bankacılık", add_rivals=("QNB",))
    aliased = service.Options(brand="Garanti BBVA", sector="bankacılık", brand_aliases=("Garanti",))
    assert service.run_id(base) == service.run_id(corrected)
    assert service.run_id(base) != service.run_id(aliased)


def test_estimate_counts_generation_searches_extraction_and_answers():
    from advisor import service

    options = service.Options(brand="X", sector="hiç-kayıtlı-olmayan-sektör", queries=3, reps=2)
    assert service.estimated_calls(options) == 1 + 3 * (1 + 2 * 2) + 1


def test_merge_appends_reducer_keys_and_replaces_the_rest():
    from advisor import service

    state = {"notes": ["a"], "search": [{"q": 1}], "diagnosis": "thin"}
    merged = service.merge(state, {"notes": ["b"], "search": [{"q": 2}], "diagnosis": "absent"})
    assert merged["notes"] == ["a", "b"]
    assert merged["search"] == [{"q": 1}, {"q": 2}]
    assert merged["diagnosis"] == "absent"
    assert state["notes"] == ["a"]  # the caller's state is not mutated


def test_save_writes_the_report_and_the_corrections(tmp_path):
    from advisor import service

    options = service.Options(brand="X", sector="y", drop_rivals=("Yanlış",), output=tmp_path)
    folder = service.save(options, {"report": "# rapor", "brand": "X", "diagnosis": "thin"})
    assert (folder / "report.md").read_text(encoding="utf-8") == "# rapor"
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert record["corrections"] == {"add": [], "drop": ["Yanlış"]}
    assert record["diagnosis"] == "thin"
