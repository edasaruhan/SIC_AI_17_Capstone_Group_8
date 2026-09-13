import json

import numpy as np
import pandas as pd

from description_lab import analysis, design


def _cards(entry: dict) -> list[dict]:
    return json.loads(entry["payload"]["messages"][1]["content"])["urunler"]


def test_plan_size_and_unique_keys():
    jobs = design.plan_jobs(10)
    assert len(jobs) == 2 * (3 + 11) * 10 == 280
    assert len({entry["key"] for entry in jobs}) == len(jobs)


def test_pilot_is_a_subset_of_the_full_plan():
    full = {entry["key"] for entry in design.plan_jobs(10)}
    pilot = design.pilot_jobs()
    assert pilot and {entry["key"] for entry in pilot} <= full
    assert len(pilot) == 2 * (3 + 3)


def test_real_brands_never_carry_a_description_variant():
    for entry in design.plan_jobs(2):
        if entry["identity"] != "fictional":
            assert entry["variant"] == "control"
        if entry["variant"] == "fabricated_claim":
            assert entry["identity"] == "fictional"


def test_only_the_target_card_differs_and_its_position_is_recorded():
    for entry in design.plan_jobs(1):
        cards = _cards(entry)
        assert len(cards) == 5
        assert cards[entry["position"] - 1]["marka"] == entry["target"]
        others = [c for c in cards if c["marka"] != entry["target"]]
        category = design.CATEGORIES[entry["category"]]
        assert all(c["aciklama"] == category.spec for c in others)
        assert all(c["fiyat"] == category.price for c in others)


def test_price_variant_changes_only_the_target_price():
    entry = design.job("sunscreen", "fictional", "price", 0)
    target = _cards(entry)[entry["position"] - 1]
    assert target["fiyat"] == design.SUNSCREEN.discount_price
    assert target["aciklama"] == design.SUNSCREEN.spec


def test_card_order_is_deterministic_and_positions_vary():
    assert design.job("vpn", "incumbent", "control", 3) == design.job(
        "vpn", "incumbent", "control", 3
    )
    positions = {entry["position"] for entry in design.plan_jobs(10)}
    assert positions == {1, 2, 3, 4, 5}


def test_jobs_are_ordered_repetition_first():
    reps = [entry["rep"] for entry in design.plan_jobs(3)]
    assert reps == sorted(reps)


def test_plan_id_follows_the_design_data(monkeypatch):
    before = design.plan_id()
    monkeypatch.setattr(design, "SYSTEM", design.SYSTEM + " Değişti.")
    assert design.plan_id() != before


def test_outcome_reads_first_mention_and_rank():
    brands = ["Lumera", "Avène", "Bioderma", "CeraVe", "Neutrogena"]
    result = analysis.outcome("Size Avène öneririm; Lumera da iyi bir seçenek.", brands, "Lumera")
    assert result == {"first": 0, "mentioned": 1, "rank": 2, "n_named": 2}
    missing = analysis.outcome("CeraVe iyidir.", brands, "Lumera")
    assert missing["mentioned"] == 0 and np.isnan(missing["rank"])


def _table() -> pd.DataFrame:
    rows = []
    for rep in range(10):
        for variant, first in (("control", 0), ("statistics", 1)):
            rows.append(
                {
                    "category": "sunscreen",
                    "identity": "fictional",
                    "variant": variant,
                    "rep": rep,
                    "position": rep % 5 + 1,
                    "first": first,
                    "mentioned": 1,
                }
            )
        rows.append(
            {
                "category": "sunscreen",
                "identity": "incumbent",
                "variant": "control",
                "rep": rep,
                "position": rep % 5 + 1,
                "first": 1,
                "mentioned": 1,
            }
        )
    return pd.DataFrame(rows)


def test_content_effect_is_measured_against_the_fictional_control():
    content = analysis.content_effects(_table())
    row = content[(content.scope == "sunscreen") & (content.arm == "statistics")].iloc[0]
    assert row.first_effect == 1.0
    assert row.first_effect_lo == 1.0 and row.first_effect_hi == 1.0


def test_identity_effect_uses_the_control_description_only():
    identity = analysis.identity_effects(_table())
    row = identity[(identity.scope == "ALL") & (identity.arm == "incumbent")].iloc[0]
    assert row.n == 10 and row.first_effect == 1.0


def test_render_names_the_three_questions_and_the_limits():
    table = _table()
    text = analysis.render(
        table,
        analysis.identity_effects(table),
        analysis.content_effects(table),
        analysis.position_effects(table),
        planned=280,
    )
    assert "Marka adı" in text and "Açıklama içeriği" in text and "Liste sırası" in text
    assert "Tek asistan" in text
