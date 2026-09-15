import csv
import json
from pathlib import Path

import pytest

from advisor import audit
from description_lab import design, round2
from visibility import ethics

REPORTS = Path("reports/description_lab")


@pytest.mark.parametrize("category", list(design.CATEGORIES.values()), ids=lambda c: c.key)
def test_every_experiment_sentence_is_detected_as_its_own_type(category):
    for key, text in category.variants:
        if not text:
            continue
        found = audit.detect(text)
        assert key in found, (key, text, found)
        if key != "fabricated_claim":
            assert "fabricated_claim" not in found, text


@pytest.mark.parametrize("category", list(design.CATEGORIES.values()), ids=lambda c: c.key)
def test_base_specification_and_filler_carry_no_measured_type(category):
    assert audit.detect(category.spec) == {}
    assert audit.detect(round2.FILLER[category.key]) == {}


def test_fabricated_claim_sentence_counts_only_as_a_risk():
    found = audit.detect(design.SUNSCREEN.text("fabricated_claim"))
    assert list(found) == ["fabricated_claim"]
    assert not audit.fabricated(
        "Bağımsız hız ölçümlerinde ortalama 450 Mbps; %99,9 çalışma süresi."
    )


def test_measured_wins_match_the_committed_reports():
    for index, assistant in enumerate(("gemini", "cerebras")):
        path = REPORTS / assistant / "round2" / "win_shares.csv"
        rows = {r["sentence"]: r for r in csv.DictReader(path.open()) if r["scope"] == "ALL"}
        for key, wins in audit.WINS.items():
            assert int(rows[key]["wins"]) == wins[index], (assistant, key)
            assert int(rows[key]["appearances"]) == audit.APPEARANCES


def test_verdicts_follow_both_assistants():
    assert audit.verdict("technical") == "strong" and audit.verdict("price") == "strong"
    assert audit.verdict("statistics") == "mixed" and audit.verdict("cited_test") == "mixed"
    for key in ("superlative", "emotional", "expert_quote", "certificate", "social_proof"):
        assert audit.verdict(key) == "weak", key
    assert audit.verdict("fabricated_claim") == "risk"


def test_empty_description_is_told_to_add_product_facts():
    result = audit.audit(design.VPN.spec)
    kinds = [item["kind"] for item in result["advice"]]
    assert kinds.count("add") == 2 and "empty" in kinds and result["findings"] == []


def test_risky_and_weak_sentences_get_risk_and_replace_advice():
    text = (
        f"{design.VPN.spec} {design.VPN.text('superlative')} {design.VPN.text('fabricated_claim')}"
    )
    result = audit.audit(text)
    kinds = [item["kind"] for item in result["advice"]]
    assert kinds[0] == "risk" and "replace" in kinds
    assert {f.key for f in result["findings"]} == {"superlative", "fabricated_claim"}


def test_rivals_with_the_same_fact_ask_for_differentiation():
    technical = design.VPN.text("technical")
    result = audit.audit(f"{design.VPN.spec} {technical}", [technical, f"Rakip. {technical}"])
    assert any(item["kind"] == "differentiate" for item in result["advice"])
    finding = next(f for f in result["findings"] if f.key == "technical")
    assert finding.in_rivals == 2


def test_every_piece_of_advice_passes_the_ethics_filter():
    texts = [design.SUNSCREEN.spec, *(t for _, t in design.SUNSCREEN.variants if t)]
    for text in texts:
        for item in audit.audit(text)["advice"]:
            assert ethics.explain(item["suggestion"]) == [], item
            assert ethics.explain(item["basis"]) == [], item


def test_render_and_cli(capsys):
    text = f"{design.SUNSCREEN.spec} {design.SUNSCREEN.text('technical')}"
    rendered = audit.render(audit.audit(text, [design.SUNSCREEN.spec]))
    assert "Ürüne özgü teknik ayrıntı" in rendered and "Rakiplerde" in rendered
    assert "neyi söylemez" in rendered
    assert audit.main(["--text", text]) == 0
    assert "Öneriler" in capsys.readouterr().out


BANK = "Yıllık aidat yok. Alışverişlerde %2 bonus kazanırsın. Başvuru 5 dakikada tamamlanır."


def test_parse_classification_keeps_only_verbatim_sentences_and_known_types():
    reply = json.dumps(
        {
            "texts": {
                "brand": [
                    {"sentence": "Yıllık aidat yok", "type": "price"},
                    {"sentence": "Başvuru 5 dakikada tamamlanır.", "type": "technical"},
                    {"sentence": "Kart en iyisidir.", "type": "superlative"},
                    {"sentence": "Alışverişlerde %2 bonus kazanırsın.", "type": "nonsense"},
                ]
            }
        }
    )
    parsed = audit.parse_classification(f"```json\n{reply}\n```", {"brand": BANK})
    assert parsed["brand"] == {
        "price": ["Yıllık aidat yok"],
        "technical": ["Başvuru 5 dakikada tamamlanır."],
    }


def test_parse_classification_rejects_what_is_not_the_schema():
    with pytest.raises(ValueError):
        audit.parse_classification("aidat yok", {"brand": BANK})
    with pytest.raises(ValueError):
        audit.parse_classification(json.dumps({"brands": []}), {"brand": BANK})


def test_ai_findings_give_the_right_advice_where_rules_do_not():
    by_rules = audit.audit(BANK)
    assert [item["kind"] for item in by_rules["advice"]].count("add") == 2
    found = {"price": ["Yıllık aidat yok."], "technical": ["Başvuru 5 dakikada tamamlanır."]}
    by_ai = audit.audit(BANK, found=found, method="ai")
    assert "add" not in [item["kind"] for item in by_ai["advice"]]
    assert by_ai["method"] == "ai"


def test_the_rule_risk_survives_a_classifier_that_misses_it():
    claim = design.SUNSCREEN.text("fabricated_claim")
    text = f"{BANK} {claim}"
    merged = audit.with_rule_risk({"statistics": [claim], "price": ["Yıllık aidat yok."]}, text)
    assert merged["fabricated_claim"] == [claim] and "statistics" not in merged
    assert merged["price"] == ["Yıllık aidat yok."]


def test_classify_goes_through_the_receipts_and_keys_by_payload():
    import asyncio

    seen = []

    class Receipts:
        async def call(self, key, service, payload, validator=None):
            seen.append((key, service))
            result = {
                "text": json.dumps(
                    {"texts": {"brand": [{"sentence": "Yıllık aidat yok.", "type": "price"}]}}
                )
            }
            assert validator is not None
            validator(result)
            return result

    classified = asyncio.run(audit.classify({"brand": BANK}, Receipts()))
    assert classified == {"brand": {"price": ["Yıllık aidat yok."]}}
    assert seen[0][1] == "gemini" and seen[0][0].startswith("audit_")
    other = audit.classification_payload({"brand": BANK + " Ek cümle."})
    assert audit.classification_payload({"brand": BANK}) != other


def test_record_and_section_name_the_method():
    record = audit.as_record(audit.audit(BANK, found={"price": ["Yıllık aidat yok."]}, method="ai"))
    assert record["findings"][0] == {
        "key": "price",
        "label": "Fiyat avantajı",
        "verdict": "strong",
        "gemini": 21 / 60,
        "cerebras": 30 / 60,
        "sentences": ["Yıllık aidat yok."],
        "in_rivals": None,
    }
    assert "Gemini 3.5 Flash Lite ile sınıflandırıldı" in "\n".join(audit.section(record))
