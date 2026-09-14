import csv
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
