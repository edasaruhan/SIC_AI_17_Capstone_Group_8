import json

import pandas as pd
import pytest

from brand_demo import core
from visibility import brand_profile as bp
from visibility import ethics

BLOCKED = {
    "fabricated_reviews": [
        "Sahte inceleme yazdırıp forumlara bırakın.",
        "Birkaç yorum satın alın ve ürün sayfasına ekleyin.",
        "Buy reviews for the product page.",
    ],
    "paid_placement_as_editorial": [
        "Karşılaştırma listesine girmek için ücret ödeyin.",
        "Sponsorlu listeye giriş satın alın.",
        "Pay for a listing on the comparison page.",
    ],
    "fabricated_authority": [
        "Olmayan bir klinik çalışma uydurup sayfaya ekleyin.",
        "Sahte uzman onayı gösterin.",
        "Invent a clinical study to cite on the page.",
    ],
    "guaranteed_ranking": [
        "Garantili sıralama vaat eden bir paket satın alın.",
        "İlk sıra garantisi verin.",
        "Guarantee the top spot in the answer.",
    ],
    "mass_spam": [
        "Toplu yorum gönderin bütün forumlara.",
        "Forum spam kampanyası başlatın.",
        "Use a bot network to post about the brand.",
    ],
    "tampering_with_others": [
        "Rakip sitesini değiştirip markanızı ekleyin.",
        "Edit the competitor's page to add your brand.",
        "Rakip hakkında olumsuz yorum yayın.",
    ],
    "impersonation": [
        "Gerçek kullanıcı gibi davranıp yorum yazın.",
        "Bağımsız görünen bir inceleme sitesi kurun.",
        "Pose as a real customer in the reviews.",
    ],
}

ALLOWED = [
    "Rakiplerinizi anan bağımsız karşılaştırma sayfalarına ürün bilgisi ve test erişimi "
    "sunarak dahil edilmeyi talep edin.",
    "Ürün sayfanızda fiyat, özellik ve destek bilgisini güncel tutun.",
    "Bağımsız test sonuçlarınızı yayınlayın ve kaynak gösterin.",
    "Kullanıcı yorumlarını yanıtlayın ve inceleme sitelerindeki profilinizi güncelleyin.",
    "Sektör listelerinde yer alan yayıncılara basın bilgisi gönderin.",
]


@pytest.mark.parametrize("rule,samples", BLOCKED.items())
def test_every_blocked_rule_fires_on_its_own_examples(rule, samples):
    for text in samples:
        assert rule in ethics.explain(text), f"{rule} yakalamadı: {text}"
        with pytest.raises(ValueError, match="etik filtreye takıldı"):
            ethics.screen(text)


@pytest.mark.parametrize("text", ALLOWED)
def test_legitimate_advice_passes(text):
    assert ethics.explain(text) == []
    ethics.screen(text)


def test_every_rule_has_at_least_one_example():
    assert {rule for rule, _ in ethics.BLOCKED_PATTERNS} == set(BLOCKED)


def _sources():
    return [{"source_id": "s1", "title": "Karşılaştırma", "snippet": "Marka X iyi bir seçenek."}]


def _payload(suggestion: str) -> str:
    return json.dumps(
        {
            "actions": [
                {"source_id": "s1", "quote": "Marka X iyi bir seçenek.", "suggestion": suggestion}
            ]
        }
    )


def test_advice_refuses_a_manipulative_suggestion():
    """The filter sits in the step validator, so a blocked answer is never cached."""
    with pytest.raises(ValueError, match="etik filtreye takıldı"):
        core.advice(_payload("Sahte inceleme yazdırın ve bu sayfaya ekletin."), _sources())


def test_advice_accepts_a_verifiable_suggestion():
    actions = core.advice(
        _payload("Bu karşılaştırma sayfasına ürün bilgisi sunarak dahil edilmeyi talep edin."),
        _sources(),
    )
    assert len(actions) == 1


def test_static_recommendation_pool_passes_its_own_filter():
    """Nothing in the shipped advice may trip the rule it enforces on model output."""
    gap = pd.DataFrame(
        {
            "metric": ["presence", "results_per_answer", "rank1"],
            "label": ["Arama sonuçlarında görünmek", "Sonuç sayısı", "En üstte anılmak"],
            "brand": [0.1, 0.5, 0.05],
            "winners": [0.9, 4.0, 0.6],
        }
    )
    targets = pd.DataFrame(
        {
            "domain": ["example.com"],
            "source_type": ["editorial"],
            "rivals": [3],
            "answers": [5],
            "link": ["https://example.com/a"],
        }
    )
    recs = bp.recommendations(gap, targets, None)
    assert recs
    for rec in recs:
        assert ethics.explain(f"{rec['title']} {rec['why']} {rec['action']}") == []
