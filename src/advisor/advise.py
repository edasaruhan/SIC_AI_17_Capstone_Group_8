"""Diagnosis and advice — the project's findings, turned into a decision.

The controlled test (``reports/intervention/``) measured three things on 1,080 calls:

* appearing in one independent comparison page that already names your rivals moves
  mention from 2.1% to 33% (+31.2 points),
* moving that page from rank 5 to rank 1 adds another +31.2,
* and **nothing** moved the brand into the first-named position -- zero in every arm.

That maps onto exactly three situations a brand can be in, and each deserves a
different answer. Telling a brand in the third situation to "write more content" is
what the tools on the market do; our own numbers say it will not work. Separating the
cases is the product.

Thresholds below are product judgement, not measurements: they decide which advice to
show, not how large an effect is. Effect sizes always come from the measured table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from visibility import ethics
from visibility.brand_profile import tested_effect, verdict

EFFECTS = Path("reports/intervention/effects.csv")

# Product judgement, stated so a reader can disagree with it.
PRESENCE_FLOOR = 0.34  # retrieved in fewer than a third of answers = effectively absent
RANK_FLOOR = 5  # the controlled test contrasted rank 5 against rank 1
MENTION_FLOOR = 0.30  # named often enough that retrieval is clearly working
LEADER_FLOOR = 0.20  # already the first brand named often enough to be the incumbent

DIAGNOSES = {
    "absent": "Aramada görünmüyorsun",
    "low_rank": "Aramada görünüyorsun ama alt sıralarda",
    "ceiling": "Anılıyorsun ama asla ilk değilsin",
    "leader": "Bu sorgularda zaten öndesin",
    "thin": "Ölçüm bu marka için yeterli veri üretmedi",
}


def load_effects(path: Path = EFFECTS) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def diagnose(measures: dict) -> str:
    """Which of the three situations is this brand in?

    Checked in order of what has to be fixed first: you cannot improve your rank in a
    result set you are not in, and you cannot become the first name until you are named.
    """
    if not measures.get("n_observations"):
        return "thin"
    if measures.get("retrieval_presence", 0.0) < PRESENCE_FLOOR:
        return "absent"
    best = measures.get("best_position_median")
    if best is not None and best > RANK_FLOOR:
        return "low_rank"
    if measures.get("first_on", 0.0) >= LEADER_FLOOR:
        return "leader"
    if measures.get("mention_on", 0.0) >= MENTION_FLOOR and measures.get("first_on", 0.0) == 0.0:
        return "ceiling"
    return "low_rank"


def _measured(effects: pd.DataFrame | None, arm: str, baseline: str) -> dict:
    test = tested_effect(effects, arm, baseline)
    return {"test": test, "verdict": verdict(test)}


def recommendations(
    diagnosis: str, measures: dict, targets: list[dict], effects: pd.DataFrame | None
) -> list[dict]:
    """Advice for this diagnosis, each item carrying the effect actually measured."""
    recs: list[dict] = []

    if diagnosis in {"absent", "low_rank", "ceiling", "leader"} and diagnosis != "leader":
        recs.append(
            {
                "title": "Rakiplerini anan bağımsız karşılaştırma sayfalarına gir",
                "why": (
                    f"Arama açık yanıtların %{100 * measures.get('retrieval_presence', 0):.0f}"
                    "'inde arama sonuçlarında görünüyorsun. Kontrollü testte tek bir bağımsız "
                    "karşılaştırma sayfasına girmek anılmayı %2'den %33'e çıkardı."
                ),
                "action": (
                    "Aşağıdaki sayfalar rakiplerini anıyor ama seni anmıyor. Editörüne ürün "
                    "bilgisi, fiyat ve test erişimi sun; listeye dahil edilmeyi talep et."
                ),
                "targets": targets,
                **_measured(effects, "neutral_p5", "control"),
            }
        )

    if diagnosis in {"low_rank", "ceiling", "leader"}:
        best = measures.get("best_position_median")
        recs.append(
            {
                "title": "Girdiğin sayfalarda üst sıraya çık",
                "why": (
                    (
                        f"Göründüğün sonuçların ortanca sırası {best}. Kontrollü testte aynı "
                        "sayfanın 5. sıra yerine 1. sırada olması anılmayı %33'ten %65'e çıkardı; "
                        "etkisi 'hiç görünmemekten görünmeye' geçmekle aynı büyüklükte."
                    )
                    if best
                    else "Göründüğün sayfaların sırası ölçülemedi."
                ),
                "action": (
                    "Zaten yer aldığın karşılaştırma ve inceleme sayfalarının kendi arama "
                    "sıralamasını hedefle; yeni sayfa yazmak yerine mevcut yerleşimini güçlendir."
                ),
                "targets": [],
                **_measured(effects, "neutral_p1", "neutral_p5"),
            }
        )

    if diagnosis in {"ceiling", "leader"}:
        recs.append(
            {
                "title": "Birinci sıraya içerikle ulaşılmıyor — beklentiyi buna göre kur",
                "why": (
                    "Kontrollü testin 1.080 çağrısının hiçbirinde hiçbir müdahale hedef markayı "
                    "ilk anılan marka yapmadı. Birincil öneriyi büyük ölçüde marka kimliği "
                    "belirliyor ve bu kısa vadede içerikle değişmiyor."
                ),
                "action": (
                    "Bütçeyi 'bir numara olmak' hedefine değil, listeye girme ve listede "
                    "yukarı çıkma hedefine ayır; birincilik marka yatırımının işi."
                ),
                "targets": [],
                "test": None,
                "verdict": "Kontrollü testte bütün kollarda sıfır etki ölçüldü.",
            }
        )

    recs.append(
        {
            "title": "Üstünlük dilini tek başına strateji yapma",
            "why": (
                "Kayıtlı yanıtlarda üstünlük dili sektörler arasında tutarsız, çoğunda ters "
                "yönde. Kontrollü testte de ölçülebilir bir kazanç vermedi."
            ),
            "action": (
                "İddiaları doğrulanabilir bilgiyle destekle: fiyat, özellik, bağımsız test "
                "sonucu. 'En iyi' demek yerine neyin ölçüldüğünü göster."
            ),
            "targets": [],
            **_measured(effects, "superlative_p5", "neutral_p5"),
        }
    )

    for rec in recs:
        ethics.screen(f"{rec['title']} {rec['why']} {rec['action']}", where=rec["title"])
    return recs


def limits(sector: str, language: str, curated: bool = True) -> list[str]:
    """What this report may not claim. Printed with every run, not buried."""
    transfer = (
        []
        if curated
        else [
            f"'{sector}' sektörü kayıtlı veri setinde yok. Öğrenilmiş sinyal skoru, beş sektörde "
            "eğitilip leave-one-domain-out testinde görmediği sektörlere taşındığı gösterilen "
            "modelden gelir; bu sektörde ayrıca doğrulanmadı.",
            "Rakip marka listesi bir dil modeline arama sonuçlarından çıkarttırıldı; yalnız "
            "metinde gerçekten geçen adlar tutuldu. Eksik veya fazla ad sonuçları etkiler.",
        ]
    )
    return [
        *transfer,
        "Ölçülen etkiler tek bir asistanda (Gemini 3.5 Flash Lite) ve İngilizce bağlamda, "
        "VPN/hosting/seyahat sektörlerinde ölçüldü. Sizin sektörünüzde ve dilinizde "
        f"({sector}/{language}) etkinin yönü aynı olsa da büyüklüğü ölçülmedi.",
        "Bu koşu canlı arama sonuçlarına dayanır; asistanın gerçekte hangi sayfaları "
        "getirdiği farklı olabilir. Test 'sayfa getirilirse ne olur' sorusunu yanıtlar.",
        "Görünürlük marka adı eşleştirmesiyle ölçüldü; 'ilk anılan marka', birincil "
        "önerinin yaklaşık bir vekilidir.",
        "Az sayıda sorgu ve tekrarla çalışıldı; oranlar yön gösterir, kesin değer değildir.",
    ]
