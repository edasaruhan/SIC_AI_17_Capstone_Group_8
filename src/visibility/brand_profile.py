"""Brand report logic: observed visibility, retrieval gaps, concrete targets, advice.

Pure functions over evidence_v2 tables; the Streamlit page only renders them.
Every number is observational (recorded answers of the frozen corpus) unless it
comes from the controlled test, and each recommendation says which.

A recommendation is made only for a signal the analysis supports: retrieval
presence, volume and rank were stable across sectors, while snippet language was
not. Each one carries the controlled test's measured effect, or says it is untested.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from modeling.features import select

TRACK_SECTORS = {"en": ("editors", "hosting", "travel", "vpn"), "tr": ("cosmetics", "vpn")}
SECTOR_TR = {
    "vpn": "VPN",
    "hosting": "Hosting / bulut",
    "editors": "Kod editörleri",
    "travel": "Seyahat",
    "cosmetics": "Kozmetik",
}
SOURCE_KINDS = ("official", "editorial", "affiliate", "forum", "retailer", "unknown")
KIND_TR = {
    "official": "resmî site",
    "editorial": "editoryal yayın",
    "affiliate": "karşılaştırma / affiliate",
    "forum": "forum / topluluk",
    "retailer": "satıcı",
    "vendor_other": "rakip sitesi",
    "unknown": "diğer",
}
OUTREACH_KINDS = ("editorial", "affiliate", "forum")
WINNERS = 5
GAP_RATIO = 0.75


def leaderboard(pairs: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Mention rate without and with retrieval, top-recommendation share, rank."""
    part = select(pairs, pairs["category"] == sector)
    rates = part.pivot_table(index="brand", columns="condition", values="y_mention", aggfunc="mean")
    on = select(part, part["condition"] == "search_on")
    board = pd.DataFrame(
        {
            "mention_off": rates.get("search_off", 0.0),
            "mention_on": rates.get("search_on", 0.0),
            "top_on": on.groupby("brand")["y_top"].mean(),
        }
    ).fillna(0.0)
    board["rank_on"] = board["mention_on"].rank(ascending=False, method="min").astype(int)
    return board.sort_values(["mention_on", "top_on"], ascending=False)


def winners(board: pd.DataFrame, brand: str, k: int = WINNERS) -> list[str]:
    return [str(b) for b in board.index if b != brand][:k]


def retrieval_profile(pairs: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Per brand, over retrieval-on answers: how often and how well it is retrieved."""
    on = select(pairs, (pairs["category"] == sector) & (pairs["condition"] == "search_on"))
    grouped = on.groupby("brand")
    volume = cast(pd.Series, grouped["n_results_mentioning"].sum()).clip(lower=1)
    profile = pd.DataFrame(
        {
            "presence": grouped["in_search_results"].mean(),
            "results_per_answer": grouped["n_results_mentioning"].mean(),
            "rank1": (on["best_position"] == 1).groupby(on["brand"]).mean(),
        }
    )
    for kind in SOURCE_KINDS:
        profile[f"share_{kind}"] = cast(pd.Series, grouped[f"n_{kind}"].sum()) / volume
    return profile.fillna(0.0)


def gaps(profile: pd.DataFrame, brand: str, rivals: list[str]) -> pd.DataFrame:
    rows = []
    labels = {
        "presence": "Arama sonuçlarında görünme oranı",
        "results_per_answer": "Yanıt başına onu anan sonuç sayısı",
        "rank1": "1. sıradaki sonuçta görünme oranı",
        "share_editorial": "Kaynaklarında editoryal yayın payı",
        "share_affiliate": "Kaynaklarında karşılaştırma/affiliate payı",
        "share_forum": "Kaynaklarında forum/topluluk payı",
    }
    for metric, label in labels.items():
        own = float(profile.at[brand, metric]) if brand in profile.index else 0.0
        peers = profile.reindex(rivals)[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "label": label,
                "brand": own,
                "winners": float(peers.mean()) if len(peers) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def brand_sources(evidence: pd.DataFrame, sector: str, brand: str, limit: int = 12):
    """Domains whose retrieved results name the brand, most answers first."""
    part = select(evidence, (evidence["category"] == sector) & (evidence["brand"] == brand))
    if part.empty:
        return pd.DataFrame(columns=pd.Index(["domain", "source_type", "answers", "title", "link"]))
    table = (
        part.groupby(["domain", "source_type"])
        .agg(answers=("record_id", "nunique"), title=("title", "first"), link=("link", "first"))
        .reset_index()
    )
    return table.sort_values("answers", ascending=False).head(limit)


def outreach_targets(
    evidence: pd.DataFrame, sector: str, brand: str, rivals: list[str], limit: int = 12
) -> pd.DataFrame:
    """Independent pages that name at least two leading rivals but never the brand."""
    part = select(
        evidence,
        (evidence["category"] == sector) & evidence["source_type"].isin(OUTREACH_KINDS),
    )
    own = sorted(set(part.loc[part["brand"] == brand, "domain"]))
    rows = select(part, part["brand"].isin(rivals) & ~part["domain"].isin(own))
    if rows.empty:
        return pd.DataFrame(
            columns=pd.Index(
                ["domain", "source_type", "rivals", "n_rivals", "answers", "title", "link"]
            )
        )
    table = (
        rows.groupby(["domain", "source_type"])
        .agg(
            rivals=("brand", lambda s: ", ".join(sorted(set(s)))),
            n_rivals=("brand", "nunique"),
            answers=("record_id", "nunique"),
            title=("title", "first"),
            link=("link", "first"),
        )
        .reset_index()
    )
    table = select(table, table["n_rivals"] >= 2)
    return table.sort_values(["answers", "n_rivals"], ascending=False).head(limit)


def tested_effect(effects: pd.DataFrame | None, arm: str, baseline: str) -> dict | None:
    if effects is None or effects.empty:
        return None
    rows = select(
        effects,
        (effects["scope"] == "ALL")
        & (effects["arm"] == arm)
        & (effects["baseline"] == baseline)
        & (effects["metric"] == "mentioned"),
    )
    return None if rows.empty else rows.iloc[0].to_dict()


def verdict(test: dict | None) -> str:
    if test is None:
        return "Kontrollü testte henüz ölçülmedi; gözlemsel ilişkiye dayanır."
    effect, low, high = (100 * test[k] for k in ("effect", "effect_lo", "effect_hi"))
    interval = f"(95% GA {low:+.0f} ile {high:+.0f} puan, {int(test['cells'])} sorgu×marka)"
    if low > 0:
        return f"Kontrollü testte anılma olasılığını {effect:+.0f} puan artırdı {interval}."
    if high < 0:
        return f"Kontrollü testte anılma olasılığını {effect:+.0f} puan düşürdü {interval}."
    return f"Kontrollü testte anlamlı etki görülmedi: {effect:+.0f} puan {interval}."


def recommendations(
    gap: pd.DataFrame, targets: pd.DataFrame, effects: pd.DataFrame | None
) -> list[dict]:
    values = gap.set_index("metric")
    lagging = {
        metric
        for metric in ("presence", "results_per_answer", "rank1")
        if values.at[metric, "brand"] < GAP_RATIO * values.at[metric, "winners"]
    }
    recs = []
    if "presence" in lagging:
        recs.append(
            {
                "title": "Bağımsız inceleme ve karşılaştırma sayfalarında yer al",
                "why": (
                    f"Arama açık yanıtların %{100 * values.at['presence', 'brand']:.0f}'inde "
                    "arama sonuçlarında görünüyorsun; en çok anılan rakiplerde bu oran "
                    f"%{100 * values.at['presence', 'winners']:.0f}."
                ),
                "action": (
                    "Aşağıdaki sayfalar rakiplerini anıyor ama seni anmıyor. Editörüne ürün "
                    "bilgisi ve test erişimi sun; karşılaştırma listelerine dahil edilmeyi talep et."
                ),
                "targets": targets.to_dict("records"),
                "test": tested_effect(effects, "neutral_p5", "control"),
            }
        )
    if "results_per_answer" in lagging:
        recs.append(
            {
                "title": "Tek bir sayfaya değil, birkaç bağımsız kaynağa yayıl",
                "why": (
                    f"Göründüğün yanıtlarda seni ortalama "
                    f"{values.at['results_per_answer', 'brand']:.1f} sonuç anıyor; rakiplerde "
                    f"{values.at['results_per_answer', 'winners']:.1f}."
                ),
                "action": (
                    "Aynı soruya yanıt veren farklı alan adlarında (yayın, forum, karşılaştırma) "
                    "anılmayı hedefle; tek bir basın bülteninin kopyaları yerine ayrı içerik."
                ),
                "targets": [],
                "test": tested_effect(effects, "neutral_p5_p8", "neutral_p5"),
            }
        )
    if "rank1" in lagging:
        recs.append(
            {
                "title": "Üst sıralarda çıkan sayfalarda anıl",
                "why": (
                    f"Arama açık yanıtların %{100 * values.at['rank1', 'brand']:.0f}'inde ilk "
                    f"sıradaki sonuçta yer alıyorsun; rakiplerde "
                    f"%{100 * values.at['rank1', 'winners']:.0f}."
                ),
                "action": (
                    "Sorgunun ilk sonuçlarında çıkan sayfaları (genelde büyük yayınlar ve "
                    "'en iyi X' listeleri) önceliklendir; yeni sayfa yazmak yerine oralarda yer al."
                ),
                "targets": [],
                "test": tested_effect(effects, "neutral_p1", "neutral_p5"),
            }
        )
    recs.append(
        {
            "title": "Üstünlük dilini ('en iyi', '#1') tek başına strateji yapma",
            "why": (
                "Kayıtlı yanıtlarda üstünlük dili sektörler arasında tutarsız, çoğunda ters "
                "yönde. Kontrollü testte de karşılaştırma sayfasındaki üstünlük ifadesi "
                "ölçülebilir bir kazanç vermedi ve markayı hiçbir koşulda ilk sıraya taşımadı."
            ),
            "action": (
                "Önce karşılaştırma sayfalarına girmeyi hedefle; iddiaları doğrulanabilir "
                "bilgiyle (fiyat, özellik, bağımsız test sonucu) destekle."
            ),
            "targets": [],
            "test": tested_effect(effects, "superlative_p5", "neutral_p5"),
        }
    )
    for rec in recs:
        rec["verdict"] = verdict(rec["test"])
    return recs
