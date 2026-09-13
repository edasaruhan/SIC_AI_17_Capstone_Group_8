"""Evidence-linked action hypotheses, including the zero-mention case; no network."""

from __future__ import annotations

import re
from pathlib import Path

from evidence_eval.io import digest, read_json, sha256, write_json, write_text
from modeling.brands import load_registry

from .core import brand_aliases, mentions, observations

METHOD_REFERENCES = [
    {
        "title": "Google: AI features and your website",
        "url": "https://developers.google.com/search/docs/appearance/ai-features",
    },
    {
        "title": "Google: helpful, reliable, people-first content",
        "url": "https://developers.google.com/search/docs/fundamentals/creating-helpful-content",
    },
]
THEMES = {
    "fiyat / ücretsiz plan": r"\b(fiyat\w*|ücretsiz|bütçe\w*|abonelik\w*|price|free|cost)\b",
    "test / denetim kanıtı": r"\b(test\w*|denetim\w*|audit\w*|pwc|klinik\w*|clinical)\b",
    "gizlilik / güvenlik": r"\b(gizli\w*|güvenli\w*|no.logs?|privacy|security|şifre\w*)\b",
    "hız / sunucu kapsamı": r"\b(hız\w*|sunucu\w*|ülke\w*|speed|server\w*|ping)\b",
    "cihaz / platform uyumu": r"\b(cihaz\w*|platform\w*|android|chrome|mobil\w*|ios)\b",
    "içerik / cilt uyumu": r"\b(içerik\w*|hassas\w*|cilt\w*|ingredient\w*|sensitive)\b",
}


def text(source: dict) -> str:
    return str(source.get("title") or "") + "\n" + str(source.get("snippet") or "")


def support(source: dict, offset: int = 0) -> dict:
    value = text(source)
    words = list(re.finditer(r"\S+", value))
    center = next((i for i, word in enumerate(words) if word.end() > offset), 0)
    start = max(0, center - 5)
    end = min(len(words), start + 18)
    quote = value[words[start].start() : words[end - 1].end()] if words else ""
    return {
        "source_id": source["source_id"],
        "quote": quote,
        "url": source["link"],
    }


def build_action_plan(report: dict) -> dict:
    config, sources, answers = report["config"], report["sources"], report["answers"]
    target = config["brand"]
    if len({s["source_id"] for s in sources}) != len(sources):
        raise ValueError("Kaynak kimlikleri benzersiz olmalı")
    aliases = brand_aliases(target, config["domain"])
    neutral = [s for s in sources if s["purpose"] == "neutral_query"]
    branded = [s for s in sources if s["purpose"] == "brand_audit_only_not_visibility"]
    neutral_hits = [s for s in neutral if mentions(text(s), aliases)]
    branded_hits = [s for s in branded if mentions(text(s), aliases)]
    own = [r for r in observations(config, answers) if r["brand"] == target]
    candidate_names = config.get("competitors", [])
    if config["domain"]:
        candidate_names = [*candidate_names, *load_registry(config["domain"]).brands]
    rivals = []
    for candidate in sorted(set(candidate_names)):
        forms = brand_aliases(candidate, config["domain"])
        if any(mentions(form, aliases) or mentions(target, [form]) for form in forms):
            continue
        counts = {
            condition: sum(
                mentions(a["text"], forms) for a in answers if a["condition"] == condition
            )
            for condition in ("search_off", "search_on")
        }
        hits = [s for s in neutral if mentions(text(s), forms)]
        if any(counts.values()):
            rivals.append(
                {
                    "brand": candidate,
                    **counts,
                    "neutral_source_count": len(hits),
                    "source_ids": [s["source_id"] for s in hits],
                }
            )
    rivals.sort(
        key=lambda r: (-r["search_on"], -r["search_off"], -r["neutral_source_count"], r["brand"])
    )
    themes = []
    for name, pattern in THEMES.items():
        for source in neutral:
            match = re.search(pattern, text(source), re.IGNORECASE)
            if match:
                themes.append({"theme": name, "support": support(source, match.start())})
                break
    zero = bool(own) and all(r["mentions"] == 0 for r in own)
    if not neutral:
        hypothesis = "Tarafsız arama bağlamı boş; bulunabilirlik ile model tercihini ayırmak için veri yetersiz."
    elif not neutral_hits and branded_hits:
        hypothesis = "Marka sorgusunda bulunuyor, fakat bu genel sorguların snippet'lerinde yok. Sorgu/kategori uyumu veya bulunabilirlik farkı araştırılmalı; neden henüz kanıtlanmadı."
    elif not neutral_hits:
        hypothesis = "Bu genel sorguların snippet'lerinde marka yok. İndeksleme hatası, ürün uygunsuzluğu veya internet genelinde yokluk sonucu çıkarılamaz."
    elif zero:
        hypothesis = "Marka snippet'lerde görüldüğü halde yanıta taşınmamış. İçerik kanıtı ve sorguya uygunluk incelenmeli; tercih nedeni henüz bilinmiyor."
    else:
        hypothesis = "Markanın bu örneklemde anılması var. Sorgu bazında kapsam ve tekrar edilebilirlik geliştirilebilir; anılma olumlu öneri anlamına gelmez."
    facts = {
        "answer_mentions": own,
        "neutral_source_mentions": len(neutral_hits),
        "neutral_sources": len(neutral),
        "branded_source_mentions": len(branded_hits),
        "branded_sources": len(branded),
        "zero_mentions": zero,
    }
    tasks = []

    def add(
        title: str,
        basis: str,
        observation: str,
        evidence: list[dict],
        owner: str,
        steps: list[str],
        deliverable: str,
        acceptance: str,
    ) -> None:
        tasks.append(
            {
                "priority": len(tasks) + 1,
                "title": title,
                "basis": basis,
                "observation": observation,
                "evidence": evidence,
                "owner": owner,
                "steps": steps,
                "deliverable": deliverable,
                "acceptance": acceptance,
                "status": "hypothesis_requires_review",
                "expected_direction": "Sorguyla ilgili, doğrulanabilir ürün bilgisinin bulunmasını/yorumlanmasını kolaylaştırma hipotezi; etki ölçülmedi.",
            }
        )

    add(
        "Ürünün hangi sorgulara gerçekten uygun olduğunu netleştir",
        "measured_observation",
        hypothesis,
        [support(s) for s in branded_hits[:1]],
        "Ürün uzmanı + yetkili içerik ekibi",
        [
            f"{target} için tam ürün adını, resmi URL'yi ve hedef kullanıcıyı doğrulayın; aynı adlı ürünleri ayırın.",
            *[
                f"“{q}” için ürün gerçekten uygun mu? Uygun / uygun değil / bilinmiyor işaretleyin ve dayanağını yazın."
                for q in config["queries"]
            ],
            "Uygun olmayan sorguda sırf anılmak için özellik uydurmayın. Farklı bir kullanım senaryosu gerekiyorsa ayrı bir deney açın; eski skorlarla aynı deneymiş gibi birleştirmeyin.",
        ],
        "3 satırlı sorgu → ürün uygunluğu → gerekçe → doğrulanmış ürün/kanıt URL'si matrisi.",
        "Her sorgunun uygunluğu ürün sahibi tarafından kontrol edilmiş; bilinmeyenler açık bırakılmış olmalı.",
    )
    if neutral:
        selected_themes = themes[:4]
        theme_names = (
            ", ".join(t["theme"] for t in selected_themes)
            or "sorguya doğrudan yanıt ve ürünün gerçek kullanım sınırları"
        )
        add(
            "Uygun sorgular için doğrulanabilir ürün bilgi sayfası hazırla / güncelle",
            "source_pattern_and_hypothesis",
            f"Kayıtlı kaynaklarda görülen konu işaretleri: {theme_names}. Bu, markada bu bilgilerin eksik olduğunu veya rakiplerin bu nedenle seçildiğini kanıtlamaz.",
            [t["support"] for t in selected_themes[:2]] or [support(neutral[0])],
            "Yetkili içerik ekibi + ürün uzmanı",
            [
                f"{target} ürününün mevcut sayfasında bu konuların yanıtını arayın; zaten yeterliyse sırf SEO için tekrar sayfa üretmeyin.",
                f"Doğrulanmış bilgi varsa {theme_names} başlıklarında kısa, Türkçe ve doğrudan yanıtlar ekleyin; olmayan özellikleri ve kullanım sınırlarını da açıklayın.",
                "Her test/fiyat/özellik iddiasına geçerli tarih, kapsam ve birincil kanıt bağlantısı koyun; test yoksa test sonucu yazmayın.",
            ],
            f"{target} için 1 ürün/kullanım senaryosu sayfası taslağı + 3 sorguya yanıt veren SSS + iddia/kanıt tablosu.",
            "Taslak ürün uzmanı tarafından doğrulanmış, kaynakları erişilebilir ve üç sorgunun yalnız uygun olanlarına açık yanıt veriyor olmalı.",
        )
    else:
        add(
            "Önce kanıt envanterini tamamla",
            "measurement_gap",
            "Tarafsız sorgular için kullanılabilir arama kaynağı yok; rakip karşılaştırması veya site eksiği uydurulmadı.",
            [],
            "Araştırmacı + ürün uzmanı",
            [
                "Resmi ürün URL'sini, ürün adını ve gerçek hedef kullanımını doğrulayın.",
                "Arama isteğinin doğru sektör/bölge ile sonuç verdiğini kontrol edin; ücretli yeniden toplama için ayrıca onay alın.",
            ],
            "Doğrulanmış ürün kaynağı listesi ve arama kapsamı kontrol kaydı.",
            "En az bir gerçek ürün kaynağı ve doğru deney kapsamı doğrulanmış olmalı.",
        )
    if not neutral_hits:
        add(
            "Bulunabilirliği yetkili site erişimiyle kontrol et",
            "verification_task_not_observed_defect",
            f"Marka, kayıtlı {len(neutral)} tarafsız snippet'in {len(neutral_hits)} tanesinde eşleşti. Bu tek başına sitenin indekslenmediğini göstermez.",
            [],
            "Site sahibi / yetkili teknik ekip",
            [
                "Resmi URL kesinleştikten sonra Search Console URL Denetimi ile indeks ve tarama durumunu kontrol edin.",
                "robots/noindex engeli, önemli bilginin metin olarak bulunması ve sayfaya iç bağlantı olup olmadığını doğrulayın; yalnız doğrulanmış sorun varsa düzeltin.",
                "Özel bir AI dosyası veya şema eklemeyi görünürlük garantisi gibi sunmayın.",
            ],
            "URL bazlı indeks/erişim kontrol listesi; varsa doğrulanmış sorun ve düzeltme kaydı.",
            "Kontrol sonucu kayda alınmış olmalı; sadece snippet'te anılmamak teknik hata kanıtı sayılmamalı.",
        )
    cited_rivals = [r for r in rivals[:3] if r["source_ids"]]
    if cited_rivals:
        source_ids = list(dict.fromkeys(sid for r in cited_rivals for sid in r["source_ids"]))[:2]
        selected_sources = [next(s for s in neutral if s["source_id"] == sid) for sid in source_ids]
        rival_evidence = []
        forms = [form for r in cited_rivals for form in brand_aliases(r["brand"], config["domain"])]
        for source in selected_sources:
            match = re.search(
                "|".join(re.escape(form) for form in forms), text(source), re.IGNORECASE
            )
            rival_evidence.append(support(source, match.start() if match else 0))
        add(
            "Rakiplerin göründüğü kaynaklarda ürünün doğru temsilini araştır",
            "cooccurrence_not_causality",
            "Yanıtlarda anılan ve snippet'lerde eşleşen seçenekler: "
            + ", ".join(r["brand"] for r in cited_rivals)
            + ". Eşleşme, modelin bu kaynağa dayandığını kanıtlamaz.",
            rival_evidence,
            "İçerik araştırmacısı / yetkili iletişim ekibi",
            [
                "Listelenen sayfaların yayıncısını, tarihini, ürün kapsamını ve ticari ilişkilerini doğrulayın.",
                "Ürün gerçekten aynı ihtiyaca uygunsa ve kaynakta bir maddi eksik/yanlış varsa, yayıncıya kanıt bağlantılı doğru ürün bilgi föyü hazırlayın; yayıncının bağımsız kararını koruyun.",
                "Sahte inceleme, ücretli olumlu yorum, toplu spam veya garanti edilmiş listeye giriş önermeyin.",
            ],
            "En çok iki ilgili yayıncı için uygunluk notu ve doğrulanmış ürün/kanıt föyü; otomatik mesaj gönderilmez.",
            "Kaynağın kapsamına uygunluk doğrulanmış ve her ürün iddiası kanıtlı olmalı; kapsam dışıysa iletişim yapılmamalı.",
        )
    return {
        "version": 2,
        "producer": "deterministic_evidence_rules_not_minimax",
        "facts": facts,
        "hypothesis": hypothesis,
        "observed_competitors": rivals[:5],
        "competitor_scope": "supplied_names_and_frozen_registry_only",
        "source_themes": themes,
        "tasks": tasks,
        "measurement_plan": {
            "queries": config["queries"],
            "model": config["model"],
            "locale": config["locale"],
            "baseline": own,
            "steps": [
                "Sadece doğrulanan ve ürünle uyumlu değişiklikleri uygulayın; değişiklik tarihini ve URL'lerini kaydedin.",
                "Önce/sonra ölçümde aynı tarafsız sorguları, modeli, arama bölgesini ve tekrar sayısını kullanın. Marka adını tarafsız sorgulara eklemeyin.",
                "Tarafsız kaynaklarda anılma ile asistan yanıtında anılmayı ayrı ölçün; aynı anda rakipleri de izleyin.",
                "Yeni --run-id gerçek API çağrısı ve maliyet oluşturur; otomatik yeniden toplama yapılmaz. Tek üç-sorguluk sonuç artışın kanıtı değildir.",
            ],
            "metrics": [
                "neutral_source_mentions / neutral_sources",
                "search_off mentions / responses",
                "search_on mentions / responses",
            ],
        },
        "method_references": METHOD_REFERENCES,
        "method_scope": "Google kaynakları Google araması için genel rehberdir; MiniMax tercihini veya artışı garanti etmez.",
    }


def rebuild(folder: Path) -> tuple[dict, Path]:
    """Enrich a verified completed report without modifying it or any paid receipts."""
    original = read_json(folder / "report.json")
    state = read_json(folder / "state.json")
    if state.get("status") != "completed" or digest(original) != state.get("report_hash"):
        raise ValueError("Kaynak rapor tamamlanmamış veya checksum değişmiş")
    original["action_plan"] = build_action_plan(original)
    original["enrichment_provenance"] = {
        "source_report_sha256": sha256(folder / "report.json"),
        "implementation_sha256": sha256(Path(__file__)),
        "api_calls": 0,
        "kind": "offline_action_plan_v2",
    }
    from .core import render

    destination = folder / "report.action-plan.v2.md"
    write_json(folder / "report.action-plan.v2.json", original)
    write_text(destination, render(original))
    return original, destination
