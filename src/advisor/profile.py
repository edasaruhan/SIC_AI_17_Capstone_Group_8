"""From a website or a brand name to a brand profile and the questions to ask.

The user gives one thing: the brand's website, or just its name. The profile -- brand,
other spellings, sector, language, products and the sentences that describe them -- is
read from the site by a model and then shown to the user to correct. Nothing in it is
taken on trust: every alias must occur in the text, every product sentence must be quoted
verbatim, or it is dropped.

Two kinds of question come out of the profile:

* **Discovery questions** never name the brand or its products. They ask for a
  recommendation in the brand's own product categories, so whether the assistant names
  the brand unprompted is a measurement, not an echo of the question.
* **Brand-and-product questions** name the brand and one of its products and ask for a
  choice against rivals. Whether the assistant then recommends it, or a rival, is the
  second measurement.

Both calls are receipted like every other paid call.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

import httpx

from evidence_eval.io import digest

from . import domains, site

MODEL = "gemini-3.5-flash-lite"
MAX_PRODUCTS = 4
MAX_SENTENCES = 4
LANGUAGE_NAME = {"tr": "Türkçe", "en": "İngilizce"}
PROFILE_PROMPT = (
    "Aşağıda bir markanın web sitesinden ya da arama sonuçlarından alınmış metin var. "
    "Yalnız bu metne dayanarak markayı tanımla.\n"
    "- brand: markanın kısa, yaygın adı.\n"
    "- aliases: markanın kendi adının metinde geçen farklı yazılışları (kısaltma, boşluksuz "
    "ya da eski ad gibi); ürün adları ve ek almış biçimler değil (en fazla 4).\n"
    "- sector: markanın kategorisi, bir kullanıcının asistana soracağı biçimde ve kısa "
    "(ör. 'bankacılık', 'VPN', 'güneş kremi').\n"
    "- language: metnin dili, 'tr' ya da 'en'.\n"
    "- products: en fazla 4 öne çıkan ürün ya da hizmet. Her biri için name (metindeki adı), "
    "category (marka adı olmadan ürün kategorisi, ör. 'kredi kartı') ve sentences (ürünü "
    "anlatan, metinden hiç değiştirmeden kopyalanmış en fazla 4 cümle).\n"
    "Metinde olmayan bilgi ekleme.\n\n"
    "Metin:\n{text}\n\n"
    'Yalnız JSON döndür: {{"brand": "...", "aliases": ["..."], "sector": "...", '
    '"language": "tr", "products": [{{"name": "...", "category": "...", "sentences": ["..."]}}]}}'
)
QUERIES_PROMPT = (
    "'{brand}' markası '{sector}' kategorisinde. Ürünleri: {products}.\n"
    "Bir kullanıcının yapay zekâ asistanına {language_name} soracağı iki tür soru yaz.\n"
    "1) discovery: {n_discovery} soru. Marka adını ve ürün adlarını hiç anma. Her soru "
    "'{sector}' kategorisindeki firmalar ya da markalar arasında seçim yaptırsın; doğal cevabı "
    "birkaç firma ya da marka adı saymak olsun ve soruda 'hangi', 'hangisi', 'en iyi' ya da "
    "'önerir misin' geçsin (ör. 'Konut kredisi için hangi banka daha avantajlı?'). Genel bilgi, "
    "nasıl yapılır, finansman ya da yatırım yöntemi soran ya da başka bir kategoriye kayan "
    "sorular yazma. Her soru markanın farklı bir ürün kategorisine odaklansın.\n"
    "2) named: {n_named} soru. Markanın adını ve bir ürününü an; ürünü rakipleriyle karşılaştıran "
    "ya da alınmaya değer mi diye soran, cevabı bir öneri olan sorular.\n"
    "Biçim olarak şu gerçek sorulara benzesin:\n{examples}\n"
    'Yalnız JSON döndür: {{"discovery": ["..."], "named": ["..."]}}'
)
EXAMPLES = (
    "- Şu anda en iyi VPN hizmeti hangisi?",
    "- Hassas ciltler için güneş kremi önerir misin?",
)


def fold(text: str) -> str:
    """Case- and accent-insensitive form: "Garanti BBVA" and "garantibbva" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text.replace("ı", "i").replace("İ", "i"))
    return "".join(ch for ch in decomposed.casefold() if ch.isalnum())


def _norm(text: str) -> str:
    return " ".join(text.replace("İ", "i").lower().split())


def _json(reply: str) -> dict:
    cleaned = reply.strip().strip("`").strip().removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Model yanıtı JSON değil") from exc
    if not isinstance(value, dict):
        raise ValueError("Model yanıtının şeması geçersiz")
    return value


def profile_payload(text: str) -> dict:
    return {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": PROFILE_PROMPT.format(text=text[: site.MAX_TEXT])}
        ],
    }


def parse_profile(reply: str, text: str) -> dict:
    value = _json(reply)
    brand = str(value.get("brand", "")).strip()
    sector = str(value.get("sector", "")).strip()
    if not brand or not sector:
        raise ValueError("Profilde marka ya da sektör yok")
    source = _norm(text)
    folded = fold(text)

    def spelling(alias: str) -> bool:
        """A different spelling of the name, not the name plus a suffix or a product word."""
        if alias.casefold() == brand.casefold() or "'" in alias or "’" in alias:
            return False
        if alias.casefold().startswith(f"{brand.casefold()} "):
            return False
        return fold(alias) in folded

    aliases = [
        alias
        for alias in dict.fromkeys(str(a).strip() for a in value.get("aliases") or [])
        if alias and spelling(alias)
    ][:4]
    products = []
    for item in value.get("products") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        category = str(item.get("category", "")).strip()
        if not name or not category:
            continue
        sentences = [
            s
            for s in (str(x).strip() for x in item.get("sentences") or [])
            if s and _norm(s) in source
        ][:MAX_SENTENCES]
        products.append({"name": name, "category": category, "sentences": sentences})
    language = str(value.get("language", "tr")).strip().lower()
    return {
        "brand": brand,
        "aliases": aliases,
        "sector": sector,
        "language": language if language in LANGUAGE_NAME else "tr",
        "products": products[:MAX_PRODUCTS],
    }


def queries_payload(profile: dict, n_discovery: int = 3, n_named: int = 2) -> dict:
    products = "; ".join(f"{p['name']} ({p['category']})" for p in profile.get("products", []))
    prompt = QUERIES_PROMPT.format(
        brand=profile["brand"],
        sector=profile["sector"],
        products=products or "belirtilmedi",
        language_name=LANGUAGE_NAME.get(profile.get("language", "tr"), "Türkçe"),
        # Ask for a few spare questions: the choice filter may drop some.
        n_discovery=n_discovery + 3,
        n_named=n_named + 1,
        examples="\n".join(EXAMPLES),
    )
    return {
        "model": MODEL,
        "temperature": 0.3,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }


# A discovery question must ask for a choice, or its answer names no brand to measure.
CHOICE = (
    "hangi",
    "hangisi",
    "öner",
    "oner",
    "en iyi",
    "en uygun",
    "tavsiye",
    "karşılaştır",
    "which",
    "best",
    "recommend",
)


def asks_for_a_choice(question: str) -> bool:
    lowered = question.replace("İ", "i").lower()
    return any(cue in lowered for cue in CHOICE)


def _names(profile: dict) -> list[str]:
    return [profile["brand"], *profile.get("aliases", [])]


def parse_queries(reply: str, profile: dict, n_discovery: int = 3, n_named: int = 2) -> dict:
    """Discovery questions naming the brand, and named ones that do not, are dropped."""
    value = _json(reply)
    brand_names = [fold(name) for name in _names(profile) if fold(name)]
    product_names = [fold(p["name"]) for p in profile.get("products", []) if fold(p["name"])]

    def names_brand(question: str) -> bool:
        return any(name in fold(question) for name in brand_names)

    def names_product(question: str) -> bool:
        return any(name in fold(question) for name in product_names)

    discovery = [
        q
        for q in dict.fromkeys(str(x).strip() for x in value.get("discovery") or [])
        if q and not names_brand(q) and not names_product(q) and asks_for_a_choice(q)
    ][:n_discovery]
    named = [
        q
        for q in dict.fromkeys(str(x).strip() for x in value.get("named") or [])
        if q and (names_brand(q) or names_product(q))
    ][:n_named]
    if not discovery:
        raise ValueError("Markayı anmayan bir keşif sorusu üretilemedi")
    return {"discovery": discovery, "named": named}


def description(profile: dict) -> str:
    """The product sentences quoted from the site: the text the description audit reads."""
    return " ".join(s for p in profile.get("products", []) for s in p.get("sentences", []))


async def source_text(value: str, receipts: Any, http: httpx.AsyncClient) -> dict:
    """The text the profile is read from: the site, or what a search says about the name."""
    url = site.as_url(value)
    if url:
        read = await site.read_site(url, http)
        return {"text": read.text(), "website": read.url, "source": "site"}
    result = await receipts.call(
        f"profile_search_{digest(value)[:10]}",
        "serper",
        {"q": value, "num": 10, "gl": "tr", "hl": "tr"},
    )
    organic = result.get("organic", []) or []
    stem = fold(value)
    own = next(
        (r for r in organic if stem and stem in fold(domains.host_label(r.get("link", "")))), None
    )
    if own:
        try:
            read = await site.read_site(own["link"], http)
        except (ValueError, httpx.HTTPError):
            read = None
        if read is not None and read.text().strip():
            return {"text": read.text(), "website": read.url, "source": "site"}
    text = "\n".join(f"{r.get('title', '')}. {r.get('snippet', '')}" for r in organic)
    return {"text": text, "website": None, "source": "search"}


async def build_profile(value: str, receipts: Any, http: httpx.AsyncClient) -> dict:
    found = await source_text(value, receipts, http)
    text = found["text"]
    if not text.strip():
        raise ValueError("Markayı anlatan bir metin bulunamadı; web sitesini girmeyi deneyin.")
    payload = profile_payload(text)
    result = await receipts.call(
        f"profile_{digest(payload)[:12]}",
        "gemini",
        payload,
        validator=lambda r: parse_profile(r["text"], text),
    )
    return parse_profile(result["text"], text) | {
        "website": found["website"],
        "source": found["source"],
    }


async def plan_queries(
    profile: dict, receipts: Any, *, n_discovery: int = 3, n_named: int = 2
) -> dict:
    payload = queries_payload(profile, n_discovery, n_named)
    result = await receipts.call(
        f"queries_{digest(payload)[:12]}",
        "gemini",
        payload,
        validator=lambda r: parse_queries(r["text"], profile, n_discovery, n_named),
    )
    return parse_queries(result["text"], profile, n_discovery, n_named)
