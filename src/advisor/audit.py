"""Description audit: what the controlled experiment says about a product text.

The description experiment (``src/description_lab``, ``reports/description_lab``) showed
two assistants -- Gemini 3.5 Flash Lite and gpt-oss-120b on Cerebras -- five product
cards with identical specifications and changed one sentence. What it found, in the order
it matters to a brand:

* A sentence that carries no information does nothing (Gemini 15%, Cerebras 0%, against
  10% and 5% with no sentence at all). What the text says counts, not its length.
* When every card carries a sentence, product facts win on both assistants: a technical
  detail specific to the product and a price advantage.
* A measured statistic and an independent test report win on one assistant only.
* Superlatives, authority wording, expert quotes, certificates, ratings and emotional
  language stay near or below chance on both.
* A fabricated institutional or clinical claim wins too, and the assistants repeat it to
  the user in 69-79% of the answers that show it, almost never with a warning. The audit
  flags it as a risk and never suggests it; ``visibility.ethics`` blocks advice to invent
  one.

This module is rule-based and free: no call, no model. It finds which of those sentence
types a description contains with Turkish and English cues, attaches the measured win
share, and writes advice that passes the ethics filter. The cues were checked against
every sentence the experiment used. Detection is a heuristic: it tells a copywriter
where to look, it does not decide whether a claim is true.

Rules only know the sentences of the two tested categories. In any other sector the
sentence types come from Gemini instead (``classify``): one call, every decision tied to
a sentence quoted verbatim from the text, and the rule for fabricated claims still
applied on top, so a risk the model misses is not lost.

    PYTHONPATH=src python -m advisor.audit --file aciklama.txt --rival-file rakip.txt
    PYTHONPATH=src python -m advisor.audit --file aciklama.txt --ai --yes   # 1 çağrı
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_eval.io import digest
from visibility import ethics

ASSISTANTS = ("Gemini 3.5 Flash Lite", "gpt-oss-120b (Cerebras)")
APPEARANCES = 60
CHANCE = 0.20
STRONG = 0.30
# Round-two tournament, both categories together: wins out of 60 appearances for
# (Gemini, Cerebras). Source: reports/description_lab/<assistant>/round2/win_shares.csv,
# scope ALL; a test keeps these in step with the reports.
WINS: dict[str, tuple[int, int]] = {
    "technical": (21, 40),
    "price": (21, 30),
    "statistics": (22, 13),
    "cited_test": (19, 2),
    "social_proof": (14, 2),
    "certificate": (4, 14),
    "authority": (8, 16),
    "expert_quote": (6, 9),
    "superlative": (12, 8),
    "emotional": (2, 2),
    "fabricated_claim": (27, 18),
}
LABELS = {
    "technical": "Ürüne özgü teknik ayrıntı",
    "price": "Fiyat avantajı",
    "statistics": "Sayısal ölçüm / istatistik",
    "cited_test": "Bağımsız test veya denetim raporu",
    "social_proof": "Puan ve kullanıcı sayısı",
    "certificate": "Sertifika",
    "authority": "Otorite ifadesi",
    "expert_quote": "Uzman alıntısı",
    "superlative": "Üstünlük ifadesi ('en iyi', 'bir numara')",
    "emotional": "Duygusal dil",
    "fabricated_claim": "Kaynağı gösterilmeyen kurum veya klinik iddiası",
}
VERDICT_TR = {
    "strong": "İki asistanda da kazandırıyor",
    "mixed": "Asistana bağlı",
    "weak": "Belirgin kazanç yok",
    "risk": "Risk: kaldırın ya da kaynağını verin",
}

# Cues run on lower-cased text with the Turkish dotted capital folded ("İ" -> "i").
CUES: dict[str, re.Pattern[str]] = {
    "technical": re.compile(
        r"aes-?\d+|wireguard|openvpn|ikev2|ram tabanlı|şifreleme|protokol|encryption|protocol"
        r"|niasinamid|niacinamide|çinko oksit|zinc oxide|titanyum dioksit|hyaluron|seramid"
        r"|ceramide|retinol|pantenol|panthenol|\bph\s?\d|filtre kombinasyon|mineral filtre"
        r"|işlemci|processor|chipset|batarya|battery|\bmah\b|amoled|oled"
    ),
    "price": re.compile(
        r"indirim|kampanya|daha uygun fiyat|uygun fiyatlı|fiyat avantaj|ücretsiz deneme"
        r"|\bdiscount|cheaper|free trial|\d+\s?% off"
    ),
    "statistics": re.compile(
        r"%\s?\d+|\b\d+([.,]\d+)?\s?%|ortalama \d+|ölçüldü|ölçümlerinde|katılımcı"
        r"|çalışma süresi|uptime|measured|on average"
    ),
    "cited_test": re.compile(
        r"bağımsız (bir )?(test|denetim|laboratuvar|inceleme)|denetim raporu|test raporu"
        r"|laboratuvar raporu|independent (test|audit|lab)|third[- ]party (test|audit)"
        r"|audit report"
    ),
    "social_proof": re.compile(
        r"ortalama \d[.,]\d|\d[.,]\d\s?(puan|yıldız|/\s?5)"
        r"|\d[\d.,]*\s?(bin|milyon)?\S*\s+(den |dan )?(fazla )?(kullanıcı|müşteri|değerlendirme|yorum|indirme)"
        r"|\b\d[\d.,]*\s?(k|m)?\+?\s(users|customers|reviews|downloads)\b|\d[.,]\d\s?(stars|rating)"
    ),
    "certificate": re.compile(
        r"sertifika|\biso\s?\d{4,5}|onaylı|onaylanmış|dermatolojik olarak test|vegan"
        r"|cruelty[- ]free|certified|certification|\btüv\b|helal belge"
    ),
    "authority": re.compile(
        r"(uzman|dermatolog|doktor|hekim|mühendis|bilim insan)\w*\s+(tarafından\s+)?(geliştiril|tasarlan)"
        r"|askeri düzey|military[- ]grade|klinik standart|developed by (experts|dermatologists|doctors)"
    ),
    "expert_quote": re.compile(
        r"(uzman|dermatolog|doktor|hekim|eczacı)\w*\s+(görüşü|yorumu|önerisi)"
        r"|(expert|dermatologist|doctor|pharmacist)\s+(says|recommends|quote)"
    ),
    "superlative": re.compile(
        r"\ben (iyi|iyisi|güçlü|hızlı|güvenli|kaliteli)\b|bir numara|1 numara|#\s?1\b"
        r"|piyasadaki en|rakipsiz|eşsiz|\bthe best\b|number one|best[- ]in[- ]class"
    ),
    "emotional": re.compile(
        r"hisset|huzur|özgürce|özgürlük|\bsev(en|er|diğiniz|gi)\b|keyif|mutlu"
        r"|peace of mind|\bfeel\b|\blove\b|freedom"
    ),
}
INSTITUTION = re.compile(
    r"\b(harvard|mit|stanford|oxford|cambridge|yale|johns hopkins|mayo clinic|nasa|fda|dsö)\b"
    r"|dünya sağlık örgütü|world health organization"
)
PROOF = re.compile(
    r"kanıtla|ispatla|klinik (olarak|çalışma)|çalışmasında|araştırmasında|denetiminde|denetimde"
    r"|clinically|proven|\bstudy\b"
)
ABSOLUTE = re.compile(
    r"%\s?(9[5-9]|100)\b|\b(9[5-9]|100)\s?%|tamamen (engeller|korur|önler)|kanser"
    r"|hastalı\w* (önler|iyileştir)|\bcures?\b"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _fold(text: str) -> str:
    return text.replace("İ", "i").lower()


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(text or "") if part.strip()]


def fabricated(sentence: str) -> bool:
    """A named institution or clinical proof tied to an absolute outcome, without a source."""
    folded = _fold(sentence)
    institution, proof, absolute = (
        bool(INSTITUTION.search(folded)),
        bool(PROOF.search(folded)),
        bool(ABSOLUTE.search(folded)),
    )
    return (institution and (proof or absolute)) or (proof and absolute)


def detect(text: str) -> dict[str, list[str]]:
    """Sentence types found in a text, each with the sentences that carry it.

    A sentence flagged as a fabricated claim counts only as that: its percentage must
    not earn the description credit for statistics.
    """
    found: dict[str, list[str]] = {}
    for sentence in _sentences(text):
        if fabricated(sentence):
            found.setdefault("fabricated_claim", []).append(sentence)
            continue
        folded = _fold(sentence)
        for key, pattern in CUES.items():
            if pattern.search(folded):
                found.setdefault(key, []).append(sentence)
    return found


def shares(key: str) -> tuple[float, float]:
    gemini, cerebras = WINS[key]
    return gemini / APPEARANCES, cerebras / APPEARANCES


def verdict(key: str) -> str:
    if key == "fabricated_claim":
        return "risk"
    measured = shares(key)
    if all(share >= STRONG for share in measured):
        return "strong"
    if any(share >= STRONG for share in measured):
        return "mixed"
    return "weak"


def _pct(value: float) -> str:
    return f"%{100 * value:.0f}"


def _both(key: str) -> str:
    gemini, cerebras = shares(key)
    return f"Gemini {_pct(gemini)}, Cerebras {_pct(cerebras)} (şans {_pct(CHANCE)})"


@dataclass(frozen=True)
class Finding:
    key: str
    sentences: tuple[str, ...]
    in_rivals: int | None  # rival texts that carry the same type; None without rivals


def _advice(found: dict[str, list[str]], rival_found: list[dict[str, list[str]]]) -> list[dict]:
    advice: list[dict] = []
    if "fabricated_claim" in found:
        advice.append(
            {
                "kind": "risk",
                "suggestion": "Yayımlanmış bir kaynağa bağlanamayan kurum veya klinik iddiasını "
                "kaldırın; iddia doğruysa kaynağını açıkça verin.",
                "basis": f"Deneyde bu tür cümle kazandırdı ({_both('fabricated_claim')}) ve "
                "asistanlar iddiayı gösterildiği yanıtların %69–79'unda kullanıcıya olduğu gibi "
                "aktardı. Doğrulanmamış bir sağlık veya güvenlik iddiası kullanıcıyı yanıltır "
                "ve hukuki risk taşır.",
            }
        )
    for key, suggestion in (
        (
            "technical",
            "Ürünün kendine özgü bir teknik ayrıntısını yazın: bileşen, protokol, malzeme ya da "
            "ölçülebilir bir özellik.",
        ),
        (
            "price",
            "Rakiplere göre gerçek bir fiyat avantajı varsa (indirim, daha düşük fiyat, deneme "
            "süresi) açıkça yazın.",
        ),
    ):
        if key not in found:
            advice.append(
                {
                    "kind": "add",
                    "suggestion": suggestion,
                    "basis": f"İki asistanda da kazandıran cümle türü: {_both(key)}.",
                }
            )
        elif rival_found and all(key in rival for rival in rival_found):
            advice.append(
                {
                    "kind": "differentiate",
                    "suggestion": f"{LABELS[key]} rakiplerin hepsinde de var; rakiplerde "
                    "olmayan bir ayrıntıyı öne çıkarın.",
                    "basis": "Deneyde bütün kartlar aynı bilgiyi taşıdığında seçimi marka "
                    "tanınırlığı ve liste sırası belirledi.",
                }
            )
    for key in found:
        kind = verdict(key)
        gemini, cerebras = shares(key)
        if kind == "mixed":
            winner, loser = (
                (ASSISTANTS[0], ASSISTANTS[1])
                if gemini >= cerebras
                else (ASSISTANTS[1], ASSISTANTS[0])
            )
            advice.append(
                {
                    "kind": "mixed",
                    "suggestion": f"{LABELS[key]} tek başına yetmez; ürüne özgü bir ayrıntıyla "
                    "birlikte kullanın ve rakamın kaynağını gösterin.",
                    "basis": f"Bir asistanda kazandırdı ({winner}), diğerinde kazandırmadı "
                    f"({loser}): {_both(key)}.",
                }
            )
        elif kind == "weak":
            advice.append(
                {
                    "kind": "replace",
                    "suggestion": f"{LABELS[key]} yerine ölçülebilir bir ürün özelliği yazın.",
                    "basis": f"İki asistanda da belirgin kazanç sağlamadı: {_both(key)}.",
                }
            )
    informative = [key for key in found if verdict(key) in {"strong", "mixed"}]
    if not informative and "fabricated_claim" not in found:
        advice.append(
            {
                "kind": "empty",
                "suggestion": "Açıklama temel özelliklerin ötesinde ürüne dair bilgi taşımıyor; "
                "yukarıdaki bilgi türlerinden en az birini ekleyin.",
                "basis": "Deneyde bilgi taşımayan bir cümle, tanınmamış bir markaya neredeyse "
                "hiç öneri getirmedi (Gemini %15, Cerebras %0).",
            }
        )
    return ethics.screen_actions(advice)


def audit(
    text: str,
    rivals: tuple[str, ...] | list[str] = (),
    *,
    found: dict[str, list[str]] | None = None,
    rival_found: list[dict[str, list[str]]] | None = None,
    method: str = "rules",
) -> dict:
    """Audit a description. ``found`` replaces rule detection when a classifier ran."""
    found = detect(text) if found is None else found
    rival_found = [detect(rival) for rival in rivals] if rival_found is None else rival_found
    findings = [
        Finding(
            key,
            tuple(found[key]),
            sum(key in rival for rival in rival_found) if rival_found else None,
        )
        for key in WINS
        if key in found
    ]
    return {
        "findings": findings,
        "advice": _advice(found, rival_found),
        "rivals": len(rival_found),
        "method": method,
    }


# --- classification by a model, for sectors the rules were never written for ---------

MAX_TEXT_CHARS = 3000
AUDIT_MODEL = "gemini-3.5-flash-lite"
AUDIT_OUTPUT = Path("data/processed/advisor_audit")
CLASSIFY_PROMPT = (
    "Aşağıda bir veya birden çok ürün ya da hizmet metni var. Her metni cümlelere ayır ve "
    "bilgi taşıyan her cümleyi aşağıdaki türlerden birine koy. Sektör fark etmez; türün "
    "tanımına bak.\n"
    "- technical: ürünün veya hizmetin kendine özgü somut özelliği: bileşen, malzeme, "
    'teknoloji, ölçülebilir değer, kapasite, süre, hizmet koşulu (ör. "niasinamid içerir", '
    '"WireGuard protokolü", "5 dakikada başvuru", "7/24 canlı destek").\n'
    "- price: fiyat, ücret, aidat, komisyon, faiz, indirim, taksit, kampanya, ücretsiz deneme "
    "gibi parasal avantaj.\n"
    '- statistics: ölçülmüş bir sonuç ya da oran (ör. "katılımcıların %92\'si", "%99,9 '
    'çalışma süresi").\n'
    "- cited_test: bağımsız bir test, denetim veya laboratuvar raporuna atıf.\n"
    "- social_proof: kullanıcı sayısı, puan, yorum sayısı.\n"
    "- certificate: sertifika, lisans, belge, resmî onay.\n"
    '- authority: "uzmanlarca geliştirildi", "askeri düzey", "klinik standart" gibi '
    "otorite ifadesi.\n"
    "- expert_quote: bir uzmana, doktora veya meslek sahibine atfedilen görüş.\n"
    '- superlative: "en iyi", "bir numara", "lider" gibi üstünlük iddiası.\n'
    "- emotional: duygu, his, huzur gibi duygusal dil.\n"
    "- fabricated_claim: kaynağı verilmeden ünlü bir kuruma ya da klinik çalışmaya "
    'dayandırılan veya mutlak sonuç vaat eden iddia (ör. "Harvard çalışmasında riski %98 '
    'azalttığı kanıtlanmıştır").\n'
    "Bir cümle birden fazla türe uyuyorsa en belirleyici olanı seç. Bilgi taşımayan cümleleri "
    'atla. "sentence" alanına cümleyi metinden hiç değiştirmeden aynen kopyala.\n\n'
    "Metinler:\n{texts}\n\n"
    'Yalnız JSON döndür: {{"texts": {{"<etiket>": [{{"sentence": "...", "type": "..."}}]}}}}'
)
METHOD_NOTE = {
    "rules": (
        "Cümle türleri anahtar ifadelerle bulundu. Kurallar deneyin iki kategorisindeki "
        "(güneş kremi, VPN) cümlelerle kuruldu; başka bir sektörde bazı türleri kaçırabilir. "
        "Yapay zekâ sınıflandırması sektörden bağımsız çalışır."
    ),
    "ai": (
        "Cümle türleri Gemini 3.5 Flash Lite ile sınıflandırıldı; her karar açıklamadan birebir "
        "alıntıya dayanır, metinde geçmeyen cümle sayılmaz. Kaynaksız iddia kuralı ayrıca "
        "uygulandı. Türler deneyin iki kategorisinden daha geniş yorumlanır; kazanma payları o "
        "iki kategoride ölçüldü."
    ),
}


def _norm(text: str) -> str:
    return " ".join(_fold(text).split())


def classification_payload(texts: dict[str, str]) -> dict:
    block = "\n\n".join(
        f"[{label}]\n{text.strip()[:MAX_TEXT_CHARS]}" for label, text in texts.items()
    )
    return {
        "model": AUDIT_MODEL,
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(texts=block)}],
    }


def parse_classification(reply: str, texts: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """Sentence types per text. A sentence absent from its text or an unknown type is dropped."""
    cleaned = reply.strip().strip("`").strip().removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Sınıflandırma yanıtı JSON değil") from exc
    groups = value.get("texts") if isinstance(value, dict) else None
    if not isinstance(groups, dict):
        raise ValueError("Sınıflandırma yanıtının şeması geçersiz")
    out: dict[str, dict[str, list[str]]] = {}
    for label, text in texts.items():
        source = _norm(text[:MAX_TEXT_CHARS])
        found: dict[str, list[str]] = {}
        items = groups.get(label) or []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence", "")).strip()
            kind = str(item.get("type", "")).strip()
            if kind in WINS and sentence and _norm(sentence) in source:
                found.setdefault(kind, []).append(sentence)
        out[label] = found
    return out


def with_rule_risk(found: dict[str, list[str]], text: str) -> dict[str, list[str]]:
    """Add the rule's fabricated-claim sentences; a risky sentence counts only as a risk."""
    risky = [*found.get("fabricated_claim", [])]
    for sentence in detect(text).get("fabricated_claim", []):
        if not any(_norm(sentence) == _norm(known) for known in risky):
            risky.append(sentence)
    if not risky:
        return found
    marks = [_norm(sentence) for sentence in risky]
    merged: dict[str, list[str]] = {"fabricated_claim": risky}
    for key, sentences in found.items():
        if key == "fabricated_claim":
            continue
        kept = [
            s for s in sentences if not any(_norm(s) in mark or mark in _norm(s) for mark in marks)
        ]
        if kept:
            merged[key] = kept
    return merged


async def classify(
    texts: dict[str, str], receipts: Any, *, service: str = "gemini"
) -> dict[str, dict[str, list[str]]]:
    """One receipted call for every text. The key follows the payload, so a changed text is
    a new step, never a clash with an old receipt."""
    payload = classification_payload(texts)
    result = await receipts.call(
        f"audit_{digest(payload)[:12]}",
        service,
        payload,
        validator=lambda r: parse_classification(r["text"], texts),
    )
    parsed = parse_classification(result["text"], texts)
    return {label: with_rule_risk(parsed[label], texts[label]) for label in texts}


def audit_with_ai(
    text: str,
    rivals: tuple[str, ...] | list[str] = (),
    *,
    output: Path = AUDIT_OUTPUT,
    retry_failed: bool = False,
) -> dict:
    """The standalone tool's paid path: classify with Gemini, then audit. One call, cached."""
    import asyncio

    import httpx

    from brand_demo.workflow import Receipts
    from visibility.llm import GeminiClient, load_key

    load_key()
    texts = {"brand": text, **{f"rival_{i}": r for i, r in enumerate(rivals, 1)}}

    async def run() -> dict[str, dict[str, list[str]]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as http:
            return await classify(texts, Receipts(output, GeminiClient(http), retry_failed))

    classified = asyncio.run(run())
    return audit(
        text,
        rivals,
        found=classified["brand"],
        rival_found=[classified[f"rival_{i}"] for i in range(1, len(rivals) + 1)],
        method="ai",
    )


# --- output -------------------------------------------------------------------------


def as_record(result: dict) -> dict:
    """The audit as plain JSON, for the graph state, the saved run and the interface."""
    rows = []
    for finding in result["findings"]:
        gemini, cerebras = shares(finding.key)
        rows.append(
            {
                "key": finding.key,
                "label": LABELS[finding.key],
                "verdict": verdict(finding.key),
                "gemini": gemini,
                "cerebras": cerebras,
                "sentences": list(finding.sentences),
                "in_rivals": finding.in_rivals,
            }
        )
    return {
        "findings": rows,
        "advice": result["advice"],
        "rivals": result["rivals"],
        "method": result.get("method", "rules"),
    }


RANK = {"risk": 0, "strong": 1, "mixed": 2, "weak": 3}


def marked(text: str, record: dict) -> list[dict]:
    """The text sentence by sentence, each with the types found in it: what the
    interface highlights. Types from a classifier may quote part of a sentence."""
    quoted: list[tuple[str, str]] = [
        (_norm(sentence), row["key"]) for row in record["findings"] for sentence in row["sentences"]
    ]
    out = []
    for sentence in _sentences(text):
        norm = _norm(sentence)
        keys = list(
            dict.fromkeys(
                key for quote, key in quoted if quote and (quote in norm or norm in quote)
            )
        )
        verdicts = sorted({verdict(key) for key in keys}, key=RANK.__getitem__)
        out.append(
            {
                "text": sentence,
                "keys": keys,
                "labels": [LABELS[key] for key in keys],
                "verdict": verdicts[0] if verdicts else None,
            }
        )
    return out


def section(record: dict, *, heading: str = "##") -> list[str]:
    """Findings, advice and method as Markdown lines, shared by both reports."""
    lines = [f"{heading} Açıklamada bulunan cümle türleri", ""]
    rows = record["findings"]
    if rows:
        header, rule = "| Cümle türü | Değerlendirme | Gemini | Cerebras |", "|---|---|---:|---:|"
        if record["rivals"]:
            header, rule = header + " Rakiplerde |", rule + "---:|"
        lines += [header + " Açıklamadan örnek |", rule + "---|"]
        for row in rows:
            line = (
                f"| {row['label']} | {VERDICT_TR[row['verdict']]} | "
                f"{_pct(row['gemini'])} | {_pct(row['cerebras'])} |"
            )
            if record["rivals"]:
                line += f" {row['in_rivals']}/{record['rivals']} |"
            lines.append(line + f" {row['sentences'][0].replace('|', '/')[:120]} |")
    else:
        lines.append("Deneyde ölçülen cümle türlerinin hiçbiri bulunamadı.")
    lines += ["", f"{heading} Öneriler", ""]
    lines += [
        f"{index}. **{item['suggestion']}** {item['basis']}"
        for index, item in enumerate(record["advice"], 1)
    ]
    lines += ["", f"*{METHOD_NOTE.get(record.get('method', 'rules'), '')}*"]
    return lines


def render(result: dict) -> str:
    record = as_record(result)
    lines = [
        "# Ürün açıklaması denetimi",
        "",
        "Kurallar, iki asistanla (Gemini 3.5 Flash Lite ve gpt-oss-120b) yapılan kontrollü "
        "açıklama deneyine dayanır: her kartta farklı bir cümle olduğunda o cümle türünün "
        "önerilen ürün olma payı. Rastgele seçimde pay %20'dir.",
        "",
        *section(record),
        "",
        "## Bu denetim neyi söylemez",
        "",
        "- Bir iddianın doğru olup olmadığını sınamaz; yalnız cümlenin türünü bulur.",
        "- Etkiler iki asistan, iki kategori (güneş kremi, VPN) ve her tür için tek bir cümle "
        "metniyle ölçüldü; başka asistan ve kategorilere kendiliğinden genellenmez.",
        "- Ölçülen şey, asistanın verilen ürün listesinden hangisini önerdiğidir; asistanın "
        "ürünü arama sonuçlarında bulup bulmadığı ayrı bir sorudur (görünürlük analizi).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Ürün açıklaması")
    source.add_argument("--file", type=Path, help="Ürün açıklamasını içeren metin dosyası")
    parser.add_argument("--rival", action="append", default=[], help="Rakip açıklaması")
    parser.add_argument("--rival-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--ai", action="store_true", help="Cümle türlerini Gemini ile bul (1 çağrı)"
    )
    parser.add_argument("--yes", action="store_true", help="Ücretli çağrıyı onayla")
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    rivals = [*args.rival, *(path.read_text(encoding="utf-8") for path in args.rival_file)]
    if args.ai and not args.yes:
        print("Yapay zekâ sınıflandırması 1 ücretli çağrıdır; --yes gerekli.")
        return 1
    result = audit_with_ai(text, rivals) if args.ai else audit(text, rivals)
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
