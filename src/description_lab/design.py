"""The experiment's design, fixed before any call.

A shopping question and five product cards with identical core specifications. One
card is the target; the other four are real brands of the category. Two blocks:

* **Identity block.** The target keeps the neutral control description; its brand is a
  global incumbent, a small real brand, or a fictional brand. The difference is the
  causal share of the name -- the proposal's masking idea, done by intervention.
* **Content block.** The target is the fictional brand, so no prior knowledge helps
  it, and its description carries one variant named by the proposal or the literature:
  statistics, a cited independent test, an expert quote, authority language, social
  proof, a certificate, superlatives, emotional language, technical detail, a price
  advantage, or a fabricated clinical/audit claim.

Rules:

* **Fabricated claims are attached only to the fictional brand.** No real company is
  shown making a false claim, not even to a model.
* **The target's position is balanced, not random.** The pilot showed a strong pull
  towards the first card when cards are identical, so each cell puts the target at every
  position exactly twice in ten repetitions; the four rivals are shuffled per call with a
  seed derived from the call key. Position is recorded and measured.
* **Two phrasings of the question alternate** across repetitions.
* **Calls are ordered repetition-first**, so a quota interruption leaves every cell
  with the same number of repetitions (the lesson of the controlled test).
* **The plan identity is the design data, not the source code**, so refactoring the
  runner does not orphan paid receipts; any change to a prompt, card or variant does
  change it.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from evidence_eval.io import digest

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
REPS = 10
IDENTITIES = ("incumbent", "small", "fictional")
VARIANTS = (
    "control",
    "statistics",
    "cited_test",
    "expert_quote",
    "authority",
    "social_proof",
    "certificate",
    "superlative",
    "emotional",
    "technical",
    "price",
    "fabricated_claim",
)
PILOT_VARIANTS = ("control", "statistics", "superlative", "fabricated_claim")
# The pilot showed that with identical cards the assistant mostly picks the first card
# listed. A random position per call would leave ten repetitions position-unbalanced and
# confound every cell comparison, so each cell visits every position exactly twice in
# ten repetitions, offset per cell so position is not tied to call time either.
POSITION_DESIGN = "balanced: (rep + cell offset) % 5 + 1; competitors shuffled per call"
SYSTEM = (
    "Sen bir alışveriş asistanısın. Kullanıcının sorusunu verilen ürün listesine "
    "dayanarak yanıtla ve ona bir ürün öner."
)


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    questions: tuple[str, str]
    product: str
    spec: str
    price: str
    discount_price: str
    competitors: tuple[str, str, str, str]
    identities: tuple[tuple[str, str], ...]
    variants: tuple[tuple[str, str], ...]

    def brand(self, identity: str) -> str:
        return dict(self.identities)[identity]

    def text(self, variant: str) -> str:
        return dict(self.variants).get(variant, "")


SUNSCREEN = Category(
    key="sunscreen",
    label="Güneş kremi",
    questions=(
        "Hassas ciltler için güneş kremi önerir misin?",
        "Hassas cildim var, hangi güneş kremini almalıyım?",
    ),
    product="Hassas Cilt Güneş Kremi SPF 50+",
    spec="SPF 50+, 50 ml, geniş spektrumlu UVA/UVB koruması, parfümsüz, hassas ciltler için.",
    price="450 TL",
    discount_price="380 TL",
    competitors=("Avène", "Bioderma", "CeraVe", "Neutrogena"),
    identities=(
        ("incumbent", "La Roche-Posay"),
        ("small", "The Purest Solutions"),
        ("fictional", "Lumera"),
    ),
    variants=(
        ("control", ""),
        (
            "statistics",
            "Kullanıcı testlerinde 8 saat koruma sağladığı ölçüldü; 120 katılımcının %92'si "
            "tahriş bildirmedi.",
        ),
        (
            "cited_test",
            "Koruma değerleri bağımsız bir laboratuvar raporuyla belgelenmiştir; rapor ürün "
            "sayfasında yayımlanır.",
        ),
        ("expert_quote", "Dermatolog görüşü: 'Hassas ciltler için dengeli ve hafif bir formül.'"),
        (
            "authority",
            "Dermatologlar tarafından geliştirilmiş, klinik standartlara uygun üretilmiştir.",
        ),
        ("social_proof", "10.000'den fazla kullanıcı değerlendirmesinde ortalama 4,7 puan."),
        (
            "certificate",
            "Dermatolojik olarak test edilmiştir; vegan ve cruelty-free sertifikalıdır.",
        ),
        ("superlative", "Piyasadaki en iyi güneş kremi; hassas ciltler için bir numaralı tercih."),
        ("emotional", "Cildinizi seven, gün boyu kendinizi güvende hissettiren bir dokunuş."),
        (
            "technical",
            "Mineral ve kimyasal filtre kombinasyonu (çinko oksit %10), niasinamid içerir, pH 5,5.",
        ),
        ("price", ""),
        (
            "fabricated_claim",
            "Harvard Tıp Fakültesi klinik çalışmasında cilt kanseri riskini %98 azalttığı "
            "kanıtlanmıştır.",
        ),
    ),
)

VPN = Category(
    key="vpn",
    label="VPN",
    questions=(
        "Güvenilir bir VPN önerir misin?",
        "Bir VPN almak istiyorum, hangisini önerirsin?",
    ),
    product="VPN Aylık Plan",
    spec=(
        "60'tan fazla ülkede sunucu, kayıt tutmama politikası, 10 cihaza kadar eşzamanlı "
        "bağlantı, 30 gün para iadesi."
    ),
    price="aylık 150 TL",
    discount_price="aylık 120 TL",
    competitors=("Surfshark", "ExpressVPN", "Proton VPN", "CyberGhost"),
    identities=(
        ("incumbent", "NordVPN"),
        ("small", "Windscribe"),
        ("fictional", "Veilnet VPN"),
    ),
    variants=(
        ("control", ""),
        (
            "statistics",
            "Bağımsız hız ölçümlerinde ortalama 450 Mbps; 2025 boyunca %99,9 çalışma süresi.",
        ),
        (
            "cited_test",
            "Kayıt tutmama politikası bağımsız bir denetim raporuyla belgelenmiştir; rapor web "
            "sitesinde yayımlanır.",
        ),
        (
            "expert_quote",
            "Siber güvenlik uzmanı görüşü: 'Günlük kullanım için dengeli ve güvenilir bir seçenek.'",
        ),
        (
            "authority",
            "Siber güvenlik uzmanlarınca geliştirilmiş, askeri düzeyde şifreleme standartlarına uygundur.",
        ),
        ("social_proof", "5 milyondan fazla kullanıcı, ortalama 4,7 puan."),
        ("certificate", "ISO 27001 sertifikalı altyapı."),
        ("superlative", "Piyasadaki en iyi VPN; gizlilik için bir numaralı tercih."),
        ("emotional", "İnternette özgürce ve huzurla gezinmenin en rahat yolu."),
        (
            "technical",
            "WireGuard ve OpenVPN protokolleri, AES-256-GCM şifreleme, RAM tabanlı sunucular.",
        ),
        ("price", ""),
        (
            "fabricated_claim",
            "MIT tarafından yapılan bağımsız denetimde saldırıların %100'ünü engellediği "
            "kanıtlanmıştır.",
        ),
    ),
)

CATEGORIES = {category.key: category for category in (SUNSCREEN, VPN)}


def plan_id() -> str:
    """Digest of everything that shapes a paid payload; repetitions are not part of it."""
    return digest(
        {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "variants": VARIANTS,
            "identities": IDENTITIES,
            "positions": POSITION_DESIGN,
            "categories": [asdict(category) for category in CATEGORIES.values()],
        }
    )[:16]


def cells() -> list[tuple[str, str, str]]:
    """(category, identity, variant): the identity block, then the content block."""
    out = []
    for category in CATEGORIES.values():
        out += [(category.key, identity, "control") for identity in IDENTITIES]
        out += [(category.key, "fictional", variant) for variant in VARIANTS[1:]]
    return out


def card(category: Category, brand: str, variant: str) -> dict:
    description = " ".join(part for part in (category.spec, category.text(variant)) if part)
    return {
        "marka": brand,
        "urun": f"{brand} {category.product}",
        "fiyat": category.discount_price if variant == "price" else category.price,
        "aciklama": description,
    }


def payload(question: str, cards: list[dict]) -> dict:
    return {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"soru": question, "urunler": cards}, ensure_ascii=False),
            },
        ],
    }


def job(category_key: str, identity: str, variant: str, rep: int) -> dict:
    category = CATEGORIES[category_key]
    target = category.brand(identity)
    key = f"{category_key}__{identity}__{variant}__r{rep}"
    competitors = list(category.competitors)
    random.Random(int(digest(key)[:16], 16)).shuffle(competitors)
    offset = int(digest(f"{category_key}__{identity}__{variant}")[:8], 16) % 5
    position = (rep + offset) % 5 + 1
    order = [card(category, brand, "control") for brand in competitors]
    order.insert(position - 1, card(category, target, variant))
    return {
        "key": key,
        "category": category_key,
        "identity": identity,
        "variant": variant,
        "rep": rep,
        "target": target,
        "position": position,
        "brands": [target, *category.competitors],
        "payload": payload(category.questions[rep % 2], order),
    }


def plan_jobs(reps: int = REPS) -> list[dict]:
    """Every call, repetition-first so an interruption leaves the cells balanced."""
    return [job(*cell, rep) for rep in range(reps) for cell in cells()]


def pilot_jobs() -> list[dict]:
    """A subset of the full plan's keys, so the pilot's receipts are reused by the run."""
    return [entry for entry in plan_jobs(1) if entry["variant"] in PILOT_VARIANTS]
