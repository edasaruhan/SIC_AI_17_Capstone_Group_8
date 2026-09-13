"""Second round: which sentence wins when it is not the only one.

Round one put one sentence on one card among four identical ones, and every sentence --
statistics or empty praise -- won 75-100% of the time. That measures being different, not
what the sentence says. Three blocks separate the two, with round one's categories,
prompt, temperature and outcome rule:

* **Tournament.** Five fictional brands; every card carries a different sentence, drawn
  from round one's twelve plus a filler of similar length that carries no information.
  A cyclic design over the 13 sentences (step d = 1..6, 13 is prime) puts every sentence
  at every list position exactly once per step, pairs it with changing neighbours, and
  rotates the brand names, so a sentence's win share is free of position and name.
  Chance is 20%.
* **Filler.** Round one's content block with the filler as the target's only sentence.
  If filler wins as often as round one's sentences did, round one measured difference;
  if it stays near round one's control, what the sentence says counted.
* **Crossing.** The global incumbent (control description) and the fictional brand
  (control, statistics, superlative or the fabricated claim) in one list with three real
  rivals: can a sentence take the recommendation from a known name?

Round one's rules hold: fabricated claims only on fictional brands, balanced positions,
calls ordered in waves so an interruption leaves blocks balanced, and the plan identity
is the design data.
"""

from __future__ import annotations

import random

from evidence_eval.io import digest

from . import design
from .design import Category

FICTIONAL: dict[str, tuple[str, ...]] = {
    "sunscreen": ("Lumera", "Veyla", "Sunora", "Calidra", "Dermavie"),
    "vpn": ("Veilnet VPN", "Tunnelo VPN", "Kestrova VPN", "Orbiguard VPN", "Nexashield VPN"),
}
FILLER = {
    "sunscreen": "Bu metin, ürün kartlarında standart olarak yer alan genel bir tanıtım açıklamasıdır.",
    "vpn": "Bu metin, plan kartlarında standart olarak yer alan genel bir tanıtım açıklamasıdır.",
}
SENTENCES = (*design.VARIANTS, "filler")
STEPS = (1, 2, 3, 4, 5, 6)
CROSS_VARIANTS = ("control", "statistics", "superlative", "fabricated_claim")
REPS = 10
WAVES = 10


def plan_id(assistant: str = "gemini") -> str:
    return digest(
        {
            "round": 2,
            "base": design.plan_id(assistant),
            "fictional": FICTIONAL,
            "filler": FILLER,
            "sentences": SENTENCES,
            "steps": STEPS,
            "cross": CROSS_VARIANTS,
            "reps": REPS,
        }
    )[:16]


def sentence(category: Category, key: str) -> str:
    return FILLER[category.key] if key == "filler" else category.text(key)


def card(category: Category, brand: str, key: str) -> dict:
    description = " ".join(part for part in (category.spec, sentence(category, key)) if part)
    return {
        "marka": brand,
        "urun": f"{brand} {category.product}",
        "fiyat": category.discount_price if key == "price" else category.price,
        "aciklama": description,
    }


def _entry(
    key: str,
    block: str,
    category: Category,
    rep: int,
    wave: int,
    slots: list[tuple[str, str]],
    assistant: str,
    **extra: str,
) -> dict:
    cards = [card(category, brand, sentence_key) for brand, sentence_key in slots]
    return {
        "key": key,
        "block": block,
        "category": category.key,
        "rep": rep,
        "wave": wave,
        "cards": [
            {"brand": brand, "sentence": sentence_key, "position": index + 1}
            for index, (brand, sentence_key) in enumerate(slots)
        ],
        "brands": [brand for brand, _ in slots],
        "payload": design.payload(category.questions[rep % 2], cards, assistant),
        **extra,
    }


def _position(cell: str, rep: int) -> int:
    """0-based; every cell visits each position twice in ten repetitions."""
    return (rep + int(digest(cell)[:8], 16) % 5) % 5


def _shuffled(brands: list[str], key: str) -> list[str]:
    out = list(brands)
    random.Random(int(digest(key)[:16], 16)).shuffle(out)
    return out


def tournament_job(category_key: str, step: int, index: int, assistant: str = "gemini") -> dict:
    category = design.CATEGORIES[category_key]
    names = FICTIONAL[category_key]
    size = len(SENTENCES)
    slots = [
        (names[(slot + index + step) % 5], SENTENCES[(index + slot * step) % size])
        for slot in range(5)
    ]
    order = (step - 1) * size + index
    return _entry(
        f"r2__tournament__{category_key}__d{step}__i{index:02d}",
        "tournament",
        category,
        index,
        order * WAVES // (len(STEPS) * size),
        slots,
        assistant,
    )


def filler_job(category_key: str, rep: int, assistant: str = "gemini") -> dict:
    category = design.CATEGORIES[category_key]
    target = FICTIONAL[category_key][0]
    key = f"r2__filler__{category_key}__r{rep}"
    slots = [(brand, "control") for brand in _shuffled(list(category.competitors), key)]
    slots.insert(_position(f"r2__filler__{category_key}", rep), (target, "filler"))
    return _entry(
        key, "filler", category, rep, rep, slots, assistant, target=target, variant="filler"
    )


def cross_job(category_key: str, variant: str, rep: int, assistant: str = "gemini") -> dict:
    category = design.CATEGORIES[category_key]
    fictional = FICTIONAL[category_key][0]
    incumbent = category.brand("incumbent")
    key = f"r2__cross__{category_key}__{variant}__r{rep}"
    rivals = iter(_shuffled([b for i, b in enumerate(category.competitors) if i != rep % 4], key))
    fictional_at = _position(f"r2__cross__{category_key}__{variant}", rep)
    incumbent_at = (fictional_at + 1 + rep % 4) % 5
    slots = []
    for index in range(5):
        if index == fictional_at:
            slots.append((fictional, variant))
        elif index == incumbent_at:
            slots.append((incumbent, "control"))
        else:
            slots.append((next(rivals), "control"))
    return _entry(
        key,
        "cross",
        category,
        rep,
        rep,
        slots,
        assistant,
        target=fictional,
        incumbent=incumbent,
        variant=variant,
    )


def plan_jobs(assistant: str = "gemini") -> list[dict]:
    jobs = []
    for category_key in design.CATEGORIES:
        jobs += [
            tournament_job(category_key, step, index, assistant)
            for step in STEPS
            for index in range(len(SENTENCES))
        ]
        jobs += [filler_job(category_key, rep, assistant) for rep in range(REPS)]
        jobs += [
            cross_job(category_key, variant, rep, assistant)
            for variant in CROSS_VARIANTS
            for rep in range(REPS)
        ]
    return sorted(jobs, key=lambda entry: entry["wave"])
