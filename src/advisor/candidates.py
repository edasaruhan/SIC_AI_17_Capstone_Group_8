"""Which brands compete in a sector nobody curated a list for.

The research used hand-curated registries for five sectors. A tool meant to carry the
signal to *other* sectors cannot depend on them, so the candidate set is built per run:

1. the assistant extracts brand names from the retrieved titles and snippets,
2. every extracted name must literally occur in that text -- a name the model made up
   is dropped, the same fail-closed rule the demo applies to quotes,
3. the user can add names the extraction missed and drop ones it got wrong,
4. where a curated registry exists, its aliases are merged in, because hand-checked
   surface forms are more reliable than extracted ones.

Extraction is not judgement. The model is asked which brand names appear, never which
brand is visible or good; visibility is still measured by matching those names in text.
The dataset's own labels were produced the same way, by an extraction model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from modeling.brands import BrandRegistry, _build, comparison_key, load_registry
from modeling.pairs import text_key

CURATED = ("vpn", "hosting", "travel", "editors", "cosmetics")
MAX_CANDIDATES = 25
EXTRACT_PROMPT = (
    "Aşağıdaki arama sonuçları '{sector}' kategorisiyle ilgili. Metinde adı geçen ticari "
    "marka, ürün veya hizmet adlarını çıkar. Genel kavramları (ör. 'VPN', 'hosting'), "
    "yayıncı veya site adlarını, kişi adlarını çıkarma. Metinde geçmeyen hiçbir adı "
    'ekleme. Yalnız JSON döndür: {{"brands": ["...", "..."]}}\n\n{text}'
)


def extraction_payload(sector: str, texts: list[str], model: str) -> dict:
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": EXTRACT_PROMPT.format(sector=sector, text="\n".join(texts)[:12000]),
            }
        ],
    }


def _json_body(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    value = json.loads(body)
    if not isinstance(value, dict) or not isinstance(value.get("brands"), list):
        raise ValueError("Marka çıkarımı JSON şeması geçersiz")
    return value


def occurs(name: str, corpus_key: str) -> bool:
    """Word-bounded, case- and i-folding-insensitive literal occurrence."""
    key = text_key(name).strip()
    return bool(key) and f" {key} " in corpus_key


def parse_extraction(text: str, corpus: str) -> list[str]:
    """Names the model returned that really occur in the text, deduplicated."""
    corpus_key = text_key(corpus)
    seen: dict[str, str] = {}
    for raw in _json_body(text)["brands"]:
        name = " ".join(str(raw).split())
        if 2 <= len(name) <= 60 and occurs(name, corpus_key):
            seen.setdefault(comparison_key(name), name)
    return list(seen.values())[:MAX_CANDIDATES]


def apply_corrections(names: Iterable[str], add: Iterable[str], drop: Iterable[str]) -> list[str]:
    """User corrections win: added names are kept even if extraction missed them."""
    dropped = {comparison_key(name) for name in drop}
    merged: dict[str, str] = {}
    for name in [*names, *add]:
        key = comparison_key(name)
        if key and key not in dropped:
            merged.setdefault(key, name)
    return list(merged.values())


def build_registry(
    brand: str,
    brand_aliases: Iterable[str],
    rivals: Iterable[str],
    sector: str,
) -> BrandRegistry:
    """One registry for this run: curated aliases where they exist, extracted names otherwise."""
    canonical: dict[str, list[str]] = {}
    keys: dict[str, str] = {}

    def add(display: str, aliases: Iterable[str] = ()) -> None:
        key = comparison_key(display)
        if not key:
            return
        owner = keys.get(key, display)
        forms = canonical.setdefault(owner, [])
        for alias in aliases:
            alias_key = comparison_key(alias)
            if alias_key and alias_key not in keys:
                forms.append(alias)
                keys[alias_key] = owner
        keys[key] = owner

    if sector in CURATED:
        curated = load_registry(sector)
        for display, forms in curated.surface_forms.items():
            add(display, [f for f in forms if f != display])
    add(brand, brand_aliases)
    for name in rivals:
        add(name)
    return _build(f"advisor:{sector}", {"canonical": canonical})
