"""Outcomes and effects of the description experiment.

Outcomes come from name matching over the five brands shown in the call, never from a
model's judgement. The pilot showed that the first brand named is a poor proxy here: with
identical cards the assistant often restates the whole list before recommending one
product. The recommended brand is therefore read in order of evidence:

1. the first bold span (``**...**``) that names exactly one brand -- the assistant bolds
   the product it recommends;
2. otherwise the first sentence with "öner" or "tercih" that names exactly one brand;
3. otherwise the first brand named, flagged as such in ``pick_method``.

Whether the target is named at all and its rank by first mention are kept as well.

Each cell visits every list position equally often, so calls are treated as independent
observations and intervals come from a bootstrap over calls (seed 42, 2.5/97.5). A
contrast is the difference between two independent groups of calls, bootstrapped as
such. With ten repetitions per arm the intervals are wide; the report says so.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from advisor import candidates, measure
from evidence_eval.io import read_json
from modeling.features import select

from . import design

N_RESAMPLES = 2000
SEED = 42
VARIANT_TR = {
    "control": "Kontrol (yalnız temel özellikler)",
    "statistics": "Sayısal kanıt / istatistik",
    "cited_test": "Bağımsız test raporu gösterme",
    "expert_quote": "Uzman alıntısı",
    "authority": "Otorite dili",
    "social_proof": "Sosyal kanıt (puan, kullanıcı sayısı)",
    "certificate": "Sertifika",
    "superlative": "Üstünlük dili ('en iyi', 'bir numara')",
    "emotional": "Duygusal dil",
    "technical": "Teknik detay",
    "price": "Fiyat avantajı",
    "fabricated_claim": "Uydurma klinik/denetim iddiası",
}
IDENTITY_TR = {
    "incumbent": "Küresel yerleşik",
    "small": "Küçük gerçek marka",
    "fictional": "Kurgusal marka",
}
CATEGORY_TR = {"sunscreen": "Güneş kremi", "vpn": "VPN", "ALL": "Tümü"}


BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
SENTENCE = re.compile(r"[^.!?\n]+")
RECOMMEND = re.compile(r"öner|tercih", re.I)


def recommended(text: str, registry) -> tuple[str | None, str]:
    """The brand the answer recommends, and which rule found it."""
    for span in BOLD.findall(text):
        named = measure.named_brands(span, registry)
        if len(named) == 1:
            return named[0], "bold"
    for sentence in SENTENCE.findall(text):
        if RECOMMEND.search(sentence):
            named = measure.named_brands(sentence, registry)
            if len(named) == 1:
                return named[0], "sentence"
    named = measure.named_brands(text, registry)
    return (named[0], "first_mention") if named else (None, "none")


def outcome(text: str, brands: list[str], target: str) -> dict:
    registry = candidates.build_registry(brands[0], [], brands[1:], "description-lab")
    named = measure.named_brands(text, registry)
    picked, method = recommended(text, registry)
    return {
        "first": int(picked == target),
        "mentioned": int(target in named),
        "rank": named.index(target) + 1 if target in named else np.nan,
        "n_named": len(named),
        "first_brand": picked,
        "pick_method": method,
        "first_named": named[0] if named else None,
    }


def outcomes(folder: Path, jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in jobs:
        path = folder / "steps" / f"{entry['key']}.json"
        step = read_json(path) if path.exists() else None
        if not step or step.get("status") != "completed":
            continue
        listed = [
            c["marka"] for c in json.loads(entry["payload"]["messages"][1]["content"])["urunler"]
        ]
        result = outcome(step["result"]["text"], entry["brands"], entry["target"])
        picked = result["first_brand"]
        rows.append(
            {
                key: entry[key]
                for key in ("key", "category", "identity", "variant", "rep", "target", "position")
            }
            | result
            | {"picked_position": listed.index(picked) + 1 if picked in listed else np.nan}
        )
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    draws = values[rng.integers(0, len(values), (N_RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    draws = a[rng.integers(0, len(a), (N_RESAMPLES, len(a)))].mean(axis=1) - b[
        rng.integers(0, len(b), (N_RESAMPLES, len(b)))
    ].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _scopes(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("ALL", table),
        *[(str(key), group) for key, group in table.groupby("category", sort=True)],
    ]


def _contrast_row(scope: str, label: str, arm: pd.DataFrame, base: pd.DataFrame, rng) -> dict:
    row: dict = {"scope": scope, "arm": label, "n": len(arm), "n_base": len(base)}
    for metric in ("first", "mentioned"):
        a = arm[metric].to_numpy(dtype=float)
        b = base[metric].to_numpy(dtype=float)
        row[f"{metric}_rate"] = float(a.mean()) if len(a) else np.nan
        row[f"{metric}_rate_lo"], row[f"{metric}_rate_hi"] = _bootstrap_mean(a, rng)
        row[f"{metric}_effect"] = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
        row[f"{metric}_effect_lo"], row[f"{metric}_effect_hi"] = _bootstrap_diff(a, b, rng)
    return row


def identity_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Control description; each identity against the fictional brand."""
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in _scopes(table):
        control = select(part, part["variant"] == "control")
        base = select(control, control["identity"] == "fictional")
        for identity in design.IDENTITIES:
            arm = select(control, control["identity"] == identity)
            rows.append(_contrast_row(scope, identity, arm, base, rng))
    return pd.DataFrame(rows)


def content_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Fictional brand; each description variant against the control description."""
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in _scopes(table):
        fictional = select(part, part["identity"] == "fictional")
        base = select(fictional, fictional["variant"] == "control")
        for variant in design.VARIANTS:
            arm = select(fictional, fictional["variant"] == variant)
            rows.append(_contrast_row(scope, variant, arm, base, rng))
    return pd.DataFrame(rows)


def position_effects(table: pd.DataFrame) -> pd.DataFrame:
    """Every call, by the target card's position in the shuffled list."""
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in _scopes(table):
        base = select(part, part["position"] == 5)
        for position in range(1, 6):
            arm = select(part, part["position"] == position)
            rows.append(_contrast_row(scope, str(position), arm, base, rng))
    return pd.DataFrame(rows)


def pick_share(table: pd.DataFrame) -> pd.DataFrame:
    """Which list position gets recommended, over every call and every card.

    Uses all five cards of every call rather than only the target, so the position
    signal rests on the whole experiment. Chance is 20%.
    """
    if "picked_position" not in table:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in _scopes(table):
        picks = part["picked_position"].dropna().to_numpy(dtype=float)
        for position in range(1, 6):
            hits = (picks == position).astype(float)
            low, high = _bootstrap_mean(hits, rng)
            rows.append(
                {
                    "scope": scope,
                    "position": position,
                    "n": len(picks),
                    "share": float(hits.mean()) if len(hits) else np.nan,
                    "share_lo": low,
                    "share_hi": high,
                }
            )
    return pd.DataFrame(rows)


def _pts(value: float) -> str:
    return "–" if pd.isna(value) else f"{100 * value:+.1f}"


def _pct(value: float) -> str:
    return "–" if pd.isna(value) else f"%{100 * value:.0f}"


def _ci(row: pd.Series, metric: str) -> str:
    low, high = float(row[f"{metric}_effect_lo"]), float(row[f"{metric}_effect_hi"])
    return "–" if np.isnan(low) else f"[{_pts(low)} · {_pts(high)}]"


def _table(frame: pd.DataFrame, labels: dict[str, str], base_arm: str) -> list[str]:
    lines = [
        "| Kapsam | Kol | n | Birinci önerilme | Fark (puan) [95% GA] | Anılma | Fark (puan) [95% GA] |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for _, r in frame.iterrows():
        base = r.arm == base_arm
        lines.append(
            f"| {CATEGORY_TR.get(r.scope, r.scope)} | {labels.get(r.arm, r.arm)} | {int(r.n)} | "
            f"{_pct(r.first_rate)} | {'taban' if base else _pts(r.first_effect) + ' ' + _ci(r, 'first')} | "
            f"{_pct(r.mentioned_rate)} | {'taban' if base else _pts(r.mentioned_effect) + ' ' + _ci(r, 'mentioned')} |"
        )
    return lines


def _pick_lines(table: pd.DataFrame) -> list[str]:
    picks = pick_share(table)
    if picks.empty:
        return []
    lines = [
        "## 4. Hangi sıradaki kart öneriliyor? (bütün kartlar)",
        "",
        "Her çağrıda önerilen ürünün listedeki sırası. Rastgele seçimde her sıra %20 olurdu.",
        "",
        "| Kapsam | Sıra | Önerilme payı [95% GA] | Çağrı |",
        "|---|---:|---|---:|",
    ]
    for _, r in picks.iterrows():
        lines.append(
            f"| {CATEGORY_TR.get(r.scope, r.scope)} | {int(r.position)} | {_pct(float(r.share))} "
            f"[{_pct(float(r.share_lo))} · {_pct(float(r.share_hi))}] | {int(r.n)} |"
        )
    return [*lines, ""]


def render(
    table: pd.DataFrame,
    identity: pd.DataFrame,
    content: pd.DataFrame,
    position: pd.DataFrame,
    *,
    planned: int,
) -> str:
    lines = [
        "# Açıklama deneyi: yapay zekâ bir ürünü neye göre öneriyor?",
        "",
        "Bu sayfa `python -m description_lab analyze` ile üretilir; elle düzenlemeyin.",
        "Tasarım ve önceden sabitlenen kurallar `src/description_lab/design.py` başındadır.",
        "",
        f"- Asistan: `{design.MODEL}`, sıcaklık {design.TEMPERATURE}.",
        f"- Tamamlanan çağrı: {len(table)}/{planned}.",
        "- Her çağrıda aynı temel özelliklere sahip 5 ürün kartı; her hücrede hedef kart her "
        "sıraya eşit sayıda konur, rakiplerin sırası çağrı başına karışır.",
        "- Ölçüm: gösterilen 5 markanın ad eşleştirmesi. 'Birinci önerilme' = yanıtın önerdiği "
        "marka hedef mi (önce kalın yazılan tek marka, yoksa 'öner/tercih' cümlesindeki tek "
        "marka, yoksa ilk anılan).",
        f"- Önerilen marka kuralla bulunamayıp ilk anılana düşülen çağrı: "
        f"{int((table.get('pick_method', pd.Series(dtype=str)) == 'first_mention').sum())}.",
        "",
        "## 1. Marka adı: aynı ürün, farklı marka",
        "",
        "Hedefin açıklaması kontrol metni; yalnız markası değişiyor. Taban: kurgusal marka.",
        "",
        *_table(identity, IDENTITY_TR, "fictional"),
        "",
        "## 2. Açıklama içeriği: aynı kurgusal marka, farklı metin",
        "",
        "Hedef kurgusal marka; yalnız açıklamasına eklenen cümle değişiyor. Taban: kontrol metni.",
        "",
        *_table(content, VARIANT_TR, "control"),
        "",
        "## 3. Liste sırası",
        "",
        "Bütün çağrılar, hedef kartın listedeki yerine göre. Taban: 5. sıra.",
        "",
        *_table(position, {str(i): f"{i}. sıra" for i in range(1, 6)}, "5"),
        "",
        *_pick_lines(table),
        "## Sınırlılıklar",
        "",
        "- Tek asistan (Gemini 3.5 Flash Lite); önerinin 'en az iki model' koşulu karşılanmadı.",
        "- Kategori başına tek ürün seti ve iki soru kalıbı; sonuç başka ürün setlerine kendiliğinden "
        "genellenmez.",
        "- Kol başına az tekrar; güven aralıkları geniştir, sıfırı içeren farklarda yön çağrılmaz.",
        "- Varyant cümleleri açıklamaya bilgi ve uzunluk ekler; 'içerik' etkisi uzunluktan tam "
        "ayrıştırılmadı.",
        "- Kimlik ve içerik tam çaprazlanmadı: içeriğin yerleşik markayı yenip yenemediği bu tasarımda "
        "doğrudan ölçülmedi.",
        "- Uydurma iddialar yalnız kurgusal markaya eklendi; ürün bunları önermez, yalnız risk olarak "
        "raporlar.",
    ]
    return "\n".join(lines) + "\n"
