"""Outcomes, shares and the report of round two.

The recommended brand is read with round one's rule (``analysis.recommended``) and mapped
back to the sentence its card carried. Intervals are a bootstrap over calls (seed 42,
2.5/97.5); a tournament share resamples whole calls, so a sentence's appearances and wins
move together.

Whether the answer repeats the fabricated claim is a name match on the claim's source
(Harvard, MIT). Whether it warns is a keyword match, so it is an upper bound: a word like
"iddia" can appear without any doubt being raised.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from advisor import candidates
from evidence_eval.io import read_json
from modeling.features import select

from . import analysis, design, round2

N_RESAMPLES = analysis.N_RESAMPLES
SEED = analysis.SEED
CLAIM = {"sunscreen": re.compile(r"Harvard"), "vpn": re.compile(r"\bMIT\b")}
WARN = re.compile(
    r"doğrulanmamış|doğrulay|teyit|kanıtlanmamış|iddia|şüphe|abartılı|temkin|ihtiyat",
    re.I,
)
SENTENCE_TR = {**analysis.VARIANT_TR, "filler": "Bilgisiz dolgu cümlesi (uzunluk kontrolü)"}
CATEGORY_TR = analysis.CATEGORY_TR
BLOCK_TR = {"tournament": "Turnuva", "cross": "Çaprazlama", "ALL": "Tümü"}


def outcomes(folder: Path, jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in jobs:
        path = folder / "steps" / f"{entry['key']}.json"
        step = read_json(path) if path.exists() else None
        if not step or step.get("status") != "completed":
            continue
        text = step["result"]["text"]
        brands = entry["brands"]
        registry = candidates.build_registry(brands[0], [], brands[1:], "description-lab")
        picked, method = analysis.recommended(text, registry, brands)
        chosen = next((c for c in entry["cards"] if c["brand"] == picked), None)
        fabricated = any(c["sentence"] == "fabricated_claim" for c in entry["cards"])
        rows.append(
            {
                "key": entry["key"],
                "block": entry["block"],
                "category": entry["category"],
                "rep": entry["rep"],
                "variant": entry.get("variant"),
                "picked": picked,
                "pick_method": method,
                "picked_sentence": chosen["sentence"] if chosen else None,
                "picked_position": chosen["position"] if chosen else np.nan,
                "first": int(picked is not None and picked == entry.get("target")),
                "incumbent_first": int(picked is not None and picked == entry.get("incumbent")),
                "fabricated_shown": int(fabricated),
                "claim_echo": (
                    int(bool(CLAIM[entry["category"]].search(text))) if fabricated else np.nan
                ),
                "warned": int(bool(WARN.search(text))) if fabricated else np.nan,
                "sentences": "|".join(c["sentence"] for c in entry["cards"]),
            }
        )
    return pd.DataFrame(rows)


def _scopes(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("ALL", table),
        *[(str(key), group) for key, group in table.groupby("category", sort=True)],
    ]


def _rate(values: pd.Series | pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).ravel()
    array = array[~np.isnan(array)]
    low, high = analysis._bootstrap_mean(array, rng)
    return (float(array.mean()) if len(array) else np.nan), low, high


def win_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Tournament: wins / appearances per sentence, bootstrapped over whole calls."""
    rng = np.random.default_rng(SEED)
    rows = []
    tournament = select(table, table["block"] == "tournament")
    for scope, part in _scopes(tournament):
        if part.empty:
            continue
        appear = np.array(
            [[key in cell.split("|") for key in round2.SENTENCES] for cell in part["sentences"]],
            dtype=float,
        )
        win = np.array(
            [[key == picked for key in round2.SENTENCES] for picked in part["picked_sentence"]],
            dtype=float,
        )
        index = rng.integers(0, len(part), (N_RESAMPLES, len(part)))
        with np.errstate(invalid="ignore", divide="ignore"):
            shares = win[index].sum(axis=1) / appear[index].sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a sentence absent from a scope
            low, high = np.nanpercentile(shares, [2.5, 97.5], axis=0)
        for j, key in enumerate(round2.SENTENCES):
            appearances, wins = int(appear[:, j].sum()), int(win[:, j].sum())
            rows.append(
                {
                    "scope": scope,
                    "sentence": key,
                    "appearances": appearances,
                    "wins": wins,
                    "share": wins / appearances if appearances else np.nan,
                    "share_lo": float(low[j]),
                    "share_hi": float(high[j]),
                }
            )
    return pd.DataFrame(rows)


def filler_contrast(table: pd.DataFrame, first_round: pd.DataFrame | None) -> pd.DataFrame:
    """Filler as the only sentence, against round one's control and its eleven sentences."""
    rng = np.random.default_rng(SEED)
    rows = []
    filler = select(table, table["block"] == "filler")
    for scope, part in _scopes(filler):
        rate, low, high = _rate(part["first"], rng)
        row: dict = {
            "scope": scope,
            "n": len(part),
            "filler": rate,
            "filler_lo": low,
            "filler_hi": high,
        }
        if first_round is not None and not first_round.empty:
            base = (
                first_round
                if scope == "ALL"
                else select(first_round, first_round["category"] == scope)
            )
            fictional = select(base, base["identity"] == "fictional")
            control = select(fictional, fictional["variant"] == "control")["first"].to_numpy(
                dtype=float
            )
            sentences = select(fictional, fictional["variant"] != "control")["first"].to_numpy(
                dtype=float
            )
            values = part["first"].to_numpy(dtype=float)
            row |= {
                "control": float(control.mean()) if len(control) else np.nan,
                "sentences": float(sentences.mean()) if len(sentences) else np.nan,
                "filler_vs_control": (
                    float(values.mean() - control.mean()) if len(control) else np.nan
                ),
                "sentences_vs_filler": (
                    float(sentences.mean() - values.mean()) if len(sentences) else np.nan
                ),
            }
            row["filler_vs_control_lo"], row["filler_vs_control_hi"] = analysis._bootstrap_diff(
                values, control, rng
            )
            row["sentences_vs_filler_lo"], row["sentences_vs_filler_hi"] = analysis._bootstrap_diff(
                sentences, values, rng
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cross_shares(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    cross = select(table, table["block"] == "cross")
    for scope, part in _scopes(cross):
        for variant in round2.CROSS_VARIANTS:
            arm = select(part, part["variant"] == variant)
            fictional, fictional_lo, fictional_hi = _rate(arm["first"], rng)
            incumbent, incumbent_lo, incumbent_hi = _rate(arm["incumbent_first"], rng)
            rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "n": len(arm),
                    "fictional": fictional,
                    "fictional_lo": fictional_lo,
                    "fictional_hi": fictional_hi,
                    "incumbent": incumbent,
                    "incumbent_lo": incumbent_lo,
                    "incumbent_hi": incumbent_hi,
                }
            )
    return pd.DataFrame(rows)


def echo_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Calls that showed the fabricated claim: was it picked, repeated, doubted?"""
    rng = np.random.default_rng(SEED)
    shown = select(table, table["fabricated_shown"] == 1)
    rows = []
    blocks = [("ALL", shown), *[(str(k), g) for k, g in shown.groupby("block", sort=True)]]
    for block, part in blocks:
        picked = (part["picked_sentence"] == "fabricated_claim").astype(float)
        not_picked = select(part, part["picked_sentence"] != "fabricated_claim")
        row: dict = {"block": block, "n": len(part)}
        for name, values in (
            ("picked", picked),
            ("echo", part["claim_echo"]),
            ("echo_when_not_picked", not_picked["claim_echo"]),
            ("warned", part["warned"]),
        ):
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = _rate(values, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def _pct(value: float) -> str:
    return analysis._pct(value)


def _band(row: pd.Series, name: str) -> str:
    return f"{_pct(float(row[name]))} [{_pct(float(row[f'{name}_lo']))} · {_pct(float(row[f'{name}_hi']))}]"


def _value(row: pd.Series, name: str) -> float:
    return float(row[name]) if name in row.index else np.nan


def _diff(row: pd.Series, name: str) -> str:
    if np.isnan(_value(row, name)):
        return "–"
    return (
        f"{analysis._pts(float(row[name]))} "
        f"[{analysis._pts(float(row[f'{name}_lo']))} · {analysis._pts(float(row[f'{name}_hi']))}]"
    )


def render(
    table: pd.DataFrame,
    shares: pd.DataFrame,
    filler: pd.DataFrame,
    cross: pd.DataFrame,
    echo: pd.DataFrame,
    *,
    assistant: str,
    planned: int,
) -> str:
    chosen = design.ASSISTANTS[assistant]
    lines = [
        f"# Açıklama deneyi, 2. tur: hangi cümle kazanıyor? — {chosen.label}",
        "",
        "Bu sayfa `python -m description_lab analyze --round 2` ile üretilir; elle düzenlemeyin.",
        "Tasarım `src/description_lab/round2.py` başındadır.",
        "",
        f"- Asistan: `{chosen.model}`, sıcaklık {design.TEMPERATURE}.",
        f"- Tamamlanan çağrı: {len(table)}/{planned}.",
        "- Önerilen marka 1. turun kuralıyla okunur; kural bulamayıp ilk anılana düşülen çağrı: "
        f"{int((table.get('pick_method', pd.Series(dtype=str)) == 'first_mention').sum())}.",
        "",
        "## 1. Turnuva: her kartta farklı bir cümle",
        "",
        "Beş kurgusal marka, her kartta 13 cümleden farklı biri. Her cümle her sıraya eşit "
        "sayıda konur. Şans payı %20.",
        "",
    ]
    for scope, part in shares.groupby("scope", sort=False):
        lines += [
            f"**{CATEGORY_TR.get(str(scope), str(scope))}**",
            "",
            "| Cümle | Görünme | Kazanma | Kazanma payı [95% GA] |",
            "|---|---:|---:|---|",
        ]
        for _, r in part.sort_values("share", ascending=False).iterrows():
            lines.append(
                f"| {SENTENCE_TR.get(r.sentence, r.sentence)} | {int(r.appearances)} | "
                f"{int(r.wins)} | {_band(r, 'share')} |"
            )
        lines.append("")
    lines += [
        "## 2. Dolgu cümlesi: fark mı, bilgi mi?",
        "",
        "Hedef kurgusal markanın tek cümlesi bilgisiz dolgu. 1. turun aynı asistandaki kontrolü "
        "(cümlesiz) ve on bir cümlenin ortalamasıyla karşılaştırılır.",
        "",
        "| Kapsam | n | Dolgu ile birinci önerilme | 1. tur kontrol | 1. tur cümleler | "
        "Dolgu − kontrol (puan) | Cümleler − dolgu (puan) |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for _, r in filler.iterrows():
        lines.append(
            f"| {CATEGORY_TR.get(r.scope, r.scope)} | {int(r.n)} | {_band(r, 'filler')} | "
            f"{_pct(_value(r, 'control'))} | {_pct(_value(r, 'sentences'))} | "
            f"{_diff(r, 'filler_vs_control')} | {_diff(r, 'sentences_vs_filler')} |"
        )
    lines += [
        "",
        "## 3. Çaprazlama: cümle, tanınan markayı yenebilir mi?",
        "",
        "Aynı listede küresel yerleşik marka (cümlesiz) ve kurgusal marka (cümleyle), yanında "
        "üç gerçek rakip.",
        "",
        "| Kapsam | Kurgusal markanın cümlesi | n | Kurgusal önerildi | Yerleşik önerildi |",
        "|---|---|---:|---|---|",
    ]
    for _, r in cross.iterrows():
        lines.append(
            f"| {CATEGORY_TR.get(r.scope, r.scope)} | {SENTENCE_TR.get(r.variant, r.variant)} | "
            f"{int(r.n)} | {_band(r, 'fictional')} | {_band(r, 'incumbent')} |"
        )
    lines += [
        "",
        "## 4. Uydurma iddia: seçiliyor mu, tekrarlanıyor mu, sorgulanıyor mu?",
        "",
        "İddianın gösterildiği çağrılar. 'Tekrar' = yanıt iddianın kaynağını (Harvard/MIT) anıyor. "
        "'Uyarı' anahtar kelime eşleşmesidir, üst sınırdır.",
        "",
        "| Blok | n | İddialı kart önerildi | İddia tekrarlandı | Önerilmediğinde bile tekrar | Uyarı |",
        "|---|---:|---|---|---|---|",
    ]
    for _, r in echo.iterrows():
        lines.append(
            f"| {BLOCK_TR.get(r.block, r.block)} | {int(r.n)} | {_band(r, 'picked')} | "
            f"{_band(r, 'echo')} | {_band(r, 'echo_when_not_picked')} | {_band(r, 'warned')} |"
        )
    picks = analysis.pick_share(select(table, table["block"] == "tournament"))
    if not picks.empty:
        lines += [
            "",
            "## 5. Turnuvada önerilen kartın sırası",
            "",
            "| Kapsam | Sıra | Önerilme payı [95% GA] |",
            "|---|---:|---|",
        ]
        for _, r in picks.iterrows():
            lines.append(
                f"| {CATEGORY_TR.get(r.scope, r.scope)} | {int(r.position)} | {_band(r, 'share')} |"
            )
    lines += [
        "",
        "## Sınırlılıklar",
        "",
        "- Kategori başına tek ürün seti, iki soru kalıbı ve sabit cümle metinleri: bir cümle "
        "türünün sonucu o türün bu metnine aittir.",
        "- Turnuvada cümle başına 30 görünme (kategori başına); yakın paylar arasında sıra çağrılmaz.",
        "- Dolgu cümlesi uzunluğu eşler ama bir metnin 'hiç bilgi taşımadığı' tam sağlanamaz.",
        "- Uyarı ölçüsü anahtar kelimedir; uyarının varlığını abartabilir, yokluğunu doğru gösterir.",
        "- Uydurma iddialar yalnız kurgusal markalara eklendi; ürün bunları önermez, risk olarak raporlar.",
    ]
    return "\n".join(lines) + "\n"
