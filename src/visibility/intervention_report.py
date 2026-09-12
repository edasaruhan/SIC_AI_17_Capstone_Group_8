"""Render the controlled recommendation test into ``reports/intervention/README.md``.

Kept apart from ``intervention.py`` on purpose: that module's source hash is part of
the experiment identity, so report wording can change without orphaning paid calls.
Reads saved receipts only; makes no API call.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from evidence_eval.io import read_json
from modeling.features import select
from visibility import intervention as iv

ARM_TR = {
    "control": "Kontrol: kayıtlı arama bağlamı",
    "neutral_p5": "+1 karşılaştırma sayfası, 5. sıra",
    "neutral_p1": "+1 karşılaştırma sayfası, 1. sıra",
    "superlative_p5": "+1 üstünlük dilli karşılaştırma sayfası, 5. sıra",
    "neutral_p5_p8": "+2 karşılaştırma sayfası (iki site), 5. ve 8. sıra",
}
METRIC_TR = {"mentioned": "Anılma", "first": "İlk anılan marka"}
SCOPE_TR = {"ALL": "Tümü", "vpn": "VPN", "hosting": "Hosting", "travel": "Seyahat"}


def points(value: Any) -> str:
    return f"{100 * float(value):+.1f}"


def share(value: Any) -> str:
    return f"%{100 * float(value):.1f}"


def completed_calls(folder: Path) -> tuple[int, int, int]:
    done = prompt = completion = 0
    for path in (folder / "steps").glob("*.json"):
        step = read_json(path)
        if step.get("status") == "completed":
            done += 1
            usage = step["result"].get("usage", {})
            prompt += int(usage.get("prompt_tokens", 0))
            completion += int(usage.get("completion_tokens", 0))
    return done, prompt, completion


def render(plan: dict, outcomes: pd.DataFrame, effects: pd.DataFrame, folder: Path) -> str:
    done, prompt, completion = completed_calls(folder)
    planned = len(iv.plan_jobs(plan["contexts"], plan["targets"], reps=plan["reps"]))
    lines = [
        "# Kontrollü öneri testi: öneriler asistanın cevabını değiştiriyor mu?",
        "",
        "Bu sayfa `make intervention-analyze` ile kaydedilmiş çağrılardan üretilir; elle "
        "düzenlemeyin. Tasarım ve gerekçe `src/visibility/intervention.py` başındadır; "
        "yorum ve sınırlılıklar [`bulgular.md`](bulgular.md) içindedir.",
        "",
        "## Tasarım",
        "",
        f"- Asistan: `{plan['model']}`, sıcaklık {plan['temperature']}, her hücre "
        f"{plan['reps']} tekrar.",
        f"- {len(plan['contexts'])} kayıtlı arama bağlamı (sektör başına 5 sorgu, İngilizce "
        "korpustan, ilk 10 sonuç).",
        "- Hedef markalar (önceden sabit kural: aramasız yanıtlarda %5–40 anılan, aramada "
        "en az görünen): "
        + "; ".join(f"{SCOPE_TR[s]}: {', '.join(t)}" for s, t in plan["targets"].items())
        + ".",
        "- Eklenen sayfa, sektörün iki liderini ve hedefi listeleyen bağımsız bir "
        "karşılaştırma sayfası; uygulamanın verdiği öneri ('rakiplerini anan karşılaştırma "
        "sayfalarında yer al') tam olarak budur. Liderler: "
        + "; ".join(f"{SCOPE_TR[s]}: {', '.join(p)}" for s, p in plan.get("peers", {}).items())
        + ".",
        "- Eklenen sayfalar ayrılmış `.example` alan adlarında; hiçbir gerçek yayın taklit "
        "edilmedi. Metin ve alan adı kollar arasında sabit.",
        "- Pilot: önce tek markalı bir inceleme sayfası denendi; 27 çağrının hiçbirinde "
        "hedef anılmadı, asistan her seferinde aynı liderleri saydı. Tam koşudan önce "
        "sayfa, verilen öneriyle uyumlu karşılaştırma biçimine getirildi. O pilot, "
        "'tek başına bir sayfa yazmak' önerisinin bu asistanda işe yaramadığını gösterir.",
        f"- Tamamlanan çağrı: {done}/{planned}. Token: {prompt:,} girdi + "
        f"{completion:,} çıktı.",
        "",
        "## Kollara göre oranlar (sorgu×marka hücreleri üzerinden)",
        "",
        "| Kol | Anılma | İlk anılan marka | Gözlem |",
        "|---|---:|---:|---:|",
    ]
    rates = outcomes.groupby("arm")[["mentioned", "first"]].mean()
    counts = {str(k): int(v) for k, v in outcomes["arm"].value_counts().items()}
    for arm in iv.ARMS:
        if arm in rates.index:
            lines.append(
                f"| {ARM_TR[arm]} | {share(rates.at[arm, 'mentioned'])} | "
                f"{share(rates.at[arm, 'first'])} | {counts[arm]} |"
            )
    lines += [
        "",
        "## Önerilerin etkisi (eşleştirilmiş fark, puan; 95% hücre-bootstrap GA)",
        "",
        "| Kapsam | Değişiklik | Ölçü | Etki | 95% GA | Hücre |",
        "|---|---|---|---:|---|---:|",
    ]
    for scope in ("ALL", *iv.SECTORS):
        for _, row in select(effects, effects["scope"] == scope).iterrows():
            lines.append(
                f"| {SCOPE_TR[scope]} | {row['contrast']} | {METRIC_TR[str(row['metric'])]} | "
                f"{points(row['effect'])} | [{points(row['effect_lo'])}, "
                f"{points(row['effect_hi'])}] | {int(row['cells'])} |"
            )
    lines += [
        "",
        "## Nasıl okunur",
        "",
        "- *Bağımsız bir sonuçta görünmek*: kontrol ile 5. sıradaki karşılaştırma sayfası "
        'farkı. "Rakiplerini anan karşılaştırma sayfalarında yer al" önerisinin testi.',
        "- *Sıra*: aynı sayfanın 1. sırada olması ile 5. sırada olması farkı.",
        "- *Dil*: aynı sayfanın üstünlük iddialı hali ile tarafsız hali farkı.",
        "- *Hacim*: ikinci bağımsız sonuç eklemenin tek sonuca göre farkı.",
        "",
        "## Sınırlılıklar",
        "",
        "- Tek asistan ve İngilizce bağlam. Sonuç başka modellere kendiliğinden genellenmez.",
        "- Sayfalar kayıtlı arama sonuçlarına eklendi. Gerçek web'de bir sayfada yer almak, "
        "aramanın o sayfayı getireceğini garanti etmez; test yalnız 'getirilirse ne olur' "
        "sorusunu yanıtlar.",
        "- Sonuç, marka adı eşleştirmesiyle ölçüldü. 'İlk anılan marka', birincil önerinin "
        "yaklaşık bir vekilidir.",
        "- `.example` alan adları modelin güvenini düşürebilir. Etkiler bu yüzden "
        "muhafazakâr (alt sınıra yakın) okunmalı.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=iv.DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=iv.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    plan = iv.build_plan(args.root)
    folder = args.output / plan["id"]
    outcomes = pd.read_csv(iv.REPORT / "outcomes.csv")
    effects = pd.read_csv(iv.REPORT / "effects.csv")
    text = render(plan, outcomes, effects, folder)
    (iv.REPORT / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {iv.REPORT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
