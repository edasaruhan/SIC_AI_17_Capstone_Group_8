"""Evidence-bound brand reports; no generated uplift, no live website claims."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import cast

import pandas as pd

from modeling.brands import comparison_key, load_registry
from modeling.features import select

from .io import read_json, safe_url, sha256, write_json, write_text
from .review import MATCHED_EN_QUERIES, checked_annotations
from .workspace import responses, verify

LIMITATIONS = [
    "Yalnız kayıtlı sorgular ve snippet'ler incelendi; canlı site denetimi yapılmadı.",
    "SHAP ve maskeleme asistanın iç düşüncesini veya nedensel katkı payını göstermez.",
    "Eylemler iyileştirme hipotezidir; garanti edilmiş görünürlük artışı değildir.",
    "İngilizce/Türkçe farkları model, tarih ve arama bölgesi farklarından da etkilenebilir.",
    "Tahmin modeli Google sıralamasını veya yeni markalara genellemeyi doğrulamış değildir.",
]


def visibility(frame: pd.DataFrame, brand: str) -> list[dict]:
    rows = []
    for key, part in frame.groupby(["condition", "model_id"], sort=True):
        condition, model = cast(tuple[str, str], key)
        mentions = int(part.brands_mentioned.map(lambda values: brand in values).sum())
        wins = int(part.top_recommendation.eq(brand).sum())
        rows.append(
            {
                "condition": condition,
                "model": model,
                "responses": len(part),
                "mentions": mentions,
                "single_wins": wins,
                "mention_rate": mentions / len(part),
                "single_win_rate": wins / len(part),
            }
        )
    return rows


def brand_report(root: Path, brand: str, category: str, language: str) -> dict:
    manifest = verify(root)
    registry = load_registry(category)
    canonical = registry.resolve(brand)
    frame = responses(root, language)
    frame = select(frame, frame.category == category)
    if frame.empty:
        raise ValueError("This language/domain is not in the dataset")
    evidence = pd.read_parquet(root / f"evidence_{language}.parquet")
    evidence = select(evidence, evidence.category == category)
    observed = set(evidence.brand) | {b for bs in frame.brands_mentioned for b in bs}
    payload = {
        "brand": canonical or brand,
        "category": category,
        "language": language,
        "experiment_identity": manifest["identity"],
        "limitations": LIMITATIONS,
        "scope": "historical_dataset",
        "status": "out_of_scope",
        "visibility": [],
        "queries": [],
        "sources": [],
        "actions": [],
        "model_explanations": [],
        "model_status": "not_trained",
        "human_review": "not_started",
        "training_validation": "surrogate_not_causal",
    }
    if canonical is None or canonical not in observed:
        payload["reason"] = (
            "Marka bu veri kapsamındaki yanıtlarda veya eşleşen kaynaklarda bulunamadı; skor üretilmedi."
        )
        return payload
    payload["status"] = "in_scope"
    payload["visibility"] = visibility(frame, canonical)
    payload["queries"] = [
        {
            "query_id": query,
            "query_text": part.iloc[0].query_text,
            "counts": visibility(part, canonical),
        }
        for query, part in frame.groupby("query_id", sort=True)
    ]
    own_sources = select(evidence, evidence.brand == canonical)
    payload["sources"] = own_sources.to_dict("records")
    # Compare the same five query intents; never describe this as a causal language effect.
    if category == "vpn":
        comparison = []
        for track in ("en", "tr"):
            subset = responses(root, track)
            subset = select(subset, subset.category == "vpn")
            if track == "en":
                subset = select(subset, subset.query_id.isin(MATCHED_EN_QUERIES))
            comparison.append(
                {
                    "language": track,
                    "query_count": subset.query_id.nunique(),
                    "counts": visibility(subset, canonical),
                }
            )
        payload["matched_intent_comparison"] = comparison
    annotations_path = root / "review" / "annotations.csv"
    if annotations_path.exists():
        annotations, status = checked_annotations(root)
        payload["human_review"] = status
        for row in annotations.to_dict("records"):
            if (
                row["record_id"] not in set(frame.record_id)
                or registry.resolve(row["brand"]) != canonical
                or row["label"] in {"unreviewed", "no_claim"}
            ):
                continue
            ids = [s.strip() for s in row["source_ids"].split(";") if s.strip()]
            all_sources = pd.read_parquet(root / f"sources_{language}.parquet")
            extra = select(all_sources, all_sources.source_id.isin(ids))
            known = {s["source_id"] for s in payload["sources"]}
            payload["sources"].extend(
                s for s in extra.to_dict("records") if s["source_id"] not in known
            )
            payload["actions"].append(
                {
                    "kind": "claim_audit",
                    "status": "human_reviewed_evidence",
                    "finding": row["claim"],
                    "support": row["label"],
                    "source_ids": ids,
                    "action": "İddiayı kaynak kapsamıyla birlikte sunun; çelişki veya belirsizlik varsa doğrulayın.",
                    "owner": "brand_content",
                    "uncertainty": "Eylemin görünürlük etkisi ölçülmedi.",
                    "reviewer": row["reviewer"],
                    "note": row["note"],
                }
            )
    # Only verifiable observations about retrieved snippets, not unseen websites.
    first = own_sources.drop_duplicates("link").head(3)
    for source in first.to_dict("records"):
        text = source["title"] + " " + source["snippet"]
        action_text = "Bu snippet'teki marka iddialarını doğrulayın; gerçek test veya rapor varsa tarih, yöntem ve bağlantıyla açıkça belgeleyin."
        finding = "Marka bu kayıtlı arama sonucunun başlık/snippet metninde geçiyor."
        if re.search(r"\b(best|leading|en iyi|lider)\b", text, re.IGNORECASE):
            finding += " Metinde üstünlük ifadesi eşleşti; bu otomatik bir dil işaretidir."
            action_text = "Üstünlük ifadesinin kapsamını ve karşılaştırma ölçütünü inceleyin; yalnız doğrulanabilen sonucu, geçerli olduğu koşullarla ifade edin."
        elif re.search(r"\b(audit|denetim|clinical|klinik|test)\b", text, re.IGNORECASE):
            finding += " Metinde test/denetim ifadesi eşleşti; testin varlığı henüz doğrulanmadı."
            action_text = "Test veya denetim iddiasını orijinal raporla doğrulayın; varsa tarih, yöntem ve kapsamı görünür biçimde belgeleyin."
        payload["actions"].append(
            {
                "kind": "source_audit",
                "status": "review_required",
                "finding": finding,
                "source_ids": [source["source_id"]],
                "action": action_text,
                "owner": (
                    "brand_content"
                    if source["source_type"] == "official"
                    else "third_party_outreach"
                ),
                "uncertainty": "Tam sayfa incelenmedi. Kaynak eşleşmesi iddiayı desteklediği anlamına gelmez; artış tahmin edilmedi.",
            }
        )
    state_path = root / "baselines" / f"{language}_y_top.json"
    if state_path.exists():
        state = read_json(state_path)
        payload["model_status"] = state["status"]
        if state["status"] == "completed":
            if state["identity"]["experiment"] != manifest["identity"]:
                raise ValueError("Model belongs to a different experiment")
            for name, expected in state["artifacts"].items():
                if sha256(root / "baselines" / name) != expected:
                    raise ValueError("Model artifact checksum mismatch")
            explanations = pd.read_parquet(root / "baselines" / f"shap_{language}_y_top.parquet")
            explanation_rows = select(
                explanations,
                (explanations.brand == canonical) & explanations.record_id.isin(frame.record_id),
            )
            payload["model_explanations"] = explanation_rows.to_dict("records")
            scores = pd.read_parquet(root / "baselines" / f"scores_{language}_y_top.parquet")
            selected_scores = select(
                scores, (scores.brand == canonical) & scores.record_id.isin(frame.record_id)
            )
            payload["surrogate_scores"] = selected_scores.to_dict("records")
    return payload


def render_report(payload: dict) -> str:
    def escape(value: object) -> str:
        return html.escape(str(value)).replace("|", "\\|")

    lines = [
        f"# {escape(payload['brand'])}: kaynaklı görünürlük raporu",
        "",
        f"Dil: {payload['language']} | Domain: {payload['category']} | Kapsam: kayıtlı deney",
        "",
    ]
    if payload["status"] == "out_of_scope":
        lines += [payload["reason"], ""]
    else:
        lines += [
            "## Gözlenen görünürlük",
            "",
            "Payda tüm yanıtlardır; kazananı olmayan yanıtlar çıkarılmadı.",
            "",
            "| Koşul | Model | Yanıt | Anılma | Tek kazanan |",
            "|---|---|---:|---:|---:|",
        ]
        for row in payload["visibility"]:
            lines.append(
                f"| {row['condition']} | {escape(row['model'])} | {row['responses']} | {row['mention_rate']:.1%} | {row['single_win_rate']:.1%} |"
            )
        lines += ["", "## Kaynaklı eylemler", ""]
        source_by_id = {s["source_id"]: s for s in payload["sources"]}
        if not payload["actions"]:
            lines += ["Kaynak dayanağı yetersiz; otomatik içerik tavsiyesi üretilmedi.", ""]
        for index, action in enumerate(payload["actions"], 1):
            lines += [
                f"### {index}. {action['status']}",
                "",
                escape(action["finding"]),
                "",
                escape(action["action"]),
                "",
                f"Sorumluluk: {action['owner']}",
                "",
            ]
            for source_id in action["source_ids"]:
                source = source_by_id.get(source_id)
                if source:
                    url = source["link"]
                    lines += [
                        f"- Kaynak `{source_id}`: {escape(source['title'])}",
                        f"  URL: {escape(url) if safe_url(url) else 'Geçersiz URL'}",
                        f"  Snippet: {escape(source['snippet'])}",
                    ]
            lines += ["", escape(action["uncertainty"]), ""]
        lines += [
            "## Tahmin modeli",
            "",
            f"Durum: {payload['model_status']}",
            "",
            "JSON çıktısı varsa sorgu-dışı SHAP katkılarını log-odds biriminde içerir. Bunlar asistanın gerekçeleri veya artış yüzdeleri değildir.",
            "",
        ]
    lines += ["## Sınırlar", "", *[f"- {line}" for line in payload["limitations"]], ""]
    return "\n".join(lines)


def save_report(root: Path, brand: str, category: str, language: str) -> dict:
    payload = brand_report(root, brand, category, language)
    name = f"{language}_{category}_{comparison_key(brand)}"
    if not comparison_key(brand):
        raise ValueError("Empty brand")
    output = root / "reports"
    write_json(output / f"{name}.json", payload)
    write_text(output / f"{name}.md", render_report(payload))
    return {
        "status": payload["status"],
        "markdown": str(output / f"{name}.md"),
        "json": str(output / f"{name}.json"),
    }
