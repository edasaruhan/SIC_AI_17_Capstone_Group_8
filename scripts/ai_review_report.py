"""Offline AI-review sidecar. Never feeds AI labels into the human-review gate.

Kept outside the frozen v1 implementation contract so existing trained artifacts
remain verifiable. Predictions and human annotations are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from evidence_eval.io import read_json, sha256, write_json, write_text
from evidence_eval.report import brand_report, render_report
from evidence_eval.review import LABELS, REVIEW_COLUMNS
from evidence_eval.workspace import responses, verify
from modeling.brands import comparison_key, load_registry


def load_ai_annotations(root: Path, path: Path) -> tuple[list[dict], dict]:
    """Validate explicit AI provenance and same-response citations, fail closed."""
    if path.resolve() == (root / "review/annotations.csv").resolve():
        raise ValueError("AI input must not be the human annotation file")
    manifest = verify(root)
    metadata = read_json(path.with_name("manifest.json"))
    if metadata.get("experiment_identity") != manifest["identity"]:
        raise ValueError("AI review belongs to another experiment")
    if (
        metadata.get("reviewer_type") != "ai"
        or metadata.get("human_review_status") != "pending"
        or metadata.get("human_reviewed_records") != 0
        or metadata.get("scope") != "selected_claims_only"
    ):
        raise ValueError("Explicit AI-only selected-claim provenance is required")
    if metadata.get("annotation_sha256") != sha256(path):
        raise ValueError("AI CSV checksum mismatch")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REVIEW_COLUMNS:
            raise ValueError("AI CSV columns changed")
        rows = list(reader)
    selection = read_json(root / "review/sample.json")
    if selection["experiment_identity"] != manifest["identity"]:
        raise ValueError("Sample belongs to another experiment")
    expected = {r["record_id"] for r in selection["records"]}
    if set(r["record_id"] for r in rows) != expected:
        raise ValueError("AI CSV must cover exactly the frozen panel")
    if len(rows) != metadata.get("claims") or len(expected) != metadata.get("records"):
        raise ValueError("AI review counts disagree with manifest")
    originals = pd.concat([responses(root, t) for t in ("en", "tr")]).set_index("record_id")
    sources = pd.concat([pd.read_parquet(root / f"sources_{t}.parquet") for t in ("en", "tr")])
    source_records = dict(zip(sources.source_id, sources.record_id, strict=True))
    seen = set()
    for row in rows:
        if any(not isinstance(value, str) for value in row.values()):
            raise ValueError("Malformed CSV row")
        if row["label"] not in LABELS or row["label"] == "no_claim":
            raise ValueError("This selected-claim export requires claim-level labels")
        if not row["reviewer"].startswith("AI:") or not row["note"].startswith(
            "[AI-only; human_review=pending; selected_claims_only;"
        ):
            raise ValueError("Row must retain explicit AI-only provenance")
        if not row["brand"].strip() or not row["claim"].strip():
            raise ValueError("Missing brand or claim")
        visible = originals.loc[row["record_id"], "final_response"].split("</think>")[-1]
        if row["claim"] not in visible:
            raise ValueError("Claim is not an exact visible-response excerpt")
        key = (row["record_id"], row["brand"], row["claim"])
        if key in seen:
            raise ValueError("Duplicate claim")
        seen.add(key)
        ids = [value.strip() for value in row["source_ids"].split(";") if value.strip()]
        if row["label"] in {"supported", "contradicted"} and not ids:
            raise ValueError("Supported/contradicted claims need citations")
        if any(source_records.get(sid) != row["record_id"] for sid in ids):
            raise ValueError("Missing or cross-record citation")
    if dict(Counter(r["label"] for r in rows)) != {
        k: v for k, v in metadata.get("label_counts", {}).items() if v
    }:
        raise ValueError("Label counts disagree with manifest")
    return rows, metadata


def ai_brand_report(root: Path, path: Path, brand: str, category: str, language: str) -> dict:
    rows, metadata = load_ai_annotations(root, path)
    payload = brand_report(root, brand, category, language)
    payload["review_basis"] = "ai_only_exploratory_preview"
    payload["ai_review"] = {
        "scope": metadata["scope"],
        "annotation_sha256": metadata["annotation_sha256"],
        "records_in_panel": metadata["records"],
        "claims_in_panel": metadata["claims"],
        "matched_claims": 0,
        "independent_human_validation": False,
    }
    payload["limitations"] = [
        *payload["limitations"],
        "AI incelemesi yalnız seçilmiş iddiaları kapsar; insan doğrulaması veya doğruluk skoru değildir.",
        "supported kaydedilmiş snippet desteğidir; kaynağın doğruluğunu veya klinik etkinliği kanıtlamaz.",
    ]
    if payload["status"] != "in_scope":
        return payload
    registry = load_registry(category)
    canonical = registry.resolve(brand)
    frame = responses(root, language)
    record_ids = set(frame.loc[frame.category == category, "record_id"])
    all_sources = pd.read_parquet(root / f"sources_{language}.parquet").set_index("source_id")
    known = {s["source_id"] for s in payload["sources"]}
    for row in rows:
        brands = [registry.resolve(value.strip()) for value in row["brand"].split(";")]
        if row["record_id"] not in record_ids or canonical not in brands:
            continue
        ids = [value.strip() for value in row["source_ids"].split(";") if value.strip()]
        for sid in ids:
            if sid not in known:
                payload["sources"].append({"source_id": sid, **all_sources.loc[sid].to_dict()})
                known.add(sid)
        payload["actions"].append(
            {
                "kind": "claim_audit",
                "status": "ai_reviewed_evidence_not_human_validated",
                "record_id": row["record_id"],
                "claim_brand_scope": row["brand"],
                "finding": row["claim"],
                "support": row["label"],
                "source_ids": ids,
                "action": f"AI etiketi: {row['label']}. {row['note']}",
                "owner": "brand_content",
                "reviewer": row["reviewer"],
                "note": row["note"],
                "uncertainty": "Kaynak/iddia inceleme hipotezidir; görünürlük artışı tahmini değildir.",
            }
        )
        payload["ai_review"]["matched_claims"] += 1
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/evidence_v1"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--brand")
    parser.add_argument("--domain", choices=("vpn", "cosmetics"), default="vpn")
    parser.add_argument("--language", choices=("en", "tr"), default="tr")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = args.annotations or args.root / "review/ai/annotations.csv"
    if args.check:
        rows, metadata = load_ai_annotations(args.root, path)
        print(
            f"AI CSV valid: {len(rows)} claims / {metadata['records']} records. Human validation: no."
        )
        return
    if not args.brand or not comparison_key(args.brand):
        parser.error("--brand is required unless --check is used")
    payload = ai_brand_report(args.root, path, args.brand, args.domain, args.language)
    output = args.root / "reports/ai_preview"
    name = f"{args.language}_{args.domain}_{comparison_key(args.brand)}"
    write_json(output / f"{name}.json", payload)
    text = (
        "> AI incelemeli deneysel önizleme. Seçilmiş iddialar; bağımsız insan doğrulaması yok.\n\n"
        + render_report(payload)
    )
    write_text(output / f"{name}.md", text)
    print(f"{output / name}.md | {payload['ai_review']['matched_claims']} AI claims")


if __name__ == "__main__":
    main()
