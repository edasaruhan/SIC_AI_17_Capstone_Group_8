"""Reproducible 3/30 record inspection; never fabricate human annotations."""

from __future__ import annotations

import html
import io
from pathlib import Path

import pandas as pd

from modeling.features import select

from .io import digest, read_json, write_json, write_text
from .workspace import responses, verify

MATCHED_EN_QUERIES = ("vpn_01", "vpn_02", "vpn_03", "vpn_04", "vpn_08")
REVIEW_COLUMNS = ["record_id", "brand", "claim", "source_ids", "label", "reviewer", "note"]
LABELS = {"supported", "contradicted", "unverifiable", "no_claim"}


def sample_records(frames: dict[str, pd.DataFrame]) -> list[dict]:
    selected = []
    for track, category in (("en", "vpn"), ("tr", "vpn"), ("tr", "cosmetics")):
        frame = frames[track]
        frame = select(frame, (frame.category == category) & (frame.condition == "search_on"))
        if track == "en":
            frame = select(frame, frame.query_id.isin(MATCHED_EN_QUERIES))
        if frame.query_id.nunique() != 5 or frame.record_id.nunique() < 10:
            raise ValueError("Sample requires five intents and ten distinct search-on responses")
        model_counts: dict[str, int] = {}
        chosen: set[str] = set()
        for _repeat in range(2):
            for query in sorted(frame.query_id.unique()):
                candidates = select(
                    frame, (frame.query_id == query) & ~frame.record_id.isin(chosen)
                )
                rows = candidates.to_dict("records")
                rows.sort(
                    key=lambda r: (model_counts.get(r["model_id"], 0), digest([42, r["record_id"]]))
                )
                row = rows[0]
                chosen.add(row["record_id"])
                model_counts[row["model_id"]] = model_counts.get(row["model_id"], 0) + 1
                selected.append(
                    {"track": track, "category": category, "record_id": row["record_id"]}
                )
    return selected


def sample(root: Path) -> dict:
    manifest = verify(root)
    frames = {track: responses(root, track) for track in ("en", "tr")}
    selected = sample_records(frames)
    frozen = read_json(Path("configs/modeling/evidence_sample.json"))
    if [r["record_id"] for r in selected] != frozen["record_ids"]:
        raise ValueError("Computed sample differs from the committed record panel")
    payload = {
        "experiment_identity": manifest["identity"],
        "seed": 42,
        "records": selected,
        "pilot_ids": [selected[i]["record_id"] for i in (0, 10, 20)],
    }
    path = root / "review" / "sample.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("Frozen sample changed; do not overwrite an existing review")
    write_json(path, payload)
    for pilot, name in ((True, "pilot"), (False, "sample")):
        lines = [
            "# Kaynak incelemesi",
            "",
            "Kayıtlı snippet'lerdir; canlı sayfa veya modelin iç düşüncesi değildir.",
            "",
        ]
        for entry in selected:
            if pilot and entry["record_id"] not in payload["pilot_ids"]:
                continue
            row = frames[entry["track"]].set_index("record_id").loc[entry["record_id"]]
            sources = pd.read_parquet(root / f"sources_{entry['track']}.parquet")
            sources = select(sources, sources.record_id == entry["record_id"])
            lines += [
                f"## {entry['record_id']}",
                "",
                f"Model: {row['model_id']}",
                "",
                f"Soru: {row['query_text']}",
                "",
                "### Son yanıt",
                "",
                row["final_response"],
                "",
                "### Gösterilen kaynaklar",
                "",
            ]
            if sources.empty:
                lines += ["Kaydedilmiş arama sonucu yok; kaynak dayanağı doğrulanamaz.", ""]
            for source in sources.to_dict("records"):
                lines += [
                    f"- Kaynak kimliği: `{source['source_id']}`",
                    f"  Arama: {source['search_query']} | tur: {source['search_round']} | sıra: {source['position']}",
                    f"  Başlık: {source['title']}",
                    f"  URL: {source['link']}",
                    f"  Snippet: {source['snippet']}",
                    "",
                ]
        write_text(root / "review" / f"{name}.md", html.escape("\n".join(lines), quote=False))
    csv_path = root / "review" / "annotations.csv"
    if not csv_path.exists():
        annotations = pd.DataFrame(
            [
                {
                    **dict.fromkeys(REVIEW_COLUMNS, ""),
                    "record_id": r["record_id"],
                    "label": "unreviewed",
                }
                for r in selected
            ]
        )
        write_text(csv_path, annotations.to_csv(index=False))
    return payload


def checked_annotations(root: Path, *, require_complete: bool = False) -> tuple[pd.DataFrame, dict]:
    verify(root)
    selection = read_json(root / "review" / "sample.json")
    if selection["experiment_identity"] != read_json(root / "manifest.json")["identity"]:
        raise ValueError("Review belongs to another experiment")
    frame = pd.read_csv(
        io.StringIO((root / "review" / "annotations.csv").read_text()), keep_default_na=False
    )
    if list(frame.columns) != REVIEW_COLUMNS:
        raise ValueError("Annotation CSV columns changed")
    expected = {r["record_id"] for r in selection["records"]}
    if set(frame.record_id) != expected:
        raise ValueError("Annotation records must cover exactly the frozen 30-record sample")
    sources = pd.concat([pd.read_parquet(root / f"sources_{t}.parquet") for t in ("en", "tr")])
    source_records = dict(zip(sources.source_id, sources.record_id, strict=True))
    originals = pd.concat([responses(root, t) for t in ("en", "tr")]).set_index("record_id")
    completed = set()
    for row in frame.to_dict("records"):
        ids = [s.strip() for s in row["source_ids"].split(";") if s.strip()]
        if any(source_records.get(s) != row["record_id"] for s in ids):
            raise ValueError("Annotation cites a missing source or another response's source")
        label = row["label"]
        if label == "unreviewed":
            continue
        if label not in LABELS or not row["reviewer"].strip() or not row["note"].strip():
            raise ValueError("Reviewed rows need a valid label, reviewer and note")
        if label != "no_claim" and (not row["claim"].strip() or not row["brand"].strip()):
            raise ValueError("Claim annotations require claim text and brand")
        if (
            label != "no_claim"
            and row["claim"] not in originals.loc[row["record_id"], "final_response"]
        ):
            raise ValueError("Claim must be an exact excerpt from the recorded final response")
        if label in {"supported", "contradicted"} and not ids:
            raise ValueError("Supported/contradicted claims require source citations")
        completed.add(row["record_id"])
    pending = sorted(
        expected - completed | set(frame.loc[frame.label == "unreviewed", "record_id"])
    )
    status = {
        "records": len(expected),
        "reviewed": len(expected) - len(pending),
        "pending": pending,
        "complete": not pending,
        "annotation_sha256": digest(frame.to_dict("records")),
    }
    if require_complete and pending:
        raise ValueError(f"Human review incomplete: {len(pending)} records remain")
    return frame, status
