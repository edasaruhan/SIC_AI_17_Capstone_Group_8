"""Optional, read-only bridge to historical data and frozen pilot rankers."""

from pathlib import Path

from evidence_eval.io import read_json


def context(config: dict, answers: list[dict], root: Path, releases: Path) -> dict:
    if config["mode"] != "live":
        return {"status": "skipped_synthetic_fixture"}
    if not config["domain"]:
        return {
            "status": "unsupported_sector",
            "note": "Genel demo çalışır; bu sektör için eğitilmiş yerel model yok.",
        }
    if not (root / "manifest.json").exists():
        return {"status": "local_dataset_unavailable"}
    from evidence_eval.report import visibility
    from evidence_eval.workspace import responses, verify
    from modeling.brands import load_registry

    verify(root)
    registry = load_registry(config["domain"])
    canonical = registry.resolve(config["brand"])
    if not canonical:
        return {
            "status": "brand_outside_frozen_registry",
            "note": "Bu marka için pilot skor uydurulmadı.",
        }
    frame = responses(root, "tr")
    result = {
        "status": "historical_only",
        "historical_visibility_not_live": visibility(
            frame.loc[frame.category == config["domain"]].copy(), canonical
        ),
    }
    available = [(read_json(p), p.parent) for p in releases.glob("*/manifest.json")]
    available = [
        (m, p)
        for m, p in available
        if m.get("status") == "completed"
        and "tr_" + config["domain"] in m.get("identity", {}).get("models", {})
    ]
    if not available:
        result["pilot_status"] = "no_completed_release"
        return result
    _, release = max(available, key=lambda pair: pair[0].get("finished_at", ""))
    from final_model.pipeline import predict

    first = next(a for a in answers if a["condition"] == "search_on")
    prediction = predict(
        root,
        release,
        {
            "language": "tr",
            "category": config["domain"],
            "condition": "search_on",
            "model_id": config["model"],
            "query_text": first["query"],
            "sources": first["sources"],
        },
        device="cpu",
    )
    rank = next(i for i, r in enumerate(prediction["ranking"], 1) if r["brand"] == canonical)
    result["pilot"] = {
        "release_id": prediction["release_id"],
        "query": first["query"],
        "brand": canonical,
        "rank": rank,
        "candidates": len(prediction["ranking"]),
        "score": prediction["ranking"][rank - 1]["score"],
        "score_semantics": prediction["score_semantics"],
        "note": "Deneysel ilk-sorgu tahmini; gerçek gözlenen anılma veya görünürlük artışı değildir.",
    }
    result["status"] = "historical_and_pilot"
    return result
