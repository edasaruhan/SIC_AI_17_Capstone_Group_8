"""Small on-disk fixtures for annotation gates and report provenance."""

from pathlib import Path

import pandas as pd
import pytest

from evidence_eval import report, review, workspace
from evidence_eval.evidence import associations, retrieved_sources
from evidence_eval.io import read_json, write_json, write_text


@pytest.fixture
def prepared(tmp_path):
    selection = []
    for track in ("en", "tr"):
        record_id = f"{track}_record"
        record = {
            "record_id": record_id,
            "language": track,
            "category": "vpn",
            "query_id": "vpn_01" if track == "en" else "vpn_tr_01",
            "query_text": "Best VPN?",
            "model_id": "fixture",
            "condition": "search_on",
            "run_index": 0,
            "final_response": "NordVPN has a report.",
            "confidence": "high",
            "brands_mentioned": ["NordVPN"],
            "top_recommendation": "NordVPN",
            "organic": [
                {
                    "title": "NordVPN audit",
                    "snippet": "NordVPN has a report.",
                    "link": "https://nordvpn.com/report",
                    "search_query": "VPN audit",
                    "search_round": 1,
                    "position": 1,
                }
            ],
        }
        frame = pd.DataFrame([record])
        write_json(tmp_path / f"responses_{track}.json", [record])
        sources = retrieved_sources(frame)
        sources.to_parquet(tmp_path / f"sources_{track}.parquet")
        associations(sources).to_parquet(tmp_path / f"evidence_{track}.parquet")
        selection.append({"record_id": record_id, "track": track, "category": "vpn"})
    write_json(
        tmp_path / "manifest.json",
        {
            "identity": "fixture",
            "status": "completed",
            "artifacts": {},
            "contract": {"implementation": {}},
        },
    )
    write_json(
        tmp_path / "review/sample.json", {"experiment_identity": "fixture", "records": selection}
    )
    rows = [
        {
            **dict.fromkeys(review.REVIEW_COLUMNS, ""),
            "record_id": r["record_id"],
            "label": "unreviewed",
        }
        for r in selection
    ]
    write_text(tmp_path / "review/annotations.csv", pd.DataFrame(rows).to_csv(index=False))
    return tmp_path


def annotate(
    root: Path,
    *,
    record_id="tr_record",
    label="supported",
    source_track="tr",
    claim="NordVPN has a report.",
):
    path = root / "review/annotations.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    source = pd.read_parquet(root / f"sources_{source_track}.parquet").iloc[0].source_id
    frame.loc[
        frame.record_id == record_id, ["brand", "claim", "source_ids", "label", "reviewer", "note"]
    ] = [
        "NordVPN",
        claim,
        source,
        label,
        "fixture-human",
        "Fixture annotation, not a real data review.",
    ]
    write_text(path, frame.to_csv(index=False))


def test_pending_reviews_do_not_pass_completion_gate(prepared):
    _, state = review.checked_annotations(prepared)
    assert state["reviewed"] == 0 and len(state["pending"]) == 2
    with pytest.raises(ValueError, match="incomplete"):
        review.checked_annotations(prepared, require_complete=True)


def test_review_rejects_other_responses_source(prepared):
    annotate(prepared, source_track="en")
    with pytest.raises(ValueError, match="another response"):
        review.checked_annotations(prepared)


def test_review_rejects_invented_claim(prepared):
    annotate(prepared, claim="Invented claim")
    with pytest.raises(ValueError, match="exact excerpt"):
        review.checked_annotations(prepared)


def test_report_does_not_mix_reviews_between_languages(prepared):
    annotate(prepared, record_id="en_record", source_track="en")
    payload = report.brand_report(prepared, "NordVPN", "vpn", "tr")
    assert payload["status"] == "in_scope"
    assert not any(a["status"] == "human_reviewed_evidence" for a in payload["actions"])
    assert payload["model_status"] == "not_trained"
    assert all(a["status"] == "review_required" for a in payload["actions"])
    assert all(a["source_ids"] for a in payload["actions"])
    rendered = report.render_report(payload)
    assert "https://nordvpn.com/report" in rendered and "garanti" in rendered


def test_complete_review_and_unknown_brand(prepared):
    annotate(prepared)
    annotate(prepared, record_id="en_record", source_track="en")
    _, status = review.checked_annotations(prepared, require_complete=True)
    assert status["complete"]
    payload = report.brand_report(prepared, "NordVPN", "vpn", "tr")
    assert any(a["status"] == "human_reviewed_evidence" for a in payload["actions"])
    unknown = report.brand_report(prepared, "Unseen Client", "vpn", "tr")
    assert unknown["status"] == "out_of_scope" and unknown["visibility"] == []


def test_report_without_sources_abstains(prepared):
    evidence = pd.read_parquet(prepared / "evidence_tr.parquet")
    evidence.iloc[0:0].to_parquet(prepared / "evidence_tr.parquet")
    payload = report.brand_report(prepared, "NordVPN", "vpn", "tr")
    assert payload["actions"] == []
    assert "yetersiz" in report.render_report(payload)


def test_resume_refuses_different_contract(prepared, monkeypatch):
    monkeypatch.setattr(workspace, "validate_inputs", lambda *_: {})
    monkeypatch.setattr(workspace, "contract", lambda *_: {"different": True})
    with pytest.raises(ValueError, match="changed"):
        workspace.prepare(prepared, Path("en"), Path("tr"))


def test_output_is_json_and_markdown_without_secret_config(prepared):
    result = report.save_report(prepared, "NordVPN", "vpn", "tr")
    assert Path(result["markdown"]).is_file()
    payload = read_json(Path(result["json"]))
    assert payload["experiment_identity"] == "fixture"
    assert "api_key" not in str(payload).lower()
