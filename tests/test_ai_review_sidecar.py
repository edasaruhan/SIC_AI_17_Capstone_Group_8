"""AI proposals remain separate from human annotations and frozen training."""

import csv
import io
import runpy
from collections import Counter

import pandas as pd
import pytest

from evidence_eval.io import read_json, sha256, write_json, write_text
from evidence_eval.review import REVIEW_COLUMNS, checked_annotations
from test_evidence_review_report import prepared as prepared

SIDECAR = runpy.run_path("scripts/ai_review_report.py")
load_ai_annotations = SIDECAR["load_ai_annotations"]
ai_brand_report = SIDECAR["ai_brand_report"]


def write_fixture(path, rows, **overrides):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, stream.getvalue())
    write_json(
        path.with_name("manifest.json"),
        {
            "experiment_identity": "fixture",
            "reviewer_type": "ai",
            "human_review_status": "pending",
            "human_reviewed_records": 0,
            "scope": "selected_claims_only",
            "annotation_sha256": sha256(path),
            "records": 2,
            "claims": len(rows),
            "label_counts": dict(Counter(r["label"] for r in rows)),
            **overrides,
        },
    )


@pytest.fixture
def ai_input(prepared):
    rows = []
    for track in ("en", "tr"):
        source = pd.read_parquet(prepared / f"sources_{track}.parquet").iloc[0]
        rows.append(
            dict(
                zip(
                    REVIEW_COLUMNS,
                    [
                        f"{track}_record",
                        "NordVPN",
                        "NordVPN has a report.",
                        source.source_id,
                        "supported",
                        "AI:fixture",
                        "[AI-only; human_review=pending; selected_claims_only; claim_id=fixture] Synthetic test only.",
                    ],
                    strict=True,
                )
            )
        )
    path = prepared / "review/ai/annotations.csv"
    write_fixture(path, rows)
    return prepared, path, rows


def test_ai_report_does_not_complete_human_review(ai_input):
    root, path, _ = ai_input
    human = root / "review/annotations.csv"
    before = human.read_bytes()
    payload = ai_brand_report(root, path, "NordVPN", "vpn", "tr")
    assert payload["human_review"]["reviewed"] == 0
    assert payload["ai_review"]["matched_claims"] == 1  # EN evidence is excluded.
    assert payload["review_basis"] == "ai_only_exploratory_preview"
    audits = [a for a in payload["actions"] if a["kind"] == "claim_audit"]
    assert len(audits) == 1
    assert audits[0]["status"] == "ai_reviewed_evidence_not_human_validated"
    assert human.read_bytes() == before
    with pytest.raises(ValueError, match="incomplete"):
        checked_annotations(root, require_complete=True)


def test_checksum_and_experiment_binding(ai_input):
    root, path, rows = ai_input
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_ai_annotations(root, path)
    write_fixture(path, rows, experiment_identity="wrong")
    with pytest.raises(ValueError, match="another experiment"):
        load_ai_annotations(root, path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("reviewer", "human-name", "AI-only"),
        ("note", "approved", "AI-only"),
        ("claim", "Invented quote", "exact visible"),
        ("brand", "", "Missing brand"),
        ("label", "no_claim", "claim-level"),
        ("source_ids", "", "need citations"),
    ],
)
def test_invalid_claims_fail_closed(ai_input, field, value, error):
    root, path, rows = ai_input
    rows[0][field] = value
    write_fixture(path, rows)
    with pytest.raises(ValueError, match=error):
        load_ai_annotations(root, path)


def test_cross_record_sources_rejected(ai_input):
    root, path, rows = ai_input
    rows[0]["source_ids"] = rows[1]["source_ids"]
    write_fixture(path, rows)
    with pytest.raises(ValueError, match="cross-record"):
        load_ai_annotations(root, path)


def test_duplicate_and_missing_panel_rejected(ai_input):
    root, path, rows = ai_input
    write_fixture(path, [*rows, rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_ai_annotations(root, path)
    write_fixture(path, rows[:1])
    with pytest.raises(ValueError, match="frozen panel"):
        load_ai_annotations(root, path)


def test_manifest_cannot_claim_human_validation(ai_input):
    root, path, rows = ai_input
    write_fixture(path, rows, human_reviewed_records=2)
    with pytest.raises(ValueError, match="AI-only"):
        load_ai_annotations(root, path)
    with pytest.raises(ValueError, match="human annotation file"):
        load_ai_annotations(root, root / "review/annotations.csv")


def test_unverifiable_without_sources_and_unknown_brand(ai_input):
    root, path, rows = ai_input
    rows[0].update(label="unverifiable", source_ids="")
    write_fixture(path, rows)
    actual, _ = load_ai_annotations(root, path)
    assert actual[0]["label"] == "unverifiable"
    payload = ai_brand_report(root, path, "Unknown Brand", "vpn", "tr")
    assert payload["status"] == "out_of_scope"
    assert payload["ai_review"]["matched_claims"] == 0
    assert read_json(path.with_name("manifest.json"))["human_reviewed_records"] == 0
