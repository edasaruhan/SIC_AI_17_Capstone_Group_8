"""Offline tests: no API, web retrieval, encoder download or human-label fabrication."""

from __future__ import annotations

import json
import socket
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import turkish_data as tr
from evidence_eval import baselines, evidence, listwise, report, review, workspace
from evidence_eval.__main__ import main
from evidence_eval.io import read_json, write_json, write_text
from modeling.brands import load_registry


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in evidence tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def corpus():
    rows = []
    for category, model, condition, query, run in product(
        tr.EXPECTED_CATEGORY_COUNTS,
        sorted(tr.EXPECTED_MODELS),
        tr.EXPECTED_CONDITION_COUNTS,
        range(1, 6),
        range(5),
    ):
        queries = yaml.safe_load(
            Path(f"configs/evaluation/queries_{category}_tr.yaml").read_text()
        )["queries"]
        row: dict[str, Any] = dict.fromkeys(tr.EXPECTED_COLUMNS, "fixture")
        row.update(
            record_id=f"{category}:{model}:{condition}:{query}:{run}",
            experiment_id=f"{category}_tr_v1",
            category=category,
            model_id=model,
            condition=condition,
            query_id=f"{category}_tr_{query:02}",
            query_text=queries[query - 1]["text"],
            run_index=run,
            language="tr",
            final_response="NordVPN has a report.",
        )
        for key in tr.JSON_COLUMNS:
            row[key] = json.dumps([] if key in ("tool_calls", "search_results") else {})
        row["core"] = json.dumps(
            {"confidence_in_extraction": "high", "top_recommendation": "NordVPN"}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_turkish_contract(corpus):
    assert tr.check(corpus)["rows"] == 300
    assert tr.REVISION != "main"


@pytest.mark.parametrize(
    "mutation",
    ["models", "language", "runs", "query", "duplicate", "json", "null", "text", "experiment"],
)
def test_turkish_rejects_bad_design(corpus, mutation):
    frame = corpus.copy()
    if mutation == "models":
        frame.loc[frame.index[2:], "model_id"] = next(iter(tr.EXPECTED_MODELS))
    elif mutation == "language":
        frame["language"] = "en"
    elif mutation == "runs":
        frame["run_index"] = 0
    elif mutation == "query":
        frame["query_id"] = "vpn_tr_01"
    elif mutation == "duplicate":
        frame.iloc[0] = frame.iloc[1]
    elif mutation == "json":
        frame.loc[0, "core"] = "[]"
    elif mutation == "null":
        frame.loc[0, "final_response"] = None
    elif mutation == "text":
        frame.loc[0, "query_text"] = "different"
    else:
        frame.loc[0, "experiment_id"] = "new_v2"
    with pytest.raises(tr.DataQualityError):
        tr.check(frame)


def test_revision_rejected_without_network():
    with pytest.raises(tr.DataQualityError):
        tr.load_frame("main")


def response_frame():
    rows = []
    for i, condition in product(range(5), ["search_off", "search_on"]):
        rows.append(
            {
                "record_id": f"r{i}_{condition}",
                "language": "tr",
                "category": "vpn",
                "query_id": f"q{i}",
                "query_text": "Best VPN?",
                "model_id": "test",
                "condition": condition,
                "run_index": 0,
                "confidence": "high",
                "brands_mentioned": ["NordVPN", "Mullvad"],
                "top_recommendation": "NordVPN" if i % 2 else "Mullvad",
                "final_response": "NordVPN has a report.",
                "organic": (
                    []
                    if condition == "search_off"
                    else [
                        {
                            "title": "NordVPN report",
                            "snippet": "NordVPN 2026 report",
                            "link": "https://nordvpn.com/report",
                            "search_query": "vpn report",
                            "search_round": 2,
                            "position": 1,
                        },
                        {
                            "title": "General privacy",
                            "snippet": "No brands here",
                            "link": "https://example.org/privacy",
                            "search_query": "privacy",
                            "search_round": 3,
                            "position": 1,
                        },
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def test_sources_keep_unmatched_results_and_exact_trace():
    sources = evidence.retrieved_sources(response_frame())
    assert len(sources) == 10
    assert (sources.matched_brands.map(len) == 0).sum() == 5
    nord = sources.iloc[0]
    assert (nord.search_query, nord.search_round, nord.position) == ("vpn report", 2, 1)
    assert sources.source_id.nunique() == len(sources)
    assert set(evidence.associations(sources).source_type) == {"official"}


def test_source_classification_uses_host_boundaries():
    rules = read_json(evidence.SOURCE_RULES)
    assert evidence.source_type("https://nordvpn.com.attacker.org", "NordVPN", rules) == "unknown"
    assert evidence.source_type("https://support.nordvpn.com", "NordVPN", rules) == "official"
    assert evidence.source_type("https://amazon.com/p", "NordVPN", rules) == "retailer"
    assert evidence.source_type("javascript:alert(1)", "NordVPN", rules) == "unknown"


def test_masking_preserves_claims_and_distinguishes_other_brands():
    registry = load_registry("vpn")
    text = "NordVPN vs Proton-VPN: 2026 audit. IVPN and ProtonVPN; supernordvpnish."
    masked = evidence.mask_text(text, "NordVPN", registry)
    assert "[TARGET_BRAND]" in masked
    assert "[OTHER_BRAND_1]" in masked and "[OTHER_BRAND_2]" in masked
    assert masked.count("[OTHER_BRAND_1]") == 2
    assert "2026 audit" in masked and "supernordvpnish" in masked
    assert "NordVPN" not in masked and "Proton" not in masked
    assert "[OTHER_BRAND_1]" in evidence.mask_text("ıvpn report", "NordVPN", registry)


def test_fixed_candidates_do_not_depend_on_test_labels():
    frame = response_frame()
    sources = evidence.associations(evidence.retrieved_sources(frame))
    pairs = workspace.pair_table(frame, sources)
    assert pairs.brand.nunique() == len(load_registry("vpn").brands)
    changed = frame.copy()
    changed["brands_mentioned"] = [[] for _ in range(len(frame))]
    changed["top_recommendation"] = None
    other = workspace.pair_table(changed, sources)
    assert pairs[["record_id", "brand"]].equals(other[["record_id", "brand"]])


def test_nested_priors_do_not_use_own_query_or_outer_test_labels():
    frame = response_frame()
    table = workspace.pair_table(frame, evidence.associations(evidence.retrieved_sources(frame)))
    table["fold"] = table.query_id.map({f"q{i}": i for i in range(5)})
    train, test = baselines.cross_fitted(table, 4)
    changed = table.copy()
    changed.loc[changed.fold.isin([0, 4]), ["y_top", "y_mention"]] = 0
    second_train, second_test = baselines.cross_fitted(changed, 4)
    cols = baselines.PRIOR_FEATURES
    pd.testing.assert_frame_equal(
        train[train.fold == 0][cols], second_train[second_train.fold == 0][cols]
    )
    only_outer = table.copy()
    only_outer.loc[only_outer.fold == 4, ["y_top", "y_mention"]] = 0
    third_train, third_test = baselines.cross_fitted(only_outer, 4)
    pd.testing.assert_frame_equal(train[cols], third_train[cols])
    pd.testing.assert_frame_equal(test[cols], third_test[cols])
    assert second_test.shape == test.shape


def test_baseline_uses_target_specific_frequency_and_held_out_shap():
    frame = response_frame()
    table = workspace.pair_table(frame, evidence.associations(evidence.retrieved_sources(frame)))
    folds = {f"q{i}": i for i in range(5)}
    params = {**baselines.ESTIMATOR, "n_estimators": 2, "min_child_samples": 1}
    scored, shap = baselines.run(table, folds, "y_mention", estimator_params=params)
    assert not scored.filter(like="score_").isna().to_numpy().any()
    assert shap.held_out_fold.eq(shap.query_id.map(folds)).all()
    frame_with_folds = table.assign(fold=table.query_id.map(folds))
    _, test = baselines.cross_fitted(frame_with_folds, 0)
    assert np.allclose(
        scored.loc[scored.fold == 0, "score_naive_frequency"], test.prior_mention_off
    )
    totals = shap.groupby(["record_id", "brand"]).contribution_log_odds.sum()
    probs = scored.set_index(["record_id", "brand"]).score_M2
    assert np.allclose(1 / (1 + np.exp(-totals)), probs.loc[totals.index])


def test_listwise_requires_exactly_one_winner_and_loss_is_permutation_invariant():
    frame = pd.DataFrame(
        {"record_id": ["a", "a"], "brand": ["x", "y"], "query_id": ["q", "q"], "y_top": [1, 0]}
    )
    listwise.validate_groups(frame)
    assert listwise.reference_loss([1, 2, 3], 2) == pytest.approx(
        listwise.reference_loss([3, 1, 2], 0)
    )
    frame.y_top = 0
    with pytest.raises(ValueError):
        listwise.validate_groups(frame)


def test_sample_is_fixed_and_balanced():
    frames = {}
    for track in ("en", "tr"):
        rows = []
        for category in ("vpn", "cosmetics"):
            queries = (
                review.MATCHED_EN_QUERIES
                if track == "en"
                else [f"{category}_tr_{i:02}" for i in range(1, 6)]
            )
            for query, model, run in product(queries, ["a", "b", "c"], range(2)):
                rows.append(
                    {
                        "record_id": f"{track}:{category}:{query}:{model}:{run}",
                        "query_id": query,
                        "model_id": model,
                        "category": category,
                        "condition": "search_on",
                    }
                )
        frames[track] = pd.DataFrame(rows)
    first = review.sample_records(frames)
    second = review.sample_records({k: f.sample(frac=1, random_state=1) for k, f in frames.items()})
    assert first == second and len(first) == 30
    assert len({r["record_id"] for r in first}) == 30


def test_safe_finite_json_and_report_no_made_up_uplift(tmp_path):
    write_json(tmp_path / "x.json", {"n": float("nan")})
    assert read_json(tmp_path / "x.json") == {"n": None}
    payload = {
        "brand": "<script>bad</script>",
        "category": "vpn",
        "language": "tr",
        "status": "out_of_scope",
        "reason": "No data; no score.",
        "limitations": report.LIMITATIONS,
    }
    text = report.render_report(payload)
    assert "<script>" not in text and "+14" not in text and "%68" not in text


def test_manifest_rejects_artifact_and_code_drift(tmp_path):
    write_text(tmp_path / "a.txt", "ok")
    from evidence_eval.io import sha256

    write_json(
        tmp_path / "manifest.json",
        {
            "status": "completed",
            "artifacts": {"a.txt": sha256(tmp_path / "a.txt")},
            "contract": {"implementation": {}},
        },
    )
    workspace.verify(tmp_path)
    write_text(tmp_path / "a.txt", "changed")
    with pytest.raises(ValueError, match="artifact"):
        workspace.verify(tmp_path)


def test_cli_missing_files_fails_without_network(tmp_path):
    assert main(["--root", str(tmp_path), "sample"]) == 1
