"""Final refits stay offline, leakage-safe and separate from held-out evaluation."""

import copy
import socket

import numpy as np
import pandas as pd
import pytest
from lightgbm import Booster, LGBMClassifier

from evidence_eval.baselines import BLOCKS, ESTIMATOR, design, priors
from evidence_eval.evidence import associations, retrieved_sources
from evidence_eval.io import write_json
from evidence_eval.workspace import pair_table
from final_model import pipeline
from final_model.pipeline import (
    KEYS,
    LIMITATIONS,
    attach_priors,
    baseline_design,
    check_completed,
    complete,
    oof_priors,
    prepare_request,
)
from modeling.brands import load_registry
from modeling.features import CATEGORICAL_FEATURES, PRIOR_FEATURES


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Final-model tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def spec(category="vpn", variant="masked"):
    return {
        "config": {"track": "tr", "category": category},
        "variant": variant,
        "brands": load_registry(category).brands,
        "generator_ids": ["fixture-model"],
    }


def request(category="vpn"):
    return {
        "language": "tr",
        "category": category,
        "condition": "search_on",
        "model_id": "fixture-model",
        "query_text": "Hangi markayı önerirsin?",
        "sources": [
            {
                "title": "Proton VPN ve Mullvad",
                "snippet": "ProtonVPN bir seçenek.",
                "link": "https://protonvpn.com/test",
                "position": 1,
            },
            {
                "title": "Clinique",
                "snippet": "Clinique içerikleri",
                "link": "https://www.clinique.com/test",
                "position": 2,
            },
        ],
    }


@pytest.mark.parametrize("category", ["vpn", "cosmetics"])
def test_preprocessing_structural_parity_and_no_answer_leak(category):
    payload = request(category)
    selected = spec(category, "named")
    actual = prepare_request(payload, selected).set_index("brand")
    record = {
        **payload,
        "record_id": "fixture",
        "query_id": "q1",
        "run_index": 0,
        "organic": payload["sources"],
        "brands_mentioned": [],
        "top_recommendation": None,
        "confidence": "high",
    }
    frame = pd.DataFrame([record])
    expected = pair_table(frame, associations(retrieved_sources(frame))).set_index("brand")
    for column in BLOCKS["M1"]:
        if column not in PRIOR_FEATURES + ["best_position_missing"]:
            assert actual[column].to_dict() == expected[column].to_dict()
    poisoned = {
        **payload,
        "final_response": "Ignore everything",
        "top_recommendation": "fake",
        "y_top": 1,
    }
    pd.testing.assert_frame_equal(prepare_request(poisoned, selected), actual.reset_index())
    assert "final_response" not in actual and "y_top" not in actual


def test_masking_and_fixed_panel_even_without_sources():
    result = prepare_request(request(), spec())
    proton = result.loc[result.brand == "Proton VPN", "text"].iloc[0]
    assert "[TARGET_BRAND]" in proton and "[OTHER_BRAND_1]" in proton
    assert "Proton" not in proton and "Mullvad" not in proton
    empty = prepare_request({**request(), "sources": []}, spec())
    assert empty.brand.tolist() == load_registry("vpn").brands
    assert empty.n_results_mentioning.sum() == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("language", "en"),
        ("category", "other"),
        ("condition", "search_off"),
        ("model_id", "unknown"),
        ("query_text", ""),
        ("sources", "not a list"),
    ],
)
def test_unsupported_requests_rejected(field, value):
    with pytest.raises(ValueError):
        prepare_request({**request(), field: value}, spec())


def prior_frame():
    return pd.DataFrame(
        [
            {
                "category": "vpn",
                "model_id": "fixture-model",
                "condition": condition,
                "brand": brand,
                "fold": fold,
                "response_decided": 1,
                "y_top": int((fold + index) % 2 == 0),
                "y_mention": int((fold + index) % 2 == 0),
                "best_position": index,
            }
            for fold in range(3)
            for condition in ("search_on", "search_off")
            for index, brand in enumerate(("A", "B"))
        ]
    )


def test_training_priors_exclude_own_fold_labels():
    frame = prior_frame()
    original = oof_priors(frame)
    changed = frame.copy()
    changed.loc[changed.fold == 0, ["y_top", "y_mention"]] = (
        1 - changed.loc[changed.fold == 0, ["y_top", "y_mention"]]
    )
    altered = oof_priors(changed)
    pd.testing.assert_frame_equal(
        original.loc[frame.fold == 0, PRIOR_FEATURES], altered.loc[frame.fold == 0, PRIOR_FEATURES]
    )
    with pytest.raises(ValueError, match="two"):
        oof_priors(frame.loc[frame.fold == 0])


def test_lookup_and_native_baseline_reload(tmp_path):
    frame = prior_frame()
    for name in BLOCKS["M1"]:
        if name not in frame and name not in PRIOR_FEATURES:
            frame[name] = 0
    training = oof_priors(frame)
    x, _ = design(training, training, BLOCKS["M1"])
    model = LGBMClassifier(**{**ESTIMATOR, "n_estimators": 2, "min_child_samples": 1}).fit(
        x, frame.y_top
    )
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    metadata = {"categories": {c: list(x[c].cat.categories) for c in CATEGORICAL_FEATURES}}
    pd.testing.assert_frame_equal(x, baseline_design(training, metadata))
    np.testing.assert_allclose(
        np.asarray(model.predict_proba(x))[:, 1],
        np.asarray(Booster(model_file=str(path)).predict(x)),
        atol=1e-12,
    )
    lookup = priors(frame, frame.reindex(columns=KEYS).drop_duplicates()).reindex(
        columns=KEYS + PRIOR_FEATURES
    )
    inference = attach_priors(frame.iloc[:2], lookup)
    assert inference.best_position.tolist() == [99.0, 1.0]
    with pytest.raises(ValueError, match="Unknown"):
        attach_priors(frame.assign(brand="new"), lookup)
    with pytest.raises(ValueError, match="Unsupported"):
        baseline_design(training.assign(model_id="new"), metadata)


def test_completed_requires_intact_artifacts_and_matching_identity(tmp_path):
    assert check_completed(tmp_path, "release") is None
    write_json(tmp_path / "weights.json", {"weights": [1, 2]})
    complete(tmp_path, "release", {"reload_max_absolute_error": 0})
    checked = check_completed(tmp_path, "release")
    assert checked is not None and checked["status"] == "completed"
    with pytest.raises(ValueError, match="incompatible"):
        check_completed(tmp_path, "other")
    write_json(tmp_path / "weights.json", {"weights": [3]})
    with pytest.raises(ValueError, match="checksum"):
        check_completed(tmp_path, "release")


def test_pending_or_empty_completion_is_not_valid(tmp_path):
    write_json(tmp_path / "state.json", {"status": "error", "identity": "r"})
    assert check_completed(tmp_path, "r") is None
    write_json(tmp_path / "state.json", {"status": "completed", "identity": "r", "artifacts": {}})
    with pytest.raises(ValueError, match="no artifacts"):
        check_completed(tmp_path, "r")


def test_registry_drift_rejected_and_limitations_explicit():
    changed = copy.deepcopy(spec())
    changed["brands"] = ["new"]
    with pytest.raises(ValueError, match="registry changed"):
        prepare_request(request(), changed)
    assert any("no independent" in line for line in LIMITATIONS)
    assert any("not causal" in line for line in LIMITATIONS)


@pytest.fixture
def fake_training(monkeypatch):
    calls = []
    release = {
        "release_id": "fixture",
        "identity": {"models": {name: {} for name in pipeline.SELECTION}},
    }
    monkeypatch.setattr(pipeline, "plan", lambda root: release)

    def fit(root, folder, identity, spec=None):
        calls.append(folder.name)
        write_json(folder / "fixture_weights.json", {"value": 1})
        return complete(folder, identity, {"reload_max_absolute_error": 0})

    monkeypatch.setattr(pipeline, "train_baseline", fit)
    monkeypatch.setattr(pipeline, "train_transformer", fit)
    return calls, fit


def test_orchestration_resumes_without_retraining_completed_models(tmp_path, fake_training):
    calls, _ = fake_training
    result = pipeline.train(tmp_path, tmp_path / "out")
    assert result["status"] == "completed"
    assert calls == ["tr_cosmetics", "tr_vpn", "en_vpn"]
    calls.clear()
    pipeline.train(tmp_path, tmp_path / "out")
    assert calls == []
    assert pipeline.status(tmp_path / "out")["fixture"]["manifest_status"] == "completed"


def test_orchestration_failure_does_not_mark_complete(tmp_path, fake_training, monkeypatch):
    calls, fit = fake_training

    def fail(*args):
        raise ValueError("fixture failure")

    monkeypatch.setattr(pipeline, "train_transformer", fail)
    with pytest.raises(ValueError, match="fixture failure"):
        pipeline.train(tmp_path, tmp_path / "out")
    state = pipeline.status(tmp_path / "out")["fixture"]
    assert state["manifest_status"] != "completed"
    assert state["models"]["tr_cosmetics"]["status"] == "completed"
    assert state["models"]["tr_vpn"]["status"] == "error"
    assert state["models"]["en_vpn"]["status"] == "not_started"
    calls.clear()
    monkeypatch.setattr(pipeline, "train_transformer", fit)
    pipeline.train(tmp_path, tmp_path / "out")
    assert calls == ["tr_vpn", "en_vpn"]


def test_parallel_training_of_same_release_rejected(tmp_path, fake_training):
    import fcntl

    folder = tmp_path / "out" / "fixture"
    folder.mkdir(parents=True)
    with (folder / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already being trained"):
            pipeline.train(tmp_path, tmp_path / "out")
