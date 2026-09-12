"""Offline integration audit: input parity and saved-model inference, not test accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_eval.baselines import BLOCKS
from evidence_eval.io import read_json, sha256, write_json
from evidence_eval.listwise import inputs
from evidence_eval.workspace import responses, verify
from final_model.pipeline import check_completed, predict, prepare_request
from modeling.features import PRIOR_FEATURES


def audit(root: Path, release: Path, *, preprocessing_only: bool = False) -> dict:
    frozen = verify(root)
    manifest = read_json(release / "manifest.json")
    if manifest["identity"]["experiment"] != frozen["identity"]:
        raise ValueError("Frozen experiment does not match release")
    result = {
        "scope": "training-input parity and technical inference only; not independent evaluation",
        "release_id": manifest["release_id"],
        "models": {},
    }
    for name, spec in manifest["identity"]["models"].items():
        config = spec["config"]
        if spec["family"] == "M3":
            selected, _ = inputs(
                root, config["track"], config["category"], spec["variant"] == "masked"
            )
        else:
            selected = pd.read_parquet(root / "pairs_tr.parquet")
            selected = selected.loc[
                (selected.category == "cosmetics")
                & (selected.condition == "search_on")
                & (selected.confidence != "low")
                & (selected.response_decided == 1)
            ].copy()
        ids = sorted(selected.record_id.unique())[:3]
        records = responses(root, config["track"]).set_index("record_id")
        checks = []
        requests = []
        for record_id in ids:
            row = records.loc[record_id]
            request = {
                "language": config["track"],
                "category": config["category"],
                "model_id": row.model_id,
                "condition": "search_on",
                "query_text": row.query_text,
                "sources": row.organic,
            }
            requests.append(request)
            actual = prepare_request(request, spec).set_index("brand").sort_index()
            expected = selected.loc[selected.record_id == record_id].set_index("brand").sort_index()
            if spec["family"] == "M3":
                pd.testing.assert_series_equal(actual.text, expected.text)
            else:
                columns = [
                    c for c in BLOCKS["M1"] if c not in PRIOR_FEATURES + ["best_position_missing"]
                ]
                pd.testing.assert_frame_equal(
                    actual.reindex(columns=columns),
                    expected.reindex(columns=columns),
                    check_dtype=False,
                )
            checks.append(
                {
                    "record_id": record_id,
                    "candidates": len(actual),
                    "preprocessing_exact_match": True,
                }
            )
        result["models"][name] = {"checks": checks}
        if not requests:
            raise ValueError(f"No eligible inference samples for {name}")
        if not preprocessing_only:
            state = check_completed(release / name, manifest["release_id"])
            if state is None:
                raise ValueError(f"Model incomplete: {name}")
            # The last parity-checked historical request is a technical example only.
            prediction = predict(root, release, requests[-1], device="cpu")
            scores = np.asarray([r["score"] for r in prediction["ranking"]])
            if not np.isfinite(scores).all() or len(scores) != len(spec["brands"]):
                raise ValueError("Invalid saved-model scores")
            if spec["family"] == "M3" and not np.isclose(scores.sum(), 1.0, atol=1e-5):
                raise ValueError("Listwise scores are not normalised")
            destination = release / "validation"
            write_json(destination / f"{name}_request.json", requests[-1])
            write_json(destination / f"{name}_prediction.json", prediction)
            result["models"][name]["saved_model_inference"] = "passed_on_cpu"
            result["models"][name]["state_sha256"] = sha256(release / name / "state.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/evidence_v1"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--preprocessing-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, args.release, preprocessing_only=args.preprocessing_only)
    if not args.preprocessing_only:
        result["script_sha256"] = sha256(Path(__file__))
        write_json(args.release / "validation" / "integration.json", result)
    print(result)


if __name__ == "__main__":
    main()
