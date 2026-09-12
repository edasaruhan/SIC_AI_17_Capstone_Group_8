"""Same-learner ablations with nested query-group priors and held-out SHAP."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from modeling.features import (
    CATEGORICAL_FEATURES,
    LANGUAGE_FEATURES,
    PRIOR_FEATURES,
    STRUCTURAL_FEATURES,
    finalise,
    select,
)

from .io import read_json, sha256, write_json
from .metrics import summarise
from .workspace import verify

STRUCTURE = [c for c in STRUCTURAL_FEATURES if c != "n_other"] + ["n_retailer", "n_unknown"]
BLOCKS = {
    "M0": PRIOR_FEATURES,
    "M1": PRIOR_FEATURES + STRUCTURE,
    "M2": PRIOR_FEATURES + STRUCTURE + LANGUAGE_FEATURES,
}
ESTIMATOR = {
    "n_estimators": 120,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 30,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}


def priors(train: pd.DataFrame, destination: pd.DataFrame) -> pd.DataFrame:
    out = destination.copy()
    keys = ["category", "model_id", "brand"]
    for suffix in ("off", "all"):
        subset = select(train, train.condition == "search_off") if suffix == "off" else train
        for target in ("top", "mention"):
            src = select(subset, subset.response_decided == 1) if target == "top" else subset
            label = "y_" + target
            base = src.groupby(["category", "model_id"])[label].mean().to_dict()
            grouped = src.groupby(keys)[label].agg(["sum", "count"])
            rates = {
                key: (row["sum"] + 10 * base[cast(tuple, key)[:2]]) / (row["count"] + 10)
                for key, row in grouped.iterrows()
            }
            out[f"prior_{target}_{suffix}"] = [
                rates.get(key, base.get(key[:2], 0.0))
                for key in out[keys].itertuples(index=False, name=None)
            ]
    return out


def cross_fitted(frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = select(frame, frame.fold != fold).copy()
    held_out = select(frame, frame.fold == fold).copy()
    if training.fold.nunique() < 2:
        raise ValueError("Nested priors require at least three outer query folds")
    parts = []
    for inner_fold in sorted(training.fold.unique()):
        parts.append(
            priors(
                select(training, training.fold != inner_fold),
                select(training, training.fold == inner_fold),
            )
        )
    return finalise(pd.concat(parts).sort_index()), finalise(priors(training, held_out))


def design(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame({c: train[c].astype(float) for c in columns})
    right = pd.DataFrame({c: test[c].astype(float) for c in columns})
    for name in CATEGORICAL_FEATURES:
        categories = sorted(set(train[name]))
        left[name] = pd.Categorical(train[name], categories=categories)
        right[name] = pd.Categorical(test[name], categories=categories)
    return left, right


def run(
    frame: pd.DataFrame, folds: dict[str, int], target: str, *, estimator_params: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target not in {"y_top", "y_mention"}:
        raise ValueError("Unsupported target")
    frame = select(frame, frame.confidence != "low").copy().reset_index(drop=True)
    if set(frame.query_id) - set(folds):
        raise ValueError("Queries missing from frozen folds")
    frame["fold"] = frame.query_id.map(folds)
    explanations = []
    scored_parts = []
    for fold in sorted(frame.fold.unique()):
        train, test = cross_fitted(frame, fold)
        if target == "y_top":
            train, test = select(train, train.response_decided == 1), select(
                test, test.response_decided == 1
            )
        if test.empty:
            continue
        if train.empty:
            raise ValueError("No training responses for target")
        result = test[
            ["record_id", "brand", "query_id", "category", "condition", "model_id", "fold", target]
        ].copy()
        result["score_naive_frequency"] = test[
            "prior_top_off" if target == "y_top" else "prior_mention_off"
        ]
        result["score_naive_position"] = test.in_search_results / test.best_position
        for model_name, columns in BLOCKS.items():
            x_train, x_test = design(train, test, columns)
            if train[target].nunique() < 2:
                result[f"score_{model_name}"] = float(train[target].mean())
                continue
            model = LGBMClassifier(**(estimator_params or ESTIMATOR))
            model.fit(x_train, train[target].to_numpy())
            result[f"score_{model_name}"] = np.asarray(model.predict_proba(x_test))[:, 1]
            if model_name == "M2":
                contributions = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
                for idx, record in enumerate(test.to_dict("records")):
                    # Full vector retained; displayed ranks are not causal importances.
                    for j, feature in enumerate([*x_test.columns, "expected_value"]):
                        explanations.append(
                            {
                                "record_id": record["record_id"],
                                "brand": record["brand"],
                                "query_id": record["query_id"],
                                "held_out_fold": int(fold),
                                "feature": feature,
                                "contribution_log_odds": float(contributions[idx, j]),
                            }
                        )
        scored_parts.append(result)
    if not scored_parts:
        raise ValueError("No scorable responses")
    explanation_columns = [
        "record_id",
        "brand",
        "query_id",
        "held_out_fold",
        "feature",
        "contribution_log_odds",
    ]
    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(explanations).reindex(
        columns=explanation_columns
    )


def train_baselines(root: Path, tracks: list[str], targets: list[str]) -> dict:
    manifest = verify(root)
    output = root / "baselines"
    output.mkdir(exist_ok=True)
    completed = []
    for track in tracks:
        pairs = pd.read_parquet(root / f"pairs_{track}.parquet")
        folds = read_json(root / f"folds_{track}.json")
        for target in targets:
            key = f"{track}_{target}"
            state_path = output / f"{key}.json"
            identity = {
                "experiment": manifest["identity"],
                "parameters": ESTIMATOR,
                "track": track,
                "target": target,
            }
            if state_path.exists():
                state = read_json(state_path)
                if state["identity"] != identity:
                    raise ValueError("Baseline run configuration changed")
                if state["status"] == "completed":
                    for name, expected in state["artifacts"].items():
                        if sha256(output / name) != expected:
                            raise ValueError("Baseline artifact changed")
                    completed.append(key)
                    continue
            write_json(state_path, {"status": "running", "identity": identity})
            try:
                scored, explanations = run(pairs, folds, target)
                scored.to_parquet(output / f"scores_{key}.parquet", index=False)
                explanations.to_parquet(output / f"shap_{key}.parquet", index=False)
                metrics = []
                for category in sorted(scored.category.unique()):
                    part = select(scored, scored.category == category).reset_index(drop=True)
                    for model in ("naive_frequency", "naive_position", *BLOCKS):
                        metrics.append(
                            {
                                "category": category,
                                "model": model,
                                **summarise(part, "score_" + model, target),
                            }
                        )
                write_json(output / f"metrics_{key}.json", metrics)
                names = [f"scores_{key}.parquet", f"shap_{key}.parquet", f"metrics_{key}.json"]
                write_json(
                    state_path,
                    {
                        "status": "completed",
                        "identity": identity,
                        "excluded_low_confidence_pairs": int((pairs.confidence == "low").sum()),
                        "artifacts": {name: sha256(output / name) for name in names},
                    },
                )
            except Exception:
                write_json(state_path, {"status": "error", "identity": identity})
                raise
            print(f"completed baseline {key}", flush=True)
            completed.append(key)
    return {"completed": completed}
