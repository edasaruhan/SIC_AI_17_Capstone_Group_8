"""The cumulative model family, scored out-of-fold.

The marketing insight is in the differences between these models, not in any one
score, so they are built one feature block at a time on identical folds:

===  =========================================  =====================================
M0   brand prior only                           how much is recognition alone
M1   + retrieval position and source type       how much is structural placement
M2   + snippet language features, LightGBM      the production model, SHAP-ready
===  =========================================  =====================================

Two rule-based baselines sit underneath: recommend the brand of the top-ranked
search result, and recommend the brand that wins most often in training. A model
that cannot beat both is not measuring anything.

Priors are refitted inside every fold, so a held-out query never contributes to
the prior its own rows are scored against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    CATEGORICAL_FEATURES,
    LANGUAGE_FEATURES,
    PRIOR_FEATURES,
    STRUCTURAL_FEATURES,
    add_priors,
    finalise,
)

MODEL_BLOCKS = {
    "M0": PRIOR_FEATURES,
    "M1": PRIOR_FEATURES + STRUCTURAL_FEATURES,
    "M2": PRIOR_FEATURES + STRUCTURAL_FEATURES + LANGUAGE_FEATURES,
}
BASELINES = ("naive_position", "naive_frequency")


def _logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed, C=1.0
                ),
            ),
        ]
    )


def _boosted(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _design(frame: pd.DataFrame, columns: list[str], *, categorical: bool) -> pd.DataFrame:
    design = frame[columns].astype(float).copy()
    if categorical:
        for name in CATEGORICAL_FEATURES:
            design[name] = frame[name].astype("category")
    return design


def run_family(
    pairs: pd.DataFrame,
    fold_by_query: dict[str, int],
    *,
    target: str = "y_top",
    seed: int = 42,
) -> pd.DataFrame:
    """Fit every model on each training fold and return pooled out-of-fold scores.

    For the top-1 target only decided responses are scored, because a response
    with no single winner has no correct answer to rank toward.
    """
    frame = pairs.copy()
    if target == "y_top":
        frame = frame[frame["response_decided"] == 1].copy()
    frame["fold"] = frame["query_id"].map(fold_by_query)
    if frame["fold"].isna().any():
        missing = sorted(frame.loc[frame["fold"].isna(), "query_id"].unique())
        raise ValueError(f"Queries missing from the frozen split: {missing}")

    for name in (*BASELINES, *MODEL_BLOCKS):
        frame[f"score_{name}"] = np.nan

    for fold in sorted(frame["fold"].unique()):
        train_mask = frame["fold"] != fold
        test_mask = ~train_mask
        prepared = finalise(add_priors(frame, train_mask))
        train, test = prepared[train_mask], prepared[test_mask]

        # Baseline 1: the brand carried by the highest-ranked retrieved result.
        frame.loc[test_mask, "score_naive_position"] = (
            1.0 / test["best_position"].to_numpy()
        ) * test["in_search_results"].to_numpy()
        # Baseline 2: the brand that wins most often in the training folds.
        frame.loc[test_mask, "score_naive_frequency"] = test["prior_top_off"].to_numpy()

        for name, columns in MODEL_BLOCKS.items():
            use_categorical = name == "M2"
            x_train = _design(train, columns, categorical=use_categorical)
            x_test = _design(test, columns, categorical=use_categorical)
            y_train = train[target].to_numpy()
            if y_train.min() == y_train.max():
                frame.loc[test_mask, f"score_{name}"] = float(y_train.mean())
                continue
            estimator = _boosted(seed) if use_categorical else _logistic(seed)
            if use_categorical:
                for column in CATEGORICAL_FEATURES:
                    categories = x_train[column].cat.categories
                    x_test[column] = pd.Categorical(x_test[column], categories=categories)
            estimator.fit(x_train, y_train)
            frame.loc[test_mask, f"score_{name}"] = estimator.predict_proba(x_test)[:, 1]

    return frame


def fit_production(pairs: pd.DataFrame, *, target: str = "y_top", seed: int = 42):
    """Refit M2 on every row, for SHAP and for the report interface."""
    frame = pairs.copy()
    if target == "y_top":
        frame = frame[frame["response_decided"] == 1].copy()
    prepared = finalise(add_priors(frame, pd.Series(True, index=frame.index)))
    columns = MODEL_BLOCKS["M2"]
    design = _design(prepared, columns, categorical=True)
    model = _boosted(seed)
    model.fit(design, prepared[target].to_numpy())
    return model, design, prepared
