"""Unseen-domain transfer and the signal stability matrix.

Query-level folds keep a query out of training, but the model still trains on the
sector it is then scored in. Two questions stay open under that design:

* Does a model that never saw a sector rank that sector's brands?
  (leave-one-domain-out, "M2-General")
* Which signals point the same way in every sector, and which flip?
  (signal stability matrix)

Rules fixed before looking at any result:

* **No brand priors and no category.** Priors encode brand identity; in an unseen
  sector they are undefined, and in a seen one they dominate SHAP. ``category``
  has no level for a sector the model never saw. What remains is retrieval
  position, source mix, snippet language and the generator model.
* **Retrieval-on responses only.** With retrieval off every retrieval feature is
  zero, so a prior-free model sees identical candidates and can only tie.
* **Same learner as M2** (``baselines.ESTIMATOR``); only features and training
  domains change, so a score difference is not a tuning difference.
* **Ties are broken at random with one fixed noise vector shared by every model.**
  Otherwise top-1 rewards whichever tied brand comes first in the table, which
  for the position baseline is most responses.
* **Signal effects are within-response.** A signal's effect is the rank-biserial
  correlation between winners and non-winners *of the same answer*, so candidate
  pool size and sector base rates cancel. Content signals are compared only among
  retrieved candidates; otherwise every content feature would just re-measure
  "was retrieved at all".
* **A direction is only called with at least ``MIN_QUERY_GROUPS`` independent
  queries**; with fewer, the cluster bootstrap has nothing to resample.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from evidence_eval.baselines import ESTIMATOR, STRUCTURE
from modeling import evaluate
from modeling.features import LANGUAGE_FEATURES, PRIOR_FEATURES, finalise, select

TARGETS = ("y_top", "y_mention")
GENERAL_FEATURES = STRUCTURE + LANGUAGE_FEATURES
EXCLUDED_FEATURES = [*PRIOR_FEATURES, "category", "condition"]
PRESENCE = "in_search_results"
# Content counts and "has a year" rise with the number of supporting results, so a
# content effect is re-checked among candidates with the same result count.
VOLUME = "n_results_mentioning"
# best_position_missing is exactly 1 - in_search_results; one of the two is enough.
MATRIX_FEATURES = [f for f in GENERAL_FEATURES if f != "best_position_missing"]
MATCHABLE_FEATURES = [f for f in MATRIX_FEATURES if f not in {PRESENCE, VOLUME}]
LOWER_IS_BETTER = frozenset({"best_position"})
MIN_QUERY_GROUPS = 5
TIE_NOISE = 1e-9

# Response-relative inputs. A raw count ("in 7 results") means different things in a
# sector whose answers retrieve 10 results and one that retrieves 40; its share, its
# rank against the other candidates of the same answer and the source mix do not.
SOURCE_KINDS = ("official", "editorial", "affiliate", "forum", "retailer", "unknown")
RELATIVE_FEATURES = [
    "volume_share",
    "volume_vs_leader",
    "volume_rank_pct",
    "is_top_retrieved",
    "position_rank_pct",
    *[f"share_{kind}" for kind in SOURCE_KINDS],
    *[f"{feature}_pct" for feature in LANGUAGE_FEATURES],
]
INVARIANT_FEATURES = [PRESENCE, *RELATIVE_FEATURES]


def _rows(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    return cast(pd.DataFrame, frame.iloc[np.flatnonzero(mask)])


def panel(pairs: pd.DataFrame, target: str) -> pd.DataFrame:
    """Retrieval-on, non-low-confidence pairs; decided responses for ``y_top``."""
    if target not in TARGETS:
        raise ValueError(f"Unsupported target {target!r}")
    keep = (pairs["condition"] == "search_on") & (pairs["confidence"] != "low")
    if target == "y_top":
        keep &= pairs["response_decided"] == 1
    return finalise(select(pairs, keep)).reset_index(drop=True)


def add_relative(frame: pd.DataFrame, results_per_response: pd.Series) -> pd.DataFrame:
    """Attach ``RELATIVE_FEATURES``, each computed within one answer's candidates.

    ``results_per_response`` maps ``record_id`` to the number of retrieved results
    in that answer. Content percentiles rank only retrieved candidates; a candidate
    absent from retrieval gets 0, the same as "no evidence".
    """
    out = frame.copy()
    record = out["record_id"]
    volume = out[VOLUME].astype(float)
    counts = {str(k): float(v) for k, v in results_per_response.items()}
    total = pd.Series([counts.get(str(r), 0.0) for r in record], index=out.index)
    out["volume_share"] = np.where(total > 0, volume / total.clip(lower=1.0), 0.0)
    leader = volume.groupby(record).transform("max")
    out["volume_vs_leader"] = np.where(leader > 0, volume / leader.clip(lower=1.0), 0.0)
    out["volume_rank_pct"] = volume.groupby(record).rank(pct=True, method="average")
    retrieved = out[PRESENCE] == 1
    best = out["best_position"].astype(float).where(retrieved)
    out["is_top_retrieved"] = (retrieved & (best == best.groupby(record).transform("min"))).astype(
        float
    )
    out["position_rank_pct"] = (-best).groupby(record).rank(pct=True).fillna(0.0)
    for kind in SOURCE_KINDS:
        out[f"share_{kind}"] = np.where(volume > 0, out[f"n_{kind}"] / volume.clip(lower=1.0), 0.0)
    for feature in LANGUAGE_FEATURES:
        ranked = out[feature].astype(float).where(retrieved).groupby(record).rank(pct=True)
        out[f"{feature}_pct"] = ranked.fillna(0.0)
    return out


def design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    with_model: bool,
    features: list[str] = GENERAL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame({c: train[c].astype(float) for c in features})
    right = pd.DataFrame({c: test[c].astype(float) for c in features})
    if with_model:
        levels = sorted(set(train["model_id"]))
        left["model_id"] = pd.Categorical(train["model_id"], categories=levels)
        right["model_id"] = pd.Categorical(test["model_id"], categories=levels)
    return left, right


def fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    with_model: bool,
    features: list[str] = GENERAL_FEATURES,
) -> tuple[LGBMClassifier | None, pd.DataFrame, np.ndarray]:
    x_train, x_test = design(train, test, with_model=with_model, features=features)
    labels = train[target].to_numpy()
    if len(np.unique(labels)) < 2:
        return None, x_test, np.full(len(test), float(labels.mean()))
    model = LGBMClassifier(**ESTIMATOR)
    model.fit(x_train, labels)
    return model, x_test, np.asarray(model.predict_proba(x_test))[:, 1]


def leave_one_domain_out(frame: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(held_out_domain, test_mask)``; training is always ``~test_mask``."""
    categories = frame["category"].to_numpy()
    if len(set(categories)) < 2:
        raise ValueError("Leave-one-domain-out needs at least two domains")
    for domain in sorted(set(categories)):
        yield str(domain), categories == domain


def seen_domain_scores(
    frame: pd.DataFrame,
    folds: dict[str, int],
    target: str,
    *,
    with_model: bool = True,
    features: list[str] = GENERAL_FEATURES,
) -> np.ndarray:
    """Query-fold out-of-fold scores: the held-out query's sector stays in training."""
    missing = set(frame["query_id"].astype(str)) - set(folds)
    if missing:
        raise ValueError(f"Queries missing from frozen folds: {sorted(missing)}")
    fold = np.array([folds[query] for query in frame["query_id"].astype(str)])
    scores = np.full(len(frame), np.nan)
    for k in sorted(set(fold)):
        test_mask = fold == k
        _, _, scores[test_mask] = fit_score(
            _rows(frame, ~test_mask),
            _rows(frame, test_mask),
            target,
            with_model=with_model,
            features=features,
        )
    return scores


def permutation_drop(
    model: LGBMClassifier | None,
    x_test: pd.DataFrame,
    labels: np.ndarray,
    *,
    repeats: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """PR-AUC lost on the held-out domain when one input is shuffled."""
    if model is None or labels.min() == labels.max():
        return {}
    rng = np.random.default_rng(seed)
    base = evaluate.pr_auc(labels, np.asarray(model.predict_proba(x_test))[:, 1])
    drops: dict[str, float] = {}
    for column in x_test.columns:
        losses = []
        for _ in range(repeats):
            shuffled = x_test.copy()
            values = shuffled[column].array
            shuffled[column] = values.take(rng.permutation(len(values)))
            permuted = np.asarray(model.predict_proba(shuffled))[:, 1]
            losses.append(base - evaluate.pr_auc(labels, permuted))
        drops[str(column)] = float(np.mean(losses))
    return drops


def unseen_domain_scores(
    frame: pd.DataFrame,
    target: str,
    *,
    with_model: bool = True,
    features: list[str] = GENERAL_FEATURES,
) -> tuple[np.ndarray, list[dict]]:
    """Leave-one-domain-out scores plus the held-out permutation importances."""
    scores = np.full(len(frame), np.nan)
    importances: list[dict] = []
    for domain, test_mask in leave_one_domain_out(frame):
        train, test = _rows(frame, ~test_mask), _rows(frame, test_mask)
        if domain in set(train["category"]):
            raise AssertionError(f"{domain} leaked into its own training set")
        model, x_test, scores[test_mask] = fit_score(
            train, test, target, with_model=with_model, features=features
        )
        for feature, drop in permutation_drop(model, x_test, test[target].to_numpy()).items():
            importances.append({"domain": domain, "feature": feature, "pr_auc_drop": drop})
    return scores, importances


def tie_break(frame: pd.DataFrame, columns: list[str], *, seed: int = 42) -> None:
    noise = np.random.default_rng(seed).uniform(0.0, TIE_NOISE, len(frame))
    for column in columns:
        frame[column] = frame[column].to_numpy(dtype=float) + noise


def _top1(group: pd.DataFrame, score: str, label: str) -> tuple[float, float]:
    positives = group.groupby("record_id", sort=False)[label].transform("sum")
    scorable = select(group, positives == 1)
    if scorable.empty:
        return (0.0, 0.0)
    winners = scorable.groupby("record_id", sort=False)[score].idxmax()
    return (float(scorable.loc[winners, label].sum()), float(len(winners)))


def _ndcg(group: pd.DataFrame, score: str, label: str) -> tuple[float, float]:
    values = [evaluate.ndcg_at_k(r, score, label) for _, r in group.groupby("record_id")]
    finite = [v for v in values if np.isfinite(v)]
    return (float(sum(finite)), float(len(finite)))


def paired_delta(
    frame: pd.DataFrame,
    a: str,
    b: str,
    label: str,
    *,
    n_resamples: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """``a - b`` on PR-AUC, top-1 and NDCG@3, with one shared query-cluster bootstrap.

    Both scores are resampled with the same queries, so the interval is for the
    difference itself rather than two overlapping marginal intervals.
    """
    blocks = []
    for _, query in frame.groupby("query_id", sort=True):
        blocks.append(
            (
                query[label].to_numpy(),
                query[a].to_numpy(),
                query[b].to_numpy(),
                np.array(_top1(query, a, label) + _top1(query, b, label)),
                np.array(_ndcg(query, a, label) + _ndcg(query, b, label)),
            )
        )

    def measure(chosen: list[int]) -> tuple[float, float, float]:
        labels = np.concatenate([blocks[i][0] for i in chosen])
        pr = evaluate.pr_auc(labels, np.concatenate([blocks[i][1] for i in chosen]))
        pr -= evaluate.pr_auc(labels, np.concatenate([blocks[i][2] for i in chosen]))
        tops = np.sum([blocks[i][3] for i in chosen], axis=0)
        top = (tops[0] - tops[2]) / tops[1] if label == "y_top" and tops[1] else float("nan")
        ndcgs = np.sum([blocks[i][4] for i in chosen], axis=0)
        ndcg = ndcgs[0] / ndcgs[1] - ndcgs[2] / ndcgs[3] if ndcgs[1] else float("nan")
        return (pr, top, ndcg)

    point = measure(list(range(len(blocks))))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            measure(list(rng.integers(0, len(blocks), len(blocks))))
            for _ in range(n_resamples if len(blocks) > 1 else 0)
        ]
    ).reshape(-1, 3)
    row: dict[str, float] = {"independent_query_groups": float(len(blocks))}
    for j, name in enumerate(("pr_auc", "top1", "ndcg@3")):
        finite = draws[:, j][np.isfinite(draws[:, j])] if len(draws) else np.array([])
        low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        row[f"delta_{name}"] = float(point[j])
        row[f"delta_{name}_lo"], row[f"delta_{name}_hi"] = float(low), float(high)
    return row


def _signed(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = frame[feature].to_numpy(dtype=float)
    return -values if feature in LOWER_IS_BETTER else values


def comparable_rows(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Presence is judged on every candidate, content only among retrieved ones."""
    return frame if feature == PRESENCE else select(frame, frame[PRESENCE] == 1)


def concordance_blocks(
    frame: pd.DataFrame, feature: str, label: str, *, matched_on: str | None = None
) -> np.ndarray:
    """Per-query ``(concordant, discordant, compared)`` winner-vs-loser pair counts.

    With ``matched_on``, only pairs sharing that column's value are compared, e.g.
    a winner and a loser that both appear in exactly three results.
    """
    rows = comparable_rows(frame, feature)
    values, labels = _signed(rows, feature), rows[label].to_numpy()
    strata = rows[matched_on].to_numpy(dtype=float) if matched_on else np.zeros(len(rows))
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3))
    for key, positions in rows.groupby(["query_id", "record_id"]).indices.items():
        won = labels[positions] == 1
        if won.all() or not won.any():
            continue
        diff = values[positions][won][:, None] - values[positions][~won][None, :]
        same = strata[positions][won][:, None] == strata[positions][~won][None, :]
        sums[str(cast(tuple, key)[0])] += (
            ((diff > 0) & same).sum(),
            ((diff < 0) & same).sum(),
            same.sum(),
        )
    blocks = np.array(list(sums.values())).reshape(-1, 3)
    return blocks[blocks[:, 2] > 0]


def signal_effect(
    frame: pd.DataFrame,
    feature: str,
    label: str,
    *,
    matched_on: str | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Within-response rank-biserial effect with a query-cluster bootstrap interval.

    +1: the winner always has the higher value; -1: always the lower; 0: no order.
    For ``best_position`` the sign is flipped, so + means "ranked higher wins".
    """
    blocks = concordance_blocks(frame, feature, label, matched_on=matched_on)
    row = {"feature": feature, "n_queries": int(len(blocks)), "n_comparisons": 0}
    if not len(blocks) or blocks[:, 2].sum() == 0:
        return {
            **row,
            "effect": np.nan,
            "effect_lo": np.nan,
            "effect_hi": np.nan,
            "direction": "n/a",
        }
    totals = blocks.sum(axis=0)
    effect = float((totals[0] - totals[1]) / totals[2])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        drawn = blocks[rng.integers(0, len(blocks), len(blocks))].sum(axis=0)
        draws.append((drawn[0] - drawn[1]) / drawn[2])
    low, high = np.percentile(draws, [2.5, 97.5])
    if len(blocks) < MIN_QUERY_GROUPS:
        direction = "n/a"
    elif low > 0:
        direction = "up"
    elif high < 0:
        direction = "down"
    else:
        direction = "flat"
    return {
        **row,
        "n_comparisons": int(totals[2]),
        "effect": effect,
        "effect_lo": float(low),
        "effect_hi": float(high),
        "direction": direction,
    }


def shap_profile(frame: pd.DataFrame, target: str, *, with_model: bool = True) -> pd.DataFrame:
    """In-domain M2-General SHAP: importance share and value-to-contribution direction.

    Fitted on every row of one domain. Descriptive only; the out-of-domain
    evidence is the leave-one-domain-out permutation drop.
    """
    model, x, _ = fit_score(frame, frame, target, with_model=with_model)
    if model is None:
        return pd.DataFrame({"feature": [], "shap_share": [], "shap_direction": []})
    contributions = np.asarray(model.booster_.predict(x, pred_contrib=True))[:, :-1]
    magnitude = np.abs(contributions).mean(axis=0)
    share = magnitude / magnitude.sum() if magnitude.sum() else magnitude
    rows = []
    for j, feature in enumerate(x.columns):
        direction = np.nan
        if feature in MATRIX_FEATURES:
            mask = comparable_rows(frame, str(feature)).index.to_numpy()
            values = pd.Series(_signed(frame, str(feature))[mask])
            contribution = pd.Series(contributions[mask, j])
            if values.nunique() > 1 and contribution.nunique() > 1:
                direction = float(values.corr(contribution, method="spearman"))
        rows.append(
            {"feature": str(feature), "shap_share": float(share[j]), "shap_direction": direction}
        )
    return pd.DataFrame(rows)


def classify(directions: list[str]) -> str:
    """Stability class from per-domain directions; rule fixed before any result."""
    called = [d for d in directions if d != "n/a"]
    up, down = called.count("up"), called.count("down")
    if up and down:
        return "conflicting"
    agree = max(up, down)
    if agree >= 4 and agree >= 2 * len(called) / 3:
        return "strong"
    if agree >= 2:
        return "moderate"
    return "weak"
