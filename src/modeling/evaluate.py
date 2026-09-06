"""Metrics, with confidence intervals that respect the repetition design.

Rows are not independent: each cell repeats one query up to 30 times, so a naive
bootstrap over rows would treat 30 near-copies as 30 observations and report
intervals several times too narrow. Every interval here resamples whole
``query_id`` clusters instead.

PR-AUC is primary. Positive rates are roughly 3% for the top-1 target and 12-26%
for the mention target, so accuracy is uninformative and ROC-AUC is optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, scores))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 0.0, 1.0)
    return float(brier_score_loss(labels, clipped))


def _top1_counts(frame: pd.DataFrame, score: str, label: str) -> tuple[int, int]:
    """Hits and scorable responses, the two sums top-1 accuracy is a ratio of."""
    positives = frame.groupby("record_id", sort=False)[label].transform("sum")
    scorable = frame[positives == 1]
    if scorable.empty:
        return (0, 0)
    winners = scorable.groupby("record_id", sort=False)[score].idxmax()
    return (int(scorable.loc[winners, label].sum()), int(len(winners)))


def top1_accuracy(frame: pd.DataFrame, score: str, label: str) -> float:
    """Share of responses whose highest-scored candidate is the true winner.

    Only defined when the target names a single winner per response. The mention
    target usually marks four to six brands per answer, leaving a handful of
    single-mention responses that would otherwise be reported as a headline
    accuracy; those cases return NaN instead.
    """
    positives = frame.groupby("record_id", sort=False)[label].sum()
    if positives.empty or (positives == 1).mean() < 0.5:
        return float("nan")
    hits, total = _top1_counts(frame, score, label)
    return hits / total if total else float("nan")


def ndcg_at_k(frame: pd.DataFrame, score: str, label: str, k: int = 3) -> float:
    """Mean NDCG@k over responses, with binary relevance."""
    scores = []
    for _, group in frame.groupby("record_id", sort=False):
        relevance = group[label].to_numpy()
        if relevance.sum() == 0:
            continue
        order = np.argsort(-group[score].to_numpy(), kind="stable")
        gains = relevance[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(relevance)[::-1][:k]
        best = float((ideal * discounts[: len(ideal)]).sum())
        scores.append(float((gains * discounts).sum()) / best if best else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _percentiles(estimates: list[float], alpha: float) -> tuple[float, float]:
    if not estimates:
        return (float("nan"), float("nan"))
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


def bootstrap_pr_auc(
    frame: pd.DataFrame,
    score: str,
    label: str,
    *,
    cluster: str = "query_id",
    n_resamples: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Cluster bootstrap for PR-AUC over concatenated numpy blocks.

    Resampling the label and score arrays directly avoids rebuilding a DataFrame
    once per resample, which dominated the runtime on the English track.
    """
    rng = np.random.default_rng(seed)
    blocks = [
        (group[label].to_numpy(), group[score].to_numpy())
        for _, group in frame.groupby(cluster, sort=True)
    ]
    if len(blocks) < 2:
        return (float("nan"), float("nan"))
    estimates = []
    for _ in range(n_resamples):
        drawn = rng.choice(len(blocks), size=len(blocks), replace=True)
        labels = np.concatenate([blocks[index][0] for index in drawn])
        scores = np.concatenate([blocks[index][1] for index in drawn])
        value = pr_auc(labels, scores)
        if not np.isnan(value):
            estimates.append(value)
    return _percentiles(estimates, alpha)


def bootstrap_top1(
    frame: pd.DataFrame,
    score: str,
    label: str,
    *,
    cluster: str = "query_id",
    n_resamples: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Cluster bootstrap for top-1 accuracy.

    Top-1 accuracy is a ratio of two sums over responses, and responses never
    straddle a query cluster, so per-cluster hit and total counts are enough --
    no rescoring is needed inside the resampling loop.
    """
    rng = np.random.default_rng(seed)
    counts = np.array(
        [_top1_counts(group, score, label) for _, group in frame.groupby(cluster, sort=True)],
        dtype=float,
    )
    if len(counts) < 2:
        return (float("nan"), float("nan"))
    estimates = []
    for _ in range(n_resamples):
        drawn = rng.choice(len(counts), size=len(counts), replace=True)
        hits, total = counts[drawn].sum(axis=0)
        if total:
            estimates.append(hits / total)
    return _percentiles(estimates, alpha)


def summarise(frame: pd.DataFrame, score: str, label: str, *, with_ci: bool = True) -> dict:
    """Full metric row for one model on one evaluation frame."""
    labels = frame[label].to_numpy()
    scores = frame[score].to_numpy()
    row = {
        "n_pairs": int(len(frame)),
        "n_responses": int(frame["record_id"].nunique()),
        "positive_rate": float(labels.mean()),
        "pr_auc": pr_auc(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "top1": top1_accuracy(frame, score, label),
        "ndcg@3": ndcg_at_k(frame, score, label),
        "brier": brier(labels, scores),
    }
    if with_ci:
        row["pr_auc_lo"], row["pr_auc_hi"] = bootstrap_pr_auc(frame, score, label)
        row["top1_lo"], row["top1_hi"] = bootstrap_top1(frame, score, label)
    return row
