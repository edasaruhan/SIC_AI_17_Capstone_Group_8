"""Query-cluster uncertainty for the v1 ranking evaluation."""

import numpy as np
import pandas as pd

from modeling import evaluate


def summarise(frame: pd.DataFrame, score: str, label: str) -> dict:
    result = evaluate.summarise(frame, score, label)
    blocks = []
    for _, query in frame.groupby("query_id", sort=True):
        values = [
            evaluate.ndcg_at_k(group, score, label) for _, group in query.groupby("record_id")
        ]
        values = [v for v in values if np.isfinite(v)]
        blocks.append((sum(values), len(values)))
    rng = np.random.default_rng(42)
    estimates = []
    if len(blocks) > 1:
        array = np.asarray(blocks)
        for _ in range(400):
            numerator, denominator = array[rng.integers(0, len(blocks), len(blocks))].sum(axis=0)
            if denominator:
                estimates.append(numerator / denominator)
    interval = np.percentile(estimates, [2.5, 97.5]) if estimates else [float("nan")] * 2
    result["ndcg@3_lo"], result["ndcg@3_hi"] = map(float, interval)
    result["independent_query_groups"] = frame.query_id.nunique()
    return result
