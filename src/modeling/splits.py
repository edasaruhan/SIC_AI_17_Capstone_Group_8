"""Query-level grouped folds, frozen to disk.

Repetitions of the same query are not independent: each cell repeats one query up
to 30 times at temperature 0.7. Splitting rows at random would put near-duplicate
responses on both sides of the split and inflate every score. Folds are therefore
assigned by ``query_id``, and every query's rows travel together.

With 10 query groups per English domain and 5 per Turkish domain a single
train/test split would waste most of the data, so the pipeline uses grouped
K-fold and evaluates on pooled out-of-fold predictions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from random import Random

import pandas as pd

SPLITS_DIR = Path("data/processed/modeling")


def assign_folds(frame: pd.DataFrame, *, n_splits: int = 5, seed: int = 42) -> dict[str, int]:
    """Map every query_id to a fold, dealing queries round-robin inside a category.

    Dealing within category keeps each fold populated for every category, so a
    fold never trains on a domain it is then asked to score.
    """
    rng = Random(seed)
    fold_by_query: dict[str, int] = {}
    pairs = frame[["category", "query_id"]].drop_duplicates()
    for _category, group in pairs.groupby("category", sort=True):
        queries = sorted(group["query_id"].astype(str))
        rng.shuffle(queries)
        for index, query_id in enumerate(queries):
            fold_by_query[query_id] = index % n_splits
    return fold_by_query


def effective_n_splits(frame: pd.DataFrame, requested: int) -> int:
    """Never ask for more folds than the smallest category has query groups."""
    per_category = frame.groupby("category")["query_id"].nunique()
    return max(2, min(int(requested), int(per_category.min())))


def freeze(track: str, fold_by_query: dict[str, int], frame: pd.DataFrame) -> Path:
    """Write the fold assignment so every later run reuses the same grouping."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"splits_{track}.json"
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for category, group in frame.groupby("category", sort=True):
        for query_id in sorted(group["query_id"].astype(str).unique()):
            counts[str(category)][query_id] = fold_by_query[query_id]
    payload = {
        "track": track,
        "n_splits": max(fold_by_query.values()) + 1,
        "grouping": "query_id",
        "seed": 42,
        "folds_by_category": dict(counts),
        "responses": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_frozen(track: str) -> dict[str, int]:
    path = SPLITS_DIR / f"splits_{track}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        query_id: fold
        for category in payload["folds_by_category"].values()
        for query_id, fold in category.items()
    }


def leaks(frame: pd.DataFrame, fold_by_query: dict[str, int]) -> list[str]:
    """Return query_ids whose rows land in more than one fold."""
    seen: dict[str, set[int]] = defaultdict(set)
    for query_id in frame["query_id"].astype(str):
        seen[query_id].add(fold_by_query[query_id])
    return sorted(q for q, folds in seen.items() if len(folds) > 1)
