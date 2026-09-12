"""Concentration and fairness: how few brands get named, and who gains from retrieval.

The generalization report says which signals decide visibility. This one asks the
distributional question behind the project's premise: when an assistant answers
"recommend me a X", how much of the market does it actually show, and does turning
retrieval on widen or narrow the gap between brands people already know and the
rest?

Rules fixed before looking at any result:

* **Effective brand count, not raw HHI, and never Gini.** A concentration index has
  to be comparable across sectors whose registries differ in size (cosmetics 74
  brands, VPN 24). Raw HHI cannot be: its floor is ``1/N``. Gini is worse here --
  it is driven mostly by how many registry brands are never named at all, which is
  a property of our brand list, not of assistant behaviour. ``N_eff = 1/HHI`` reads
  directly: "of 24 VPN brands, the assistant behaves as if about three existed."
  Normalised HHI is written to the CSV for cross-sector arithmetic, and the count of
  never-named brands is reported separately, descriptively.
* **Shares are over naming events, not responses.** One answer names several brands,
  so a share whose denominator is responses would not sum to one.
* **The top-1 share uses decided responses only**, the same denominator ``y_top``
  is defined on.
* **Recognition is the retrieval-off mention rate**, the only recognition proxy in
  this corpus. It is a measure of prior visibility, *not* of company size; the
  report says so, because "small brand" and "brand this model has not memorised"
  are different things.
* **Tercile membership is assigned once, from the full sample**, and the bootstrap
  then varies only the gain inside fixed terciles. Letting the terciles move with
  each resample would measure regression to the mean.
* **Origin is read from a file sealed before this module existed** and pinned by
  hash; see ``configs/visibility/brand_origin.yaml``.
* **A direction is only called with at least ``MIN_QUERY_GROUPS`` independent
  queries**, the same threshold the generalization report uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

from evidence_eval.io import sha256
from modeling.features import select

from .generalization import MIN_QUERY_GROUPS

ORIGIN_PATH = Path("configs/visibility/brand_origin.yaml")
# Pins the sealed labels: the analysis refuses to run against an edited file, the
# same way the evidence workspace pins its inputs.
ORIGIN_SHA256 = "ad2f4f702524093749a60a5349f02ec822c24af26b2ba797fa612f852e5aac57"
TOP_K = 3
TERCILE_LABELS = ("düşük", "orta", "yüksek")
N_RESAMPLES = 2000
SEED = 42
TARGETS = ("y_mention", "y_top")


def load_origin(path: Path = ORIGIN_PATH, *, expected: str | None = ORIGIN_SHA256) -> dict:
    """Read the sealed origin labels, refusing a file that changed after sealing."""
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(
            f"{path} değişmiş (sha256={digest}); menşe etiketleri mühürlendikten sonra "
            "düzenlenmişse rapor yeniden üretilmeli ve ORIGIN_SHA256 güncellenmelidir."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def origin_of(labels: dict, category: str) -> pd.Series:
    """brand -> tr | global | unclear for one sector."""
    entries = labels.get(category) or {}
    return pd.Series(
        {brand: str(value.get("origin", "unclear")) for brand, value in entries.items()},
        dtype="object",
        name="origin",
    )


# --- concentration -------------------------------------------------------------


def shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype=float)


def hhi(share: np.ndarray) -> float:
    return float(np.square(share).sum())


def effective_brands(share: np.ndarray) -> float:
    """1/HHI: how many equally-named brands would produce this concentration."""
    index = hhi(share)
    return float(1.0 / index) if index > 0 else float("nan")


def normalised_hhi(share: np.ndarray, n_brands: int) -> float:
    """(HHI - 1/N) / (1 - 1/N): removes the registry-size floor for cross-sector use."""
    if n_brands < 2:
        return float("nan")
    floor = 1.0 / n_brands
    index = hhi(share)
    return float((index - floor) / (1.0 - floor)) if index > 0 else float("nan")


def top_k_share(share: np.ndarray, k: int = TOP_K) -> float:
    if not share.sum():
        return float("nan")
    return float(np.sort(share)[::-1][:k].sum())


def _event_counts(part: pd.DataFrame, target: str, queries: list, n_brands: int) -> np.ndarray:
    """Naming events per (query, brand): one row per query, so queries resample as blocks."""
    counts = np.zeros((len(queries), n_brands), dtype=float)
    rows = select(part, part[target] == 1)
    if len(rows):
        index = {q: i for i, q in enumerate(queries)}
        np.add.at(
            counts,
            (
                rows["query_id"].map(index).to_numpy(dtype=int),  # type: ignore[arg-type]
                rows["brand_code"].to_numpy(dtype=int),
            ),
            1.0,
        )
    return counts


def _measure(counts: np.ndarray, n_brands: int) -> tuple[float, float, float]:
    share = shares(counts)
    return effective_brands(share), top_k_share(share), normalised_hhi(share, n_brands)


def concentration(
    pairs: pd.DataFrame,
    *,
    target: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Effective brand count per (category, condition, model), with a query bootstrap.

    ``y_top`` is measured on decided responses only; ``model_id`` "ALL" pools the
    assistants, which is the row to quote when a sector is described as a whole.
    """
    decided = cast("pd.Series", pairs["response_decided"].astype(bool))
    frame = pairs if target != "y_top" else select(pairs, decided)
    rows = []
    for category, sector in frame.groupby("category", sort=True):
        brands = sorted(sector["brand"].unique())
        codes = {brand: i for i, brand in enumerate(brands)}
        sector = sector.assign(brand_code=sector["brand"].map(codes))  # type: ignore[arg-type]
        for condition, part in sector.groupby("condition", sort=True):
            models = [*sorted(part["model_id"].unique()), "ALL"]
            queries = sorted(part["query_id"].unique())
            for model in models:
                cell = part if model == "ALL" else select(part, part["model_id"] == model)
                counts = _event_counts(cell, target, queries, len(brands))
                named = counts.sum(axis=0)
                point = _measure(named, len(brands))
                usable = int((counts.sum(axis=1) > 0).sum())
                rng = np.random.default_rng(seed)
                draws = np.array(
                    [
                        _measure(
                            counts[rng.integers(0, len(queries), len(queries))].sum(axis=0),
                            len(brands),
                        )
                        for _ in range(n_resamples if usable > 1 else 0)
                    ]
                ).reshape(-1, 3)
                row = {
                    "category": category,
                    "condition": condition,
                    "model_id": model,
                    "target": target,
                    "n_brands": len(brands),
                    "n_events": int(named.sum()),
                    "n_never_named": int((named == 0).sum()),
                    "independent_query_groups": usable,
                }
                for j, name in enumerate(("n_eff", "top3_share", "hhi_normalised")):
                    row[name] = float(point[j])
                    column = draws[:, j][np.isfinite(draws[:, j])] if len(draws) else np.array([])
                    low, high = (
                        np.percentile(column, [2.5, 97.5]) if len(column) else (np.nan, np.nan)
                    )
                    row[f"{name}_lo"], row[f"{name}_hi"] = float(low), float(high)
                rows.append(row)
    return pd.DataFrame(rows)


# --- who gains when retrieval is switched on -----------------------------------


def _rate_matrix(part: pd.DataFrame, condition: str, queries: list, brands: list) -> np.ndarray:
    """Mention rate per (query, brand) under one retrieval condition.

    Rates are taken per query first, so every query weighs the same however many
    responses it produced -- the unit the cluster bootstrap resamples.
    """
    rows = select(part, part["condition"] == condition)
    table = rows.pivot_table(index="query_id", columns="brand", values="y_mention", aggfunc="mean")
    return table.reindex(index=queries, columns=brands).to_numpy(dtype=float, na_value=0.0)


def _brand_rates(part: pd.DataFrame) -> pd.DataFrame:
    """Per brand: mention rate with retrieval off, with it on, and the difference."""
    queries = sorted(part["query_id"].astype(str).unique())
    brands = sorted(part["brand"].unique())
    part = part.assign(query_id=part["query_id"].astype(str))
    off = _rate_matrix(part, "search_off", queries, brands)
    on = _rate_matrix(part, "search_on", queries, brands)
    table = pd.DataFrame(
        {"mention_off": off.mean(axis=0), "mention_on": on.mean(axis=0)},
        index=pd.Index(brands, name="brand"),
    )
    table["retrieval_gain"] = table["mention_on"] - table["mention_off"]
    return table


def _group_bootstrap(
    part: pd.DataFrame,
    membership: pd.Series,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Mean retrieval gain per group, resampling queries and keeping groups fixed.

    The per-query gain matrix is built once; a resample is then an average over
    sampled rows, so the interval costs a matrix product rather than a regroup.
    """
    queries = sorted(part["query_id"].astype(str).unique())
    brands = sorted(part["brand"].unique())
    part = part.assign(query_id=part["query_id"].astype(str))
    gain = _rate_matrix(part, "search_on", queries, brands) - _rate_matrix(
        part, "search_off", queries, brands
    )
    point = _brand_rates(part).join(membership.rename("group"), how="inner")
    position = {brand: i for i, brand in enumerate(brands)}
    rng = np.random.default_rng(seed)
    samples = (
        rng.integers(0, len(queries), (n_resamples, len(queries)))
        if len(queries) > 1
        else np.empty((0, len(queries)), dtype=int)
    )
    rows = []
    for group in cast(list, membership.dropna().unique()):
        members = select(point, point["group"] == group)
        columns = [position[b] for b in members.index if b in position]
        draws = (
            gain[samples][:, :, columns].mean(axis=(1, 2))
            if len(samples) and columns
            else np.array([])
        )
        low, high = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
        rows.append(
            {
                "group": group,
                "n_brands": int(len(members)),
                "mention_off": float(members["mention_off"].mean()),
                "mention_on": float(members["mention_on"].mean()),
                "retrieval_gain": float(members["retrieval_gain"].mean()),
                "retrieval_gain_lo": float(low),
                "retrieval_gain_hi": float(high),
                "independent_query_groups": len(queries),
            }
        )
    return pd.DataFrame(rows)


def recognition_terciles(
    pairs: pd.DataFrame,
    category: str,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Does retrieval lift the brands the model already knows, or the others?

    Brands are split into terciles by their retrieval-off mention rate once, on the
    full sample; the bootstrap then only moves the gain within those fixed groups.
    """
    part = select(pairs, pairs["category"] == category)
    board = _brand_rates(part)
    ranks = board["mention_off"].rank(method="first", pct=True)
    membership = cast(
        "pd.Series",
        pd.Series(
            pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0], labels=list(TERCILE_LABELS), include_lowest=True),
            index=board.index,
        ).astype("object"),
    )
    table = _group_bootstrap(part, membership, n_resamples=n_resamples, seed=seed)
    order = {label: i for i, label in enumerate(TERCILE_LABELS)}
    table = table.assign(category=category).sort_values("group", key=lambda s: s.map(order))
    return table.reset_index(drop=True)


def origin_groups(
    pairs: pd.DataFrame,
    category: str,
    labels: dict,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Turkish-origin vs global brands in one sector; ``unclear`` is left out."""
    origins = origin_of(labels, category)
    known = cast("pd.Series", origins[origins.isin(("tr", "global"))])
    part = select(pairs, pairs["category"] == category)
    if not (known == "tr").any():
        return pd.DataFrame()
    table = _group_bootstrap(part, known, n_resamples=n_resamples, seed=seed)
    dropped = int((origins == "unclear").sum())
    return table.assign(category=category, n_unclear=dropped).reset_index(drop=True)


def callable_direction(groups: int = 0) -> bool:
    """Enough independent queries to call a direction at all."""
    return groups >= MIN_QUERY_GROUPS
