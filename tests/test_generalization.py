import numpy as np
import pandas as pd
import pytest

from modeling import evaluate
from modeling.features import PRIOR_FEATURES, select
from visibility import generalization as g


def _pairs(categories=("a", "b", "c"), queries=5, records=4, brands=6, seed=0) -> pd.DataFrame:
    """Candidates whose winner is the best-ranked retrieved brand, in every domain."""
    rng = np.random.default_rng(seed)
    rows = []
    for category in categories:
        for q in range(queries):
            for r in range(records):
                for condition in ("search_on", "search_off"):
                    positions = rng.permutation(brands) + 1
                    retrieved = positions <= 4
                    for b in range(brands):
                        on = condition == "search_on" and bool(retrieved[b])
                        rows.append(
                            {
                                "record_id": f"{category}{q}{r}{condition}",
                                "category": category,
                                "query_id": f"{category}_{q}",
                                "model_id": f"m{r % 2}",
                                "condition": condition,
                                "brand": f"{category}{b}",
                                "confidence": "high",
                                "response_decided": 1,
                                "y_top": int(positions[b] == 1),
                                "y_mention": int(positions[b] <= 3),
                                "in_search_results": int(on),
                                "best_position": int(positions[b]) if on else 0,
                                "n_results_mentioning": int(on) * int(5 - positions[b] % 5),
                                **{
                                    c: 0.0
                                    for c in g.GENERAL_FEATURES
                                    if c
                                    not in {
                                        "in_search_results",
                                        "best_position",
                                        "n_results_mentioning",
                                        "best_position_missing",
                                    }
                                },
                                **{p: 0.9 for p in PRIOR_FEATURES},
                            }
                        )
    return pd.DataFrame(rows)


def test_general_model_cannot_see_priors_category_or_condition() -> None:
    frame = g.panel(_pairs(), "y_top")
    left, right = g.design(frame, frame, with_model=True)
    columns = set(left.columns) | set(right.columns)
    assert not columns & set(g.EXCLUDED_FEATURES)
    assert "model_id" in columns
    assert "model_id" not in g.design(frame, frame, with_model=False)[0].columns


def test_panel_keeps_retrieval_on_and_drops_low_confidence_and_undecided() -> None:
    pairs = _pairs()
    low, undecided = "a00search_on", "a01search_on"
    pairs.loc[pairs["record_id"] == low, "confidence"] = "low"
    pairs.loc[pairs["record_id"] == undecided, "response_decided"] = 0
    top, mention = g.panel(pairs, "y_top"), g.panel(pairs, "y_mention")
    assert set(top["condition"]) == set(mention["condition"]) == {"search_on"}
    assert low not in set(top["record_id"]) | set(mention["record_id"])
    assert undecided not in set(top["record_id"])
    assert undecided in set(mention["record_id"])
    assert "best_position_missing" in top.columns
    with pytest.raises(ValueError):
        g.panel(pairs, "y_other")


def test_leave_one_domain_out_never_trains_on_the_held_out_domain() -> None:
    frame = g.panel(_pairs(), "y_top")
    seen = []
    for domain, test_mask in g.leave_one_domain_out(frame):
        assert set(frame.loc[test_mask, "category"]) == {domain}
        assert domain not in set(frame.loc[~test_mask, "category"])
        seen.append(domain)
    assert seen == ["a", "b", "c"]
    with pytest.raises(ValueError):
        next(g.leave_one_domain_out(select(frame, frame["category"] == "a")))


def test_unseen_domain_scores_transfer_a_shared_signal() -> None:
    frame = g.panel(_pairs(queries=6, records=6), "y_top")
    scores, importances = g.unseen_domain_scores(frame, "y_top")
    assert np.isfinite(scores).all()
    labels = frame["y_top"].to_numpy()
    assert evaluate.pr_auc(labels, scores) > 2 * labels.mean()
    assert {row["domain"] for row in importances} == {"a", "b", "c"}


def test_seen_domain_scores_reject_an_unfrozen_query() -> None:
    frame = g.panel(_pairs(), "y_top")
    folds = {q: i % 3 for i, q in enumerate(sorted(frame["query_id"].unique()))}
    assert np.isfinite(g.seen_domain_scores(frame, folds, "y_top")).all()
    folds.pop("a_0")
    with pytest.raises(ValueError, match="a_0"):
        g.seen_domain_scores(frame, folds, "y_top")


def _toy(winner: dict, loser: dict, non_retrieved: dict, queries: int = 6) -> pd.DataFrame:
    rows = []
    for q in range(queries):
        for label, values, retrieved in ((1, winner, 1), (0, loser, 1), (0, non_retrieved, 0)):
            rows.append(
                {
                    "query_id": f"q{q}",
                    "record_id": f"r{q}",
                    "y_top": label,
                    "in_search_results": retrieved,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def test_signal_effect_is_within_response_and_flips_rank_direction() -> None:
    # Winner ranked 1, retrieved loser ranked 5: lower rank wins, so the flipped effect is +1.
    frame = _toy({"best_position": 1.0}, {"best_position": 5.0}, {"best_position": 99.0})
    effect = g.signal_effect(frame, "best_position", "y_top")
    assert effect["effect"] == pytest.approx(1.0)
    assert effect["direction"] == "up"
    # Content is compared only among retrieved candidates: the non-retrieved zero is ignored.
    frame = _toy({"n_editorial": 1.0}, {"n_editorial": 2.0}, {"n_editorial": 0.0})
    effect = g.signal_effect(frame, "n_editorial", "y_top")
    assert effect["effect"] == pytest.approx(-1.0)
    assert effect["n_comparisons"] == 6
    assert effect["direction"] == "down"


def test_signal_direction_is_not_called_with_too_few_queries() -> None:
    frame = _toy({"n_editorial": 2.0}, {"n_editorial": 1.0}, {"n_editorial": 0.0}, queries=3)
    effect = g.signal_effect(frame, "n_editorial", "y_top")
    assert effect["effect"] == pytest.approx(1.0)
    assert effect["direction"] == "n/a"


def test_volume_matching_compares_only_candidates_with_equal_result_counts() -> None:
    # Winner: 3 results, has a year. Losers: 3 results without a year, 1 result with a
    # year, 1 result without. Marginally the winner ties the 1-result loser that has a
    # year (effect 2/3); matched on volume only the 3-result loser is comparable.
    rows = []
    for q in range(6):
        for label, volume, year in ((1, 3, 1.0), (0, 3, 0.0), (0, 1, 1.0), (0, 1, 0.0)):
            rows.append(
                {
                    "query_id": f"q{q}",
                    "record_id": f"r{q}",
                    "y_top": label,
                    "in_search_results": 1,
                    "n_results_mentioning": volume,
                    "snippet_has_year": year,
                }
            )
    frame = pd.DataFrame(rows)
    marginal = g.signal_effect(frame, "snippet_has_year", "y_top")
    matched = g.signal_effect(frame, "snippet_has_year", "y_top", matched_on=g.VOLUME)
    assert marginal["n_comparisons"] == 18
    assert marginal["effect"] == pytest.approx(2 / 3)
    assert matched["n_comparisons"] == 6
    assert matched["effect"] == pytest.approx(1.0)
    # When no equal-volume pair exists the matched effect is undefined, not zero.
    frame = select(frame, frame["n_results_mentioning"].ne(3) | frame["y_top"].eq(1))
    assert (
        g.signal_effect(frame, "snippet_has_year", "y_top", matched_on=g.VOLUME)["direction"]
        == "n/a"
    )


def test_tie_break_makes_top1_independent_of_row_order() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["r"] * 2 * 200,
            "query_id": ["q"] * 400,
            "y_top": [1, 0] * 200,
            "score": [0.5] * 400,
        }
    )
    frame["record_id"] = [f"r{i // 2}" for i in range(400)]
    g.tie_break(frame, ["score"])
    hits, total = g._top1(frame, "score", "y_top")
    assert total == 200
    assert 0.35 < hits / total < 0.65


def test_paired_delta_is_zero_for_identical_scores() -> None:
    frame = g.panel(_pairs(), "y_top")
    frame["s"] = frame["in_search_results"] / frame["best_position"]
    frame["t"] = frame["s"]
    delta = g.paired_delta(frame, "s", "t", "y_top", n_resamples=20)
    assert delta["delta_pr_auc"] == pytest.approx(0.0)
    assert delta["delta_top1"] == pytest.approx(0.0)
    assert delta["delta_pr_auc_lo"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("directions", "expected"),
    [
        (["up"] * 5 + ["n/a"], "strong"),
        (["up", "up", "up", "up", "flat", "flat"], "strong"),
        (["up", "up", "up", "flat", "flat", "flat"], "moderate"),
        (["up", "flat", "flat", "n/a"], "weak"),
        (["up", "up", "up", "up", "down"], "conflicting"),
        (["down"] * 4, "strong"),
    ],
)
def test_classify_follows_the_preregistered_rule(directions, expected) -> None:
    assert g.classify(directions) == expected
