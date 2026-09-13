import numpy as np
import pandas as pd
import pytest

from modeling.features import LANGUAGE_FEATURES
from visibility import generalization as g


def _response(record: str, scale: float = 1.0) -> pd.DataFrame:
    """A: 3 results at rank 1; B: 1 result at rank 4; C: not retrieved."""
    rows = [
        ("A", 1, 3, 1.0, {"editorial": 2, "forum": 1}, 120.0),
        ("B", 1, 1, 4.0, {"affiliate": 1}, 40.0),
        ("C", 0, 0, 99.0, {}, 0.0),
    ]
    frame = []
    for brand, retrieved, volume, position, kinds, words in rows:
        frame.append(
            {
                "record_id": record,
                "brand": brand,
                "in_search_results": retrieved,
                "n_results_mentioning": volume * scale,
                "best_position": position,
                **{f"n_{k}": kinds.get(k, 0) * scale for k in g.SOURCE_KINDS},
                **{f: 0.0 for f in LANGUAGE_FEATURES},
                "snippet_words": words * scale,
            }
        )
    return pd.DataFrame(frame)


def test_relative_features_for_a_single_answer() -> None:
    frame = g.add_relative(_response("r1"), pd.Series({"r1": 5}))
    a, b, c = (frame.set_index("brand").loc[x] for x in "ABC")
    assert a["volume_share"] == pytest.approx(0.6)
    assert (a["volume_vs_leader"], b["volume_vs_leader"]) == (1.0, pytest.approx(1 / 3))
    assert (a["is_top_retrieved"], b["is_top_retrieved"], c["is_top_retrieved"]) == (1, 0, 0)
    assert (a["position_rank_pct"], b["position_rank_pct"]) == (1.0, 0.5)
    assert a["share_editorial"] == pytest.approx(2 / 3)
    assert b["share_affiliate"] == 1.0
    assert (a["snippet_words_pct"], b["snippet_words_pct"]) == (1.0, 0.5)
    # A candidate outside retrieval carries no content or position evidence.
    assert c[[f"{f}_pct" for f in LANGUAGE_FEATURES] + ["position_rank_pct"]].sum() == 0


def test_relative_features_ignore_the_sector_scale_of_raw_counts() -> None:
    small = g.add_relative(_response("r1"), pd.Series({"r1": 5}))
    large = g.add_relative(_response("r1", scale=4.0), pd.Series({"r1": 20}))
    assert np.allclose(small[g.RELATIVE_FEATURES], large[g.RELATIVE_FEATURES])
    assert not np.allclose(small["n_results_mentioning"], large["n_results_mentioning"])


def test_invariant_set_has_no_raw_count_or_identity_inputs() -> None:
    raw = {"n_results_mentioning", "best_position", "snippet_words", "n_editorial"}
    assert not set(g.INVARIANT_FEATURES) & (raw | set(g.EXCLUDED_FEATURES))
    assert g.PRESENCE in g.INVARIANT_FEATURES
