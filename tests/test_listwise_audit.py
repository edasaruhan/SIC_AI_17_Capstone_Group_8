"""Score-panel checks do not import torch, access the network or train models."""

import runpy

import numpy as np
import pandas as pd
import pytest

CHECK = runpy.run_path("scripts/audit_listwise_run.py")["check_scores"]


@pytest.fixture
def panel():
    return pd.DataFrame(
        {
            "record_id": ["r1", "r1", "r2", "r2"],
            "brand": ["A", "B", "A", "B"],
            "query_id": ["q1", "q1", "q2", "q2"],
            "fold": [0, 0, 1, 1],
            "y_top": [1, 0, 0, 1],
            "score_M3": [0.7, 0.3, 0.4, 0.6],
        }
    )


def test_valid_panel_can_be_reordered(panel):
    CHECK(panel.iloc[::-1], panel, {"q1": 0, "q2": 1})


@pytest.mark.parametrize("corruption", ["candidate", "label", "query", "fold", "duplicate"])
def test_bad_candidate_or_fold_panel_rejected(panel, corruption):
    changed = panel.copy()
    if corruption == "candidate":
        changed.loc[0, "brand"] = "C"
    elif corruption == "label":
        changed.loc[[0, 1], "y_top"] = [0, 1]
    elif corruption == "query":
        changed.loc[[0, 1], "query_id"] = "q2"
    elif corruption == "fold":
        changed.loc[0, "fold"] = 1
    else:
        changed = pd.concat([changed, changed.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        CHECK(changed, panel, {"q1": 0, "q2": 1})


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1, 0.1])
def test_invalid_probabilities_rejected(panel, value):
    changed = panel.copy()
    changed.loc[0, "score_M3"] = value
    with pytest.raises(ValueError, match="probabilities"):
        CHECK(changed, panel, {"q1": 0, "q2": 1})
