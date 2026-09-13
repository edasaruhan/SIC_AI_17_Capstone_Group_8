import numpy as np
import pandas as pd
import pytest

from visibility import fairness as f


def _pairs(rates: dict[str, tuple[float, float]], queries: int = 6, reps: int = 4):
    """A corpus where each brand is named at fixed rates: (retrieval off, retrieval on)."""
    rows = []
    for query in range(queries):
        for rep in range(reps):
            for brand, (off, on) in rates.items():
                for condition, rate in (("search_off", off), ("search_on", on)):
                    named = int((rep + query) % reps < round(rate * reps))
                    rows.append(
                        {
                            "category": "vpn",
                            "condition": condition,
                            "model_id": "m1",
                            "query_id": f"q{query}",
                            "record_id": f"q{query}_{rep}_{condition}",
                            "brand": brand,
                            "y_mention": named,
                            "y_top": named,
                            "response_decided": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_effective_brands_counts_equally_named_brands():
    assert f.effective_brands(f.shares(np.array([1.0, 1.0, 1.0, 1.0]))) == pytest.approx(4.0)
    assert f.effective_brands(f.shares(np.array([9.0, 0.0, 0.0]))) == pytest.approx(1.0)
    # Hand-computed: shares 0.75/0.25 -> HHI 0.5625 + 0.0625 = 0.625 -> 1/0.625 = 1.6
    assert f.effective_brands(f.shares(np.array([3.0, 1.0]))) == pytest.approx(1.6)


def test_normalised_hhi_is_zero_when_every_brand_is_equal():
    assert f.normalised_hhi(f.shares(np.ones(5)), 5) == pytest.approx(0.0)
    assert f.normalised_hhi(f.shares(np.array([1.0, 0.0, 0.0, 0.0, 0.0])), 5) == pytest.approx(1.0)


def test_top_k_share_takes_the_largest_shares():
    share = f.shares(np.array([5.0, 3.0, 1.0, 1.0]))
    assert f.top_k_share(share, k=2) == pytest.approx(0.8)
    assert np.isnan(f.top_k_share(np.zeros(3)))


def test_concentration_reports_never_named_brands_and_a_narrow_interval():
    pairs = _pairs({"A": (1.0, 1.0), "B": (0.0, 0.0), "C": (0.0, 0.0)})
    table = f.concentration(pairs, target="y_mention", n_resamples=200)
    pooled = table[(table["model_id"] == "ALL") & (table["condition"] == "search_on")].iloc[0]
    assert pooled["n_brands"] == 3
    assert pooled["n_never_named"] == 2
    # One brand takes every naming event, so the effective count is 1 in every resample.
    assert pooled["n_eff"] == pytest.approx(1.0)
    assert pooled["n_eff_lo"] == pytest.approx(1.0)
    assert pooled["n_eff_hi"] == pytest.approx(1.0)


def test_concentration_on_y_top_uses_decided_responses_only():
    pairs = _pairs({"A": (1.0, 1.0), "B": (0.5, 0.5)})
    undecided = pairs.assign(response_decided=False)
    # Nothing is decided, so there is no y_top panel at all -- not a zero-event row.
    assert f.concentration(undecided, target="y_top", n_resamples=0).empty
    assert f.concentration(pairs, target="y_top", n_resamples=0)["n_events"].sum() > 0


def test_recognition_terciles_split_brands_and_keep_the_gain_sign():
    # Unknown brand gains from retrieval; the well-known one loses.
    pairs = _pairs({"low": (0.0, 0.5), "mid": (0.25, 0.25), "high": (1.0, 0.5)})
    table = f.recognition_terciles(pairs, "vpn", n_resamples=200).set_index("group")
    assert list(table.index) == list(f.TERCILE_LABELS)
    assert (table["n_brands"] == 1).all()
    assert table.loc["düşük", "retrieval_gain"] > 0
    assert table.loc["yüksek", "retrieval_gain"] < 0
    assert table.loc["düşük", "retrieval_gain_lo"] <= table.loc["düşük", "retrieval_gain"]
    assert table.loc["düşük", "retrieval_gain_hi"] >= table.loc["düşük", "retrieval_gain"]
    assert (table["independent_query_groups"] == 6).all()


def test_origin_groups_is_empty_without_a_turkish_brand():
    pairs = _pairs({"A": (0.5, 0.5), "B": (0.5, 0.5)})
    labels = {"vpn": {"A": {"origin": "global"}, "B": {"origin": "global"}}}
    assert f.origin_groups(pairs, "vpn", labels, n_resamples=0).empty


def test_origin_groups_drops_unclear_brands():
    pairs = _pairs({"A": (0.2, 0.6), "B": (0.5, 0.5), "C": (0.5, 0.5)})
    labels = {
        "vpn": {
            "A": {"origin": "tr"},
            "B": {"origin": "global"},
            "C": {"origin": "unclear"},
        }
    }
    table = f.origin_groups(pairs, "vpn", labels, n_resamples=100)
    assert set(table["group"]) == {"tr", "global"}
    assert table["n_brands"].sum() == 2
    assert table["n_unclear"].iloc[0] == 1
    assert table.set_index("group").loc["tr", "retrieval_gain"] > 0


def test_load_origin_refuses_an_edited_file(tmp_path):
    path = tmp_path / "brand_origin.yaml"
    path.write_text("rule: test\nvpn:\n  A: {origin: tr}\n", encoding="utf-8")
    assert f.load_origin(path, expected=None)["vpn"]["A"]["origin"] == "tr"
    with pytest.raises(ValueError, match="değişmiş"):
        f.load_origin(path, expected="0" * 64)


def test_sealed_labels_match_the_pinned_hash():
    """The committed labels are the ones the analysis was run against."""
    labels = f.load_origin()
    assert labels["sealed_before_analysis"] is True
    assert (f.origin_of(labels, "cosmetics") == "tr").sum() > 0
    # No Turkish-origin VPN brand exists in the registry; the report says so.
    assert (f.origin_of(labels, "vpn") == "tr").sum() == 0
