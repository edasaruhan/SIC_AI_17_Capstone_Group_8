import pandas as pd
import pytest

from modeling import features, pairs, splits
from modeling.brands import comparison_key, load_registry


def test_comparison_key_undoes_turkish_casefolding_of_english_brands() -> None:
    # The collection pipeline casefolded with Turkish rules, so "IVPN" was stored
    # as "ıvpn" and "Private Internet Access" as "private ınternet access".
    assert comparison_key("ıvpn") == comparison_key("IVPN")
    assert comparison_key("private ınternet access") == comparison_key("Private Internet Access")
    assert comparison_key("ınnisfree") == comparison_key("Innisfree")


def test_comparison_key_ignores_spacing_punctuation_and_symbols() -> None:
    assert comparison_key("proton vpn") == comparison_key("ProtonVPN")
    assert comparison_key("hide.me") == comparison_key("hide me")
    assert comparison_key("NordVPN™") == comparison_key("nordvpn")


def test_comparison_key_keeps_genuinely_different_products_apart() -> None:
    assert comparison_key("VS Code") != comparison_key("Visual Studio")
    assert comparison_key("Cloudflare Pages") != comparison_key("Cloudflare Workers")


def test_registry_merges_aliases_and_drops_non_brands() -> None:
    registry = load_registry("vpn")
    assert registry.resolve("mullvad vpn") == "Mullvad"
    assert registry.resolve("ıvpn") == "IVPN"
    # Protocols, auditors and publications are not consumer VPN services.
    assert registry.resolve("WireGuard") is None
    assert registry.resolve("Deloitte") is None
    assert registry.resolve("PCMag") is None


def test_registry_exclusion_never_overrides_a_curated_brand() -> None:
    hosting = load_registry("hosting")
    assert hosting.resolve("Cloudflare") is None
    assert hosting.resolve("Cloudflare Pages") == "Cloudflare Pages"


def test_registry_rejects_a_surface_form_claimed_by_two_brands() -> None:
    from modeling.brands import _build

    with pytest.raises(ValueError, match="maps to both"):
        _build("vpn", {"canonical": {"Alpha": ["shared name"], "Beta": ["Shared Name"]}})


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["vpn"] * 6,
            "query_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
        }
    )


def test_folds_never_split_a_query_across_sides() -> None:
    frame = _frame()
    fold_by_query = splits.assign_folds(frame, n_splits=3, seed=42)
    assert splits.leaks(frame, fold_by_query) == []


def test_effective_n_splits_never_exceeds_the_smallest_category() -> None:
    frame = pd.DataFrame(
        {
            "category": ["vpn", "vpn", "vpn", "cosmetics", "cosmetics"],
            "query_id": ["a", "b", "c", "d", "e"],
        }
    )
    assert splits.effective_n_splits(frame, 5) == 2


def _pairs() -> pd.DataFrame:
    rows = []
    for index, (query, brand, top, mention) in enumerate(
        [
            ("q1", "NordVPN", 1, 1),
            ("q1", "Mullvad", 0, 1),
            ("q2", "NordVPN", 1, 1),
            ("q2", "Mullvad", 0, 0),
            ("q3", "NordVPN", 0, 0),
            ("q3", "Mullvad", 1, 1),
        ]
    ):
        rows.append(
            {
                "record_id": f"r{index}",
                "category": "vpn",
                "query_id": query,
                "brand": brand,
                "condition": "search_off",
                "y_top": top,
                "y_mention": mention,
                "response_decided": 1,
                "best_position": 0,
            }
        )
    return pd.DataFrame(rows)


def test_priors_are_fitted_on_training_rows_only() -> None:
    frame = _pairs()
    train_mask = frame["query_id"] != "q3"
    with_priors = features.add_priors(frame, train_mask)
    # q3 is the only response where Mullvad wins. If its outcome reached the
    # prior, Mullvad's prior would rise above NordVPN's.
    mullvad = with_priors.loc[with_priors["brand"] == "Mullvad", "prior_top_off"].iloc[0]
    nord = with_priors.loc[with_priors["brand"] == "NordVPN", "prior_top_off"].iloc[0]
    assert nord > mullvad


def test_priors_fall_back_to_the_category_base_rate_for_unseen_brands() -> None:
    frame = _pairs()
    extra = frame.iloc[[0]].copy()
    extra["brand"] = "Surfshark"
    extra["record_id"] = "r99"
    extra["query_id"] = "q3"
    frame = pd.concat([frame, extra], ignore_index=True)
    train_mask = frame["query_id"] != "q3"
    with_priors = features.add_priors(frame, train_mask)
    surfshark = with_priors.loc[with_priors["brand"] == "Surfshark", "prior_top_off"].iloc[0]
    assert 0.0 <= surfshark <= 1.0
    assert not pd.isna(surfshark)


def test_finalise_flags_a_missing_retrieval_position_instead_of_calling_it_rank_zero() -> None:
    frame = features.add_priors(_pairs(), pd.Series(True, index=_pairs().index))
    prepared = features.finalise(frame)
    assert (prepared["best_position"] == 99).all()
    assert (prepared["best_position_missing"] == 1).all()


def test_no_answer_derived_column_is_in_the_feature_set() -> None:
    # The task is to predict the answer, so anything computed from the answer is
    # a label in disguise.
    forbidden = {
        "response_length_words",
        "number_of_brands_mentioned",
        "hedging_level",
        "answer_mode",
        "evidence_style",
        "confidence",
        "first_mentioned_brand",
    }
    allowed = set(features.NUMERIC_FEATURES) | set(features.CATEGORICAL_FEATURES)
    assert forbidden & allowed == set()


def test_feature_blocks_cover_the_declared_numeric_features() -> None:
    blocks = set(
        features.PRIOR_FEATURES + features.STRUCTURAL_FEATURES + features.LANGUAGE_FEATURES
    )
    assert blocks == set(features.NUMERIC_FEATURES)


def test_ambiguous_short_brand_names_need_a_capitalised_match() -> None:
    registry = load_registry("hosting")
    forms = pairs._match_forms(registry, "Render")
    assert pairs._mentions(forms, "Deploy on Render for free", " deploy on render for free ")
    # "render the page" is ordinary prose, not a brand mention.
    assert not pairs._mentions(
        forms, "how to render the page fast", " how to render the page fast "
    )


def test_length_grouped_batches_cover_every_row_exactly_once() -> None:
    import numpy as np

    from modeling.cross_encoder import length_grouped_batches

    lengths = [7, 200, 12, 350, 9, 15, 500, 11, 8, 22, 300, 13]
    for rng in (None, np.random.default_rng(0)):
        batches = length_grouped_batches(lengths, batch_size=4, rng=rng, megabatch_factor=2)
        flat = [index for batch in batches for index in batch]
        assert sorted(flat) == list(range(len(lengths)))
        assert all(1 <= len(batch) <= 4 for batch in batches)


def test_length_grouped_batches_put_similar_lengths_together() -> None:
    from modeling.cross_encoder import length_grouped_batches

    lengths = [5, 400, 6, 410, 7, 420, 8, 430]
    batches = length_grouped_batches(lengths, batch_size=4, rng=None, megabatch_factor=10)
    widths = [max(lengths[index] for index in batch) for batch in batches]
    # Padding cost is the batch maximum. The short rows must end up in a batch of
    # their own rather than being stretched to the longest row in the set.
    assert min(widths) < 100
    padded = sum(len(batch) * width for batch, width in zip(batches, widths, strict=True))
    assert padded < len(lengths) * max(lengths)


def test_scattering_scores_back_restores_the_callers_row_order() -> None:
    import numpy as np

    from modeling.cross_encoder import length_grouped_batches

    lengths = [5, 400, 6, 410, 7, 420, 8, 430, 9, 440]
    truth = np.arange(len(lengths), dtype=float)
    scores = np.zeros(len(lengths))
    for batch in length_grouped_batches(lengths, batch_size=3, rng=None, megabatch_factor=2):
        # Stand in for the model: return each row's own index as its score.
        scores[batch] = truth[batch]
    assert np.array_equal(scores, truth)


def test_unambiguous_brand_names_match_case_insensitively() -> None:
    registry = load_registry("vpn")
    forms = pairs._match_forms(registry, "NordVPN")
    assert pairs._mentions(forms, "nordvpn review 2026", " nordvpn review 2026 ".replace(" ", ""))
