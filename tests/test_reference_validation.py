import pandas as pd
import pytest

from reference_validation import (
    RecommendationRate,
    ValidationError,
    build_validation_results,
    percentage_point_change,
    recommendation_rate,
    validate_expected_findings,
)


def test_recommendation_rate_keeps_null_winners_in_denominator() -> None:
    frame = pd.DataFrame(
        {
            "category": ["vpn", "vpn", "vpn", "editors"],
            "condition": ["search_on", "search_on", "search_on", "search_on"],
            "top_recommendation_canonical": ["NordVPN", None, "Mullvad", "VS Code"],
        }
    )

    result = recommendation_rate(
        frame,
        category="vpn",
        condition="search_on",
        brand="NordVPN",
    )

    assert result.count == 1
    assert result.denominator == 3
    assert result.rate_percent == pytest.approx(100 / 3)


def test_recommendation_rate_rejects_an_empty_segment() -> None:
    frame = pd.DataFrame(
        {
            "category": ["vpn"],
            "condition": ["search_off"],
            "top_recommendation_canonical": [None],
        }
    )

    with pytest.raises(ValidationError, match="No rows"):
        recommendation_rate(
            frame,
            category="editors",
            condition="search_on",
            brand="VS Code",
        )


def test_percentage_point_change_requires_comparable_rates() -> None:
    before = RecommendationRate("vpn", "search_off", "NordVPN", 1, 4)
    after = RecommendationRate("vpn", "search_on", "NordVPN", 2, 4)

    assert percentage_point_change(after, before) == 25

    with pytest.raises(ValidationError, match="same category and brand"):
        percentage_point_change(
            RecommendationRate("editors", "search_on", "VS Code", 2, 4),
            before,
        )


def test_build_and_validate_expected_findings() -> None:
    rows = [
        {
            "category": "vpn",
            "condition": "search_off",
            "top_recommendation_canonical": (
                "Mullvad" if index < 589 else "NordVPN" if index < 652 else None
            ),
        }
        for index in range(1200)
    ]
    rows.extend(
        {
            "category": "vpn",
            "condition": "search_on",
            "top_recommendation_canonical": "NordVPN" if index < 397 else None,
        }
        for index in range(1188)
    )
    rows.extend(
        {
            "category": "editors",
            "condition": "search_off",
            "top_recommendation_canonical": "VS Code" if index < 975 else None,
        }
        for index in range(1200)
    )
    rows.extend(
        {
            "category": "editors",
            "condition": "search_on",
            "top_recommendation_canonical": "VS Code" if index < 970 else None,
        }
        for index in range(1200)
    )
    frame = pd.DataFrame(rows)

    results = build_validation_results(frame)
    validation = validate_expected_findings(results)

    assert validation["assessment"] == "Ready to share"
    assert validation["vpn_nordvpn_delta_pp"] == pytest.approx(28.1675084)
    assert validation["editors_vscode_delta_pp"] == pytest.approx(-0.4166667)


def test_validate_expected_findings_rejects_changed_counts() -> None:
    results = pd.DataFrame(
        [
            {
                "category": category,
                "condition": condition,
                "brand": brand,
                "count": count - (1 if brand == "Mullvad" else 0),
                "denominator": denominator,
                "rate_percent": count / denominator * 100,
            }
            for (category, condition, brand), (count, denominator) in {
                ("vpn", "search_off", "Mullvad"): (589, 1200),
                ("vpn", "search_off", "NordVPN"): (63, 1200),
                ("vpn", "search_on", "NordVPN"): (397, 1188),
                ("editors", "search_off", "VS Code"): (975, 1200),
                ("editors", "search_on", "VS Code"): (970, 1200),
            }.items()
        ]
    )

    with pytest.raises(ValidationError, match="Finding count changed"):
        validate_expected_findings(results)
