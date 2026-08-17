"""Reproduce selected findings from the English brand-bias reference dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from reference_data import KNOWN_INCOMPLETE_CELLS

REQUIRED_COLUMNS = {
    "record_id",
    "category",
    "condition",
    "model_id",
    "query_id",
    "top_recommendation_canonical",
    "core__confidence_in_extraction",
}
EXPECTED_FINDING_COUNTS = {
    ("vpn", "search_off", "Mullvad"): (589, 1200),
    ("vpn", "search_off", "NordVPN"): (63, 1200),
    ("vpn", "search_on", "NordVPN"): (397, 1188),
    ("editors", "search_off", "VS Code"): (975, 1200),
    ("editors", "search_on", "VS Code"): (970, 1200),
}


class ValidationError(ValueError):
    """Raised when an analysis input or reproduced finding is not trustworthy."""


@dataclass(frozen=True)
class RecommendationRate:
    """A recommendation count and its full response-level denominator."""

    category: str
    condition: str
    brand: str
    count: int
    denominator: int

    @property
    def rate_percent(self) -> float:
        if self.denominator == 0:
            raise ZeroDivisionError("Recommendation rate denominator cannot be zero")
        return self.count / self.denominator * 100

    def as_record(self, *, finding: str) -> dict[str, Any]:
        return {
            "finding": finding,
            "category": self.category,
            "brand": self.brand,
            "condition": self.condition,
            "count": self.count,
            "denominator": self.denominator,
            "rate_percent": self.rate_percent,
        }


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValidationError(f"Missing analysis columns: {sorted(missing)}")


def recommendation_rate(
    frame: pd.DataFrame,
    *,
    category: str,
    condition: str,
    brand: str,
) -> RecommendationRate:
    """Calculate a brand rate using every response in the segment as denominator."""

    _require_columns(frame, {"category", "condition", "top_recommendation_canonical"})
    segment = frame.loc[frame["category"].eq(category) & frame["condition"].eq(condition)]
    if segment.empty:
        raise ValidationError(f"No rows for category={category}, condition={condition}")

    return RecommendationRate(
        category=category,
        condition=condition,
        brand=brand,
        count=int(segment["top_recommendation_canonical"].eq(brand).sum()),
        denominator=len(segment),
    )


def percentage_point_change(after: RecommendationRate, before: RecommendationRate) -> float:
    """Return after-minus-before change in percentage points."""

    if after.category != before.category or after.brand != before.brand:
        raise ValidationError("Percentage-point comparisons require the same category and brand")
    return after.rate_percent - before.rate_percent


def build_validation_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the five rates needed for the two selected source findings."""

    specifications = [
        ("vpn_shift", "vpn", "search_off", "Mullvad"),
        ("vpn_shift", "vpn", "search_off", "NordVPN"),
        ("vpn_shift", "vpn", "search_on", "NordVPN"),
        ("editors_control", "editors", "search_off", "VS Code"),
        ("editors_control", "editors", "search_on", "VS Code"),
    ]
    records = []
    for finding, category, condition, brand in specifications:
        rate = recommendation_rate(
            frame,
            category=category,
            condition=condition,
            brand=brand,
        )
        records.append(rate.as_record(finding=finding))
    return pd.DataFrame.from_records(records)


def validate_expected_findings(results: pd.DataFrame) -> dict[str, Any]:
    """Require exact source counts and return the two headline changes."""

    _require_columns(
        results,
        {"category", "condition", "brand", "count", "denominator", "rate_percent"},
    )
    actual_counts: dict[tuple[str, str, str], tuple[int, int]] = {}
    for row in results.to_dict(orient="records"):
        key = (str(row["category"]), str(row["condition"]), str(row["brand"]))
        if key in actual_counts:
            raise ValidationError(f"Duplicate finding row: {key}")
        actual_counts[key] = (int(row["count"]), int(row["denominator"]))

    for key, (expected_count, expected_denominator) in EXPECTED_FINDING_COUNTS.items():
        if key not in actual_counts:
            raise ValidationError(f"Missing expected finding row: {key}")
        actual = actual_counts[key]
        if actual != (expected_count, expected_denominator):
            raise ValidationError(
                f"Needs revision: Finding count changed for {key}: expected "
                f"{(expected_count, expected_denominator)}, found {actual}"
            )

    vpn_off = RecommendationRate("vpn", "search_off", "NordVPN", 63, 1200)
    vpn_on = RecommendationRate("vpn", "search_on", "NordVPN", 397, 1188)
    editors_off = RecommendationRate("editors", "search_off", "VS Code", 975, 1200)
    editors_on = RecommendationRate("editors", "search_on", "VS Code", 970, 1200)
    vpn_delta = percentage_point_change(vpn_on, vpn_off)
    editors_delta = percentage_point_change(editors_on, editors_off)

    if not math.isclose(vpn_delta, 28.16750841750842, abs_tol=1e-12):
        raise ValidationError(
            f"Needs revision: Unexpected VPN percentage-point change: {vpn_delta}"
        )
    if not math.isclose(editors_delta, -0.4166666666666572, abs_tol=1e-12):
        raise ValidationError(
            "Needs revision: Unexpected editor percentage-point change: " f"{editors_delta}"
        )

    return {
        "assessment": "Ready to share",
        "vpn_nordvpn_delta_pp": vpn_delta,
        "editors_vscode_delta_pp": editors_delta,
    }


def build_quality_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact source-quality and incomplete-cell tables for the notebook."""

    _require_columns(frame, REQUIRED_COLUMNS)
    unique_record_ids = int(frame["record_id"].nunique())
    search_aware_null_rows = int(
        frame.filter(like="search_aware__").isna().to_numpy().all(axis=1).sum()
    )
    summary = pd.DataFrame(
        {
            "check": [
                "Rows",
                "Columns",
                "Unique record_id",
                "Duplicate record_id",
                "Unique query_id",
                "search_aware null rows",
            ],
            "value": [
                len(frame),
                len(frame.columns),
                unique_record_ids,
                len(frame) - unique_record_ids,
                int(frame["query_id"].nunique()),
                search_aware_null_rows,
            ],
        }
    )

    grouped_sizes = cast(
        pd.Series,
        frame.groupby(["category", "model_id", "condition", "query_id"]).size(),
    )
    cell_counts = cast(pd.DataFrame, grouped_sizes.rename("rows").reset_index())
    incomplete = cell_counts.loc[
        cell_counts.loc[:, "rows"].to_numpy() < 30
    ].reset_index(drop=True)
    actual_incomplete = {
        (
            str(row["category"]),
            str(row["model_id"]),
            str(row["condition"]),
            str(row["query_id"]),
        ): int(row["rows"])
        for row in incomplete.to_dict(orient="records")
    }
    if actual_incomplete != KNOWN_INCOMPLETE_CELLS:
        raise ValidationError(f"Incomplete-cell profile changed: {actual_incomplete}")
    return summary, incomplete
