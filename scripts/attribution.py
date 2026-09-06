"""M2 attribution and the Turkish visibility baseline.

Two outputs:

* **SHAP attribution** -- exact tree SHAP values via LightGBM's ``pred_contrib``,
  so no extra dependency is needed. This is what turns a score into "change this,
  expect roughly this much".
* **ASoV baseline** -- the share of responses naming each brand, split by
  retrieval condition. The retrieval-off column is the recognition measure; the
  gap between the two columns is what retrieval changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modeling import models  # noqa: E402
from modeling.datasets import load_track  # noqa: E402

PAIRS = Path("data/processed/modeling")
OUT = Path("reports/modeling")


def shap_table(track: str, target: str = "y_top") -> pd.DataFrame:
    table = pd.read_parquet(PAIRS / f"pairs_{track}.parquet")
    model, design, _ = models.fit_production(table, target=target)
    contributions = model.booster_.predict(design, pred_contrib=True)
    # The final column of pred_contrib is the expected value, not a feature.
    values = np.abs(contributions[:, :-1]).mean(axis=0)
    return (
        pd.DataFrame({"feature": design.columns, "mean_abs_shap": values})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .assign(track=track, target=target)
    )


def asov_table(track: str) -> pd.DataFrame:
    frame = load_track(track).frame
    rows = []
    for (category, condition), group in frame.groupby(["category", "condition"], sort=True):
        total = len(group)
        counts: dict[str, int] = {}
        wins: dict[str, int] = {}
        for _, row in group.iterrows():
            for brand in row["brands_mentioned"]:
                counts[brand] = counts.get(brand, 0) + 1
            if row["top_recommendation"]:
                wins[row["top_recommendation"]] = wins.get(row["top_recommendation"], 0) + 1
        for brand, count in counts.items():
            rows.append(
                {
                    "track": track,
                    "category": category,
                    "condition": condition,
                    "brand": brand,
                    "responses": total,
                    "mentions": count,
                    "asov": count / total,
                    "wins": wins.get(brand, 0),
                    "win_rate": wins.get(brand, 0) / total,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    shap_rows = pd.concat([shap_table(track) for track in ("en", "tr")], ignore_index=True)
    shap_rows.to_csv(OUT / "m2_shap.csv", index=False)
    print("Top M2 drivers (mean |SHAP|), target = top recommendation")
    for track in ("en", "tr"):
        top = shap_rows[shap_rows.track == track].head(8)
        print(f"\n  {track.upper()}")
        for _, row in top.iterrows():
            print(f"    {row.mean_abs_shap:8.4f}  {row.feature}")

    asov = pd.concat([asov_table(track) for track in ("en", "tr")], ignore_index=True)
    asov.to_csv(OUT / "asov_baseline.csv", index=False)

    print("\n\nTurkish VPN visibility baseline (share of responses naming the brand)")
    tr_vpn = asov[(asov.track == "tr") & (asov.category == "vpn")]
    wide = tr_vpn.pivot_table(index="brand", columns="condition", values="asov").fillna(0.0)
    wide["change"] = wide.get("search_on", 0) - wide.get("search_off", 0)
    wide = wide.sort_values("search_off", ascending=False)
    print(wide.round(3).to_string())

    print("\n\nEnglish VPN visibility baseline, same measure")
    en_vpn = asov[(asov.track == "en") & (asov.category == "vpn")]
    wide_en = en_vpn.pivot_table(index="brand", columns="condition", values="asov").fillna(0.0)
    wide_en["change"] = wide_en.get("search_on", 0) - wide_en.get("search_off", 0)
    print(wide_en.sort_values("search_off", ascending=False).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
