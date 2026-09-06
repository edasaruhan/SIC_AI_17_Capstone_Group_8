"""Recompute the metric tables from saved out-of-fold scores, without refitting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from modeling import evaluate, models  # noqa: E402

PAIRS = Path("data/processed/modeling")
OUT = Path("reports/modeling")
MODEL_ORDER = [*models.BASELINES, *models.MODEL_BLOCKS]


def main() -> None:
    rows = []
    for track in ("tr", "en"):
        for target in ("y_top", "y_mention"):
            path = PAIRS / f"scored_{track}_{target}.parquet"
            if not path.exists():
                continue
            scored = pd.read_parquet(path)
            for category in [None, *sorted(scored["category"].unique())]:
                subset = scored if category is None else scored[scored["category"] == category]
                if subset[target].sum() == 0:
                    continue
                for model in MODEL_ORDER:
                    rows.append(
                        {
                            "track": track,
                            "category": category or "ALL",
                            "target": target,
                            "model": model,
                            **evaluate.summarise(
                                subset, f"score_{model}", target, with_ci=(category is not None)
                            ),
                        }
                    )
                print(f"  scored {track}/{category or 'ALL'}/{target}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_family.csv", index=False)
    print(f"\nwrote {OUT / 'model_family.csv'} ({len(results)} rows)")


if __name__ == "__main__":
    main()
