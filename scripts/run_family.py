"""Run the M0-M2 family on both tracks and write the comparison tables."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from modeling import evaluate, models, splits  # noqa: E402

OUT = Path("reports/modeling")
PAIRS = Path("data/processed/modeling")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for track_name in ("tr", "en"):
        fold_by_query = splits.load_frozen(track_name)
        pair_path = PAIRS / f"pairs_{track_name}.parquet"
        if not pair_path.exists():
            raise SystemExit(
                f"{pair_path} is missing. Run `make modeling-data` first -- building the "
                "pairs here would leave no saved table for the later stages to read."
            )
        table = pd.read_parquet(pair_path)

        for target in ("y_top", "y_mention"):
            scored = models.run_family(table, fold_by_query, target=target)
            scored.to_parquet(PAIRS / f"scored_{track_name}_{target}.parquet")
            for category in [None, *sorted(scored["category"].unique())]:
                subset = scored if category is None else scored[scored["category"] == category]
                if subset[target].sum() == 0:
                    continue
                for model in (*models.BASELINES, *models.MODEL_BLOCKS):
                    metrics = evaluate.summarise(
                        subset, f"score_{model}", target, with_ci=(category is not None)
                    )
                    rows.append(
                        {
                            "track": track_name,
                            "category": category or "ALL",
                            "target": target,
                            "model": model,
                            **metrics,
                        }
                    )
                print(f"  done {track_name}/{category or 'ALL'}/{target}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_family.csv", index=False)
    show = [
        "track",
        "category",
        "target",
        "model",
        "n_responses",
        "positive_rate",
        "pr_auc",
        "top1",
        "ndcg@3",
        "brier",
    ]
    print()
    print(results[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
