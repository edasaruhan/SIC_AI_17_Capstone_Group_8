"""The head-to-head table: every model on VPN, in both languages.

M3 only sees retrieval-on responses, because masking needs snippets to mask. To
keep the comparison honest, M0-M2 are re-scored on exactly the rows M3 saw
rather than on their own larger evaluation set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from modeling import evaluate, models  # noqa: E402

PAIRS = Path("data/processed/modeling")
OUT = Path("reports/modeling")
TARGET = "y_top"
CATEGORY = "vpn"


def main() -> None:
    rows = []
    for track in ("en", "tr"):
        scored = pd.read_parquet(PAIRS / f"scored_{track}_{TARGET}.parquet")
        subset = scored[
            (scored["category"] == CATEGORY) & (scored["condition"] == "search_on")
        ].copy()
        for model in (*models.BASELINES, *models.MODEL_BLOCKS):
            rows.append(
                {
                    "track": track,
                    "model": model,
                    **evaluate.summarise(subset, f"score_{model}", TARGET),
                }
            )
        for variant in ("named", "masked"):
            # Prefer the pinned seed-42 copy: replication runs overwrite the plain
            # filename, so reading it would mix seeds into one table.
            pinned = PAIRS / f"m3_{track}_{CATEGORY}_{TARGET}_{variant}_seed42.parquet"
            plain = PAIRS / f"m3_{track}_{CATEGORY}_{TARGET}_{variant}.parquet"
            path = pinned if pinned.exists() else plain
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            rows.append(
                {
                    "track": track,
                    "model": f"M3_{variant}",
                    **evaluate.summarise(frame, "score_M3", TARGET),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "vpn_head_to_head.csv", index=False)

    show = [
        "track",
        "model",
        "n_responses",
        "positive_rate",
        "pr_auc",
        "pr_auc_lo",
        "pr_auc_hi",
        "top1",
        "ndcg@3",
        "brier",
    ]
    print("VPN, retrieval-on responses only, target = single top recommendation")
    print(table[show].round(4).to_string(index=False))

    print("\nMasking ablation (named minus masked):")
    for track in ("en", "tr"):
        named = table[(table.track == track) & (table.model == "M3_named")]
        masked = table[(table.track == track) & (table.model == "M3_masked")]
        if named.empty or masked.empty:
            continue
        gap = float(named.pr_auc.iloc[0]) - float(masked.pr_auc.iloc[0])
        retained = float(masked.pr_auc.iloc[0]) / float(named.pr_auc.iloc[0])
        print(
            f"  {track}: PR-AUC {named.pr_auc.iloc[0]:.4f} -> {masked.pr_auc.iloc[0]:.4f}"
            f"  (gap {gap:+.4f}, {retained:.0%} retained without brand names)"
        )


if __name__ == "__main__":
    main()
