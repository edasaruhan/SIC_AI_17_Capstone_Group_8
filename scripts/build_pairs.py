"""Turn the two corpora into the tables every later stage reads.

Run once after `make reference-data` and `make turkish-data`. Three artifacts per
track, all under `data/processed/modeling/`:

``splits_<track>.json``   query-level fold assignment, frozen and committed
``pairs_<track>.parquet`` one row per (response, candidate brand) with the model
                          features -- what M0-M3 train on
``evidence_<track>.parquet`` one row per (response, brand, retrieved result) with
                          the search query, round, rank, URL, domain, source type
                          and snippet kept intact -- what a recommendation cites

The split is written first and reused if already present, so re-running this
never silently reshuffles folds under results that were scored on the old ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from modeling import pairs, splits  # noqa: E402
from modeling.datasets import load_track  # noqa: E402

OUT = Path("data/processed/modeling")
TRACKS = ("en", "tr")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for track_name in TRACKS:
        track = load_track(track_name)
        frame = track.frame

        split_path = OUT / f"splits_{track_name}.json"
        if split_path.exists():
            fold_by_query = splits.load_frozen(track_name)
            unmapped = sorted(set(frame["query_id"].astype(str)) - set(fold_by_query))
            if unmapped:
                raise SystemExit(
                    f"{split_path} predates these queries: {unmapped}. Delete it to refreeze, "
                    "but every score recorded against the old split then has to be rerun."
                )
            print(f"{track_name}: reusing frozen split at {split_path}")
        else:
            n_splits = splits.effective_n_splits(frame, 5)
            fold_by_query = splits.assign_folds(frame, n_splits=n_splits)
            if splits.leaks(frame, fold_by_query):
                raise SystemExit(f"{track_name}: fold assignment leaks a query across folds")
            splits.freeze(track_name, fold_by_query, frame)
            print(f"{track_name}: froze {n_splits} folds at {split_path}")

        pair_table = pairs.build(frame)
        pair_path = OUT / f"pairs_{track_name}.parquet"
        pair_table.to_parquet(pair_path, index=False)

        evidence = pairs.build_evidence(frame)
        evidence_path = OUT / f"evidence_{track_name}.parquet"
        evidence.to_parquet(evidence_path, index=False)

        summary.append(
            {
                "track": track_name,
                "responses": len(frame),
                "pairs": len(pair_table),
                "evidence_rows": len(evidence),
                "brands": pair_table["brand"].nunique(),
                "urls": evidence["link"].nunique(),
            }
        )
        print(f"  {pair_path}      {len(pair_table):>7,} rows")
        print(f"  {evidence_path}  {len(evidence):>7,} rows")

    print()
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
