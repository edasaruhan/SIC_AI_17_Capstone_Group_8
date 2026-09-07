"""Train M3 on the VPN domain in both languages, named and masked.

VPN is the only category both corpora cover, so it is the one place the two
tracks can be set side by side. Each run reuses the frozen folds from M0-M2, so
the resulting scores drop straight into the same comparison table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from modeling import cross_encoder, evaluate, pairs, splits  # noqa: E402
from modeling.datasets import load_track  # noqa: E402

PAIRS = Path("data/processed/modeling")
OUT = Path("reports/modeling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["tr", "en"])
    parser.add_argument("--category", default="vpn")
    parser.add_argument("--target", default="y_top")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    rows = []
    for track_name in args.tracks:
        track = load_track(track_name)
        # Snippets must be built from the same frame the pairs were built from,
        # or the candidate universe narrows and rows fail to line up.
        full_frame = track.frame
        frame = full_frame[full_frame["category"] == args.category].copy()
        scored = pd.read_parquet(PAIRS / f"scored_{track_name}_{args.target}.parquet")
        scored = scored[
            (scored["category"] == args.category) & (scored["condition"] == "search_on")
        ].copy()
        scored = scored[scored["record_id"].isin(set(frame["record_id"]))]
        fold_by_query = splits.load_frozen(track_name)
        config = cross_encoder.EncoderConfig(
            model_name=cross_encoder.DEFAULT_MODELS[track_name],
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(
            f"{track_name}/{args.category}: {scored['record_id'].nunique()} responses, "
            f"{len(scored)} pairs, model={config.model_name}",
            flush=True,
        )

        for masked in (False, True):
            variant = "masked" if masked else "named"
            snippets = pairs.brand_snippets(frame, mask=masked)
            prepared = cross_encoder.build_inputs(scored, snippets, masked=masked)
            started = time.time()
            result = cross_encoder.run_variant(prepared, fold_by_query, args.target, config)
            elapsed = round(time.time() - started)
            result.to_parquet(
                PAIRS / f"m3_{track_name}_{args.category}_{args.target}_{variant}.parquet"
            )
            metrics = evaluate.summarise(result, "score_M3", args.target)
            rows.append(
                {
                    "seed": args.seed,
                    "track": track_name,
                    "category": args.category,
                    "target": args.target,
                    "model": f"M3_{variant}",
                    "encoder": config.model_name,
                    "seconds": elapsed,
                    **metrics,
                }
            )
            print(
                f"  {track_name}/{variant}: PR-AUC {metrics['pr_auc']:.4f} "
                f"top1 {metrics['top1']:.4f} ({elapsed}s)",
                flush=True,
            )

    table = pd.DataFrame(rows)
    path = OUT / f"m3_{args.category}_{args.target}{args.tag}.csv"
    table.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(
        table[["track", "model", "pr_auc", "top1", "ndcg@3", "brier"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
