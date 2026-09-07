"""Materialize the published Turkish brand-bias dataset for the modelling code.

`src/modeling/datasets.py` reads `data/interim/turkish_raw.parquet`. That file is
derived data and is not committed, so this script rebuilds it from the published
HuggingFace dataset -- the same role `reference_data.py` plays for the English
reference.

The collection pipeline in `src/bias_eval/` produces the dataset in the first
place and writes its own Parquet under `data/processed/`. This script is the
other direction: it pulls the published copy so a fresh clone can run the
modelling stage without holding any API keys.

The expected counts below are the frozen experiment design (2 domains x 5 queries
x 3 models x 2 conditions x 5 repetitions). A mismatch means the published
dataset moved and the modelling results are no longer comparable, so it fails
rather than proceeding quietly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

DATASET_ID = "furkankarli/turkish-brand-bias-evaluations"
CONFIG_NAME = "all"
SPLIT = "train"
OUTPUT_PATH = Path("data/interim/turkish_raw.parquet")

EXPECTED_ROWS = 300
EXPECTED_CATEGORY_COUNTS = {"cosmetics": 150, "vpn": 150}
EXPECTED_CONDITION_COUNTS = {"search_off": 150, "search_on": 150}
EXPECTED_MODELS = {
    "MiniMax-M2.7",
    "abliterated-model-large-v2",
    "gemini-3.5-flash-lite",
}
EXPECTED_COLUMNS = (
    "record_id",
    "experiment_id",
    "model_id",
    "provider",
    "condition",
    "category",
    "language",
    "query_id",
    "query_text",
    "run_index",
    "temperature",
    "final_response",
    "tool_calls",
    "search_results",
    "core",
    "domain",
    "search_aware",
    "deterministic",
    "judge_model",
    "judge_prompt_version",
)
JSON_COLUMNS = ("core", "domain", "search_aware", "deterministic", "tool_calls", "search_results")


class DataQualityError(ValueError):
    """Raised when the published dataset no longer matches the frozen design."""


def load_frame(revision: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, CONFIG_NAME, split=SPLIT, revision=revision)
    # to_pandas() is typed as possibly returning an iterator of chunks; a single
    # named split always comes back whole.
    return cast(pd.DataFrame, dataset.to_pandas())


def check(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail loudly on any drift from the design the published scores assume."""
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"Published dataset is missing columns: {missing}")

    if len(frame) != EXPECTED_ROWS:
        raise DataQualityError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")

    duplicates = int(len(frame) - frame["record_id"].nunique())
    if duplicates:
        raise DataQualityError(f"{duplicates} duplicate record_id values")

    categories = frame["category"].value_counts().to_dict()
    if categories != EXPECTED_CATEGORY_COUNTS:
        raise DataQualityError(f"Category counts changed: {categories}")

    conditions = frame["condition"].value_counts().to_dict()
    if conditions != EXPECTED_CONDITION_COUNTS:
        raise DataQualityError(f"Condition counts changed: {conditions}")

    models = set(frame["model_id"].unique())
    if models != EXPECTED_MODELS:
        raise DataQualityError(f"Generation models changed: {sorted(models)}")

    decode_errors = 0
    for column in JSON_COLUMNS:
        for value in frame[column]:
            if value is None:
                continue
            try:
                json.loads(value)
            except (TypeError, json.JSONDecodeError):
                decode_errors += 1
    if decode_errors:
        raise DataQualityError(f"{decode_errors} JSON fields failed to decode")

    cores = [json.loads(value) for value in frame["core"]]
    confidence = pd.Series([core["confidence_in_extraction"] for core in cores])
    return {
        "dataset_id": DATASET_ID,
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
        "categories": dict(sorted(categories.items())),
        "conditions": dict(sorted(conditions.items())),
        "models": sorted(models),
        "confidence_counts": dict(sorted(confidence.value_counts().to_dict().items())),
        "decided_responses": int(sum(1 for core in cores if core["top_recommendation"])),
        "json_decode_errors": decode_errors,
        "output_path": str(OUTPUT_PATH),
    }


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file so an interrupted run leaves no partial Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temporary, index=False)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin a dataset revision; omit to take the current published version.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    frame = load_frame(args.revision)
    report = check(frame)
    write_parquet(frame, args.output)
    report["output_path"] = str(args.output)
    report["dataset_revision"] = args.revision
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
