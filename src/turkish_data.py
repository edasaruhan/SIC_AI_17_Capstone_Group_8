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
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, cast

import pandas as pd

DATASET_ID = "furkankarli/turkish-brand-bias-evaluations"
CONFIG_NAME = "all"
SPLIT = "train"
OUTPUT_PATH = Path("data/interim/turkish_raw.parquet")
REVISION = "4d274b954be7d9b0abbfe3314b2cbc387dfd800f"
SOURCE_SHA256 = "bd7319afb3f717449b8fe0baa28fe8a4008da15f59e3af1a6b2cf2bde2801a71"
SOURCE_FILE = "all/train.parquet"
CONTENT_SHA256 = "1b77bc902487212e9be13aa846a16d27ae20d987e54f257252716faf4d43a9e4"


def content_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values("record_id").reindex(sorted(frame.columns), axis=1)
    return sha256(
        str(canonical.to_json(orient="records", force_ascii=False, double_precision=15)).encode()
    ).hexdigest()


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


def load_frame(revision: str = REVISION) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    if revision != REVISION:
        raise DataQualityError("New releases require a new revision/hash contract and output path")
    path = Path(hf_hub_download(DATASET_ID, SOURCE_FILE, repo_type="dataset", revision=revision))
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise DataQualityError("Published source checksum does not match the pinned release")
    return pd.read_parquet(path)


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

    required = [c for c in EXPECTED_COLUMNS if c != "search_aware"]
    if pd.DataFrame(frame[required]).isna().to_numpy().any():
        raise DataQualityError("Null required field")
    if not all(value == "tr" for value in frame["language"]):
        raise DataQualityError("Every response must have language=tr")
    for column in ("record_id", "query_text", "final_response"):
        if not all(isinstance(value, str) and bool(value.strip()) for value in frame[column]):
            raise DataQualityError(f"Empty or non-text {column}")
    expected = {
        (category, model, condition, f"{category}_tr_{query:02}", run)
        for category, model, condition, query, run in product(
            EXPECTED_CATEGORY_COUNTS,
            EXPECTED_MODELS,
            EXPECTED_CONDITION_COUNTS,
            range(1, 6),
            range(5),
        )
    }
    keys = ["category", "model_id", "condition", "query_id", "run_index"]
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in frame["run_index"]):
        raise DataQualityError("run_index must be an integer")
    actual = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or actual != expected:
        raise DataQualityError("Experiment cells differ from the frozen 2x5x3x2x5 design")
    if not all(e == c + "_tr_v1" for e, c in zip(frame.experiment_id, frame.category, strict=True)):
        raise DataQualityError("Wrong experiment_id/category mapping")
    import yaml

    for category in EXPECTED_CATEGORY_COUNTS:
        query_path = (
            Path(__file__).resolve().parents[1] / f"configs/evaluation/queries_{category}_tr.yaml"
        )
        queries = yaml.safe_load(query_path.read_text(encoding="utf-8"))["queries"]
        expected_text = {q["id"]: q["text"] for q in queries}
        subset = cast(pd.DataFrame, frame[frame["category"] == category])
        if not all(
            text == expected_text.get(query)
            for text, query in zip(subset.query_text, subset.query_id, strict=True)
        ):
            raise DataQualityError("Query text differs from the frozen prompts")

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
                decoded = json.loads(value)
                expected_type = list if column in ("tool_calls", "search_results") else dict
                if decoded is None and column == "search_aware":
                    continue
                if not isinstance(decoded, expected_type):
                    decode_errors += 1
            except (TypeError, json.JSONDecodeError):
                decode_errors += 1
    if decode_errors:
        raise DataQualityError(f"{decode_errors} JSON fields failed to decode")

    cores = [json.loads(value) for value in frame["core"]]
    if any("confidence_in_extraction" not in c or "top_recommendation" not in c for c in cores):
        raise DataQualityError("Missing core extraction fields")
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
        default=REVISION,
        choices=[REVISION],
        help="Frozen published revision; a new release needs a new contract.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    frame = load_frame(args.revision)
    report = check(frame)
    if args.output.exists():
        existing = pd.read_parquet(args.output)
        if not existing.equals(frame):
            raise DataQualityError("Refusing to replace a different existing dataset")
    write_parquet(frame, args.output)
    report["output_path"] = str(args.output)
    report["dataset_revision"] = args.revision
    report["source_file"] = SOURCE_FILE
    report["source_sha256"] = SOURCE_SHA256
    report["output_sha256"] = sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
