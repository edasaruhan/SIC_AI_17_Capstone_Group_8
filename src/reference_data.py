"""Prepare the pinned English brand-bias reference dataset for analysis."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from datasets import load_dataset

JSON_DICT_COLUMNS = ("core", "domain", "search_aware", "deterministic")
JSON_LIST_COLUMNS = ("tool_calls", "search_results")
JSON_COLUMNS = JSON_DICT_COLUMNS + JSON_LIST_COLUMNS

EXPECTED_SOURCE_COLUMNS = (
    "record_id",
    "experiment_id",
    "model_id",
    "provider",
    "condition",
    "category",
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
EXPECTED_CATEGORY_COUNTS = {
    "editors": 2400,
    "hosting": 2398,
    "travel": 2400,
    "vpn": 2388,
}
EXPECTED_CONDITION_COUNTS = {"search_off": 4800, "search_on": 4786}
EXPECTED_MODELS = {
    "claude-opus-4-6",
    "gpt-5.4",
    "grok-4.20-0309-reasoning",
    "zai-org/GLM-5",
}
EXPECTED_CONFIDENCE_COUNTS = {"high": 9576, "low": 3, "medium": 7}
KNOWN_INCOMPLETE_CELLS = {
    ("hosting", "zai-org/GLM-5", "search_on", "hosting_04"): 29,
    ("hosting", "zai-org/GLM-5", "search_on", "hosting_08"): 29,
    ("vpn", "zai-org/GLM-5", "search_on", "vpn_06"): 20,
    ("vpn", "zai-org/GLM-5", "search_on", "vpn_08"): 28,
}
BRAND_ALIASES = {
    "mullvad vpn": "Mullvad",
    "protonvpn": "Proton VPN",
    "visual studio code": "VS Code",
}


class DataQualityError(ValueError):
    """Raised when the pinned source no longer matches its expected quality profile."""


@dataclass(frozen=True)
class ReferenceDatasetConfig:
    """Configuration needed to load and materialize the reference dataset."""

    dataset_id: str
    config_name: str
    split: str
    revision: str
    expected_rows: int
    output_path: Path


def load_reference_config(path: str | Path) -> ReferenceDatasetConfig:
    """Load the reference dataset section from the project YAML configuration."""

    config_path = Path(path)
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        reference = content["reference_dataset"]
        return ReferenceDatasetConfig(
            dataset_id=str(reference["dataset_id"]),
            config_name=str(reference["config_name"]),
            split=str(reference["split"]),
            revision=str(reference["revision"]),
            expected_rows=int(reference["expected_rows"]),
            output_path=Path(reference["output_path"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid reference_dataset configuration in {config_path}") from error


def decode_json_value(
    value: Any,
    *,
    column: str,
    record_id: str,
    max_depth: int = 2,
) -> Any:
    """Decode normal or double-encoded JSON and retain an actual null value."""

    if value is None:
        return None

    decoded = value
    try:
        for _ in range(max_depth):
            if not isinstance(decoded, str):
                break
            decoded = json.loads(decoded)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Invalid JSON in {column} for record {record_id}") from error

    if isinstance(decoded, str):
        raise ValueError(
            f"JSON in {column} for record {record_id} remains encoded after {max_depth} passes"
        )
    return decoded


def parse_json_columns(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """Decode and type-check all JSON columns without dropping failed rows."""

    missing = set(JSON_COLUMNS).difference(frame.columns)
    if missing:
        raise DataQualityError(f"Missing JSON columns: {sorted(missing)}")
    if "record_id" not in frame:
        raise DataQualityError("Missing record_id column")

    parsed: dict[str, list[Any]] = {}
    for column in JSON_COLUMNS:
        values = [
            decode_json_value(value, column=column, record_id=str(record_id))
            for value, record_id in zip(frame[column], frame["record_id"], strict=True)
        ]
        expected_type = dict if column in JSON_DICT_COLUMNS else list
        invalid = [
            str(record_id)
            for value, record_id in zip(values, frame["record_id"], strict=True)
            if value is not None and not isinstance(value, expected_type)
        ]
        if invalid:
            preview = ", ".join(invalid[:3])
            raise DataQualityError(
                f"Unexpected parsed type in {column}; affected record(s): {preview}"
            )
        parsed[column] = values
    return parsed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_parquet_cell(value: Any) -> Any:
    """Convert nested values into stable Arrow-compatible scalar/list values."""

    if isinstance(value, list):
        return [item if isinstance(item, str) else _canonical_json(item) for item in value]
    if isinstance(value, dict):
        return _canonical_json(value)
    return value


def canonicalize_brand(value: Any) -> Any:
    """Normalize the documented aliases while preserving every other brand label."""

    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        raise TypeError(f"Brand value must be a string or null, got {type(value).__name__}")

    cleaned = " ".join(value.split())
    return BRAND_ALIASES.get(cleaned.casefold(), cleaned)


def flatten_reference_frame(
    frame: pd.DataFrame,
    parsed: dict[str, list[Any]],
) -> pd.DataFrame:
    """Create one analysis row per response with prefixed parsed feature columns."""

    parts = [frame.drop(columns=list(JSON_COLUMNS)).reset_index(drop=True)]
    for column in JSON_DICT_COLUMNS:
        values = [value if value is not None else {} for value in parsed[column]]
        normalized = pd.json_normalize(values).add_prefix(f"{column}__")
        for normalized_column in normalized.columns:
            if normalized[normalized_column].dtype == object:
                normalized[normalized_column] = normalized[normalized_column].map(
                    normalize_parquet_cell
                )
        parts.append(normalized)

    for column in JSON_LIST_COLUMNS:
        parts.append(
            pd.DataFrame(
                {f"{column}__items": [normalize_parquet_cell(value) for value in parsed[column]]}
            )
        )

    output = pd.concat(parts, axis=1)
    output["top_recommendation_canonical"] = output["core__top_recommendation"].map(
        canonicalize_brand
    )
    if not output.columns.is_unique:
        raise DataQualityError("Flattening produced duplicate column names")
    return output


def validate_source_frame(frame: pd.DataFrame, config: ReferenceDatasetConfig) -> None:
    """Fail when the pinned dataset differs from the documented source profile."""

    if tuple(frame.columns) != EXPECTED_SOURCE_COLUMNS:
        raise DataQualityError("Source columns or column order do not match the pinned schema")
    if len(frame) != config.expected_rows:
        raise DataQualityError(f"Expected {config.expected_rows} rows, found {len(frame)}")
    record_id_has_null = bool(frame["record_id"].isna().to_numpy().any())
    if record_id_has_null or not frame["record_id"].is_unique:
        raise DataQualityError("record_id must be non-null and unique")
    if frame["query_id"].nunique() != 40:
        raise DataQualityError(f"Expected 40 query IDs, found {frame['query_id'].nunique()}")
    if frame["category"].value_counts().sort_index().to_dict() != EXPECTED_CATEGORY_COUNTS:
        raise DataQualityError("Category counts do not match the pinned dataset")
    if frame["condition"].value_counts().sort_index().to_dict() != EXPECTED_CONDITION_COUNTS:
        raise DataQualityError("Condition counts do not match the pinned dataset")
    if set(frame["model_id"].unique()) != EXPECTED_MODELS:
        raise DataQualityError("Model identifiers do not match the pinned dataset")
    if set(frame["temperature"].unique()) != {0.7}:
        raise DataQualityError("All rows must use temperature 0.7")
    if (int(frame["run_index"].min()), int(frame["run_index"].max())) != (0, 29):
        raise DataQualityError("run_index must span 0 through 29")
    required_non_null = [column for column in frame.columns if column != "search_aware"]
    unexpected_nulls = frame[required_non_null].isna().sum()
    if unexpected_nulls.any():
        columns = unexpected_nulls[unexpected_nulls > 0].to_dict()
        raise DataQualityError(f"Unexpected null values: {columns}")

    cell_counts = frame.groupby(["category", "model_id", "condition", "query_id"], sort=True).size()
    if len(cell_counts) != 320:
        raise DataQualityError(f"Expected 320 experiment cells, found {len(cell_counts)}")
    incomplete = {index: int(count) for index, count in cell_counts.items() if count < 30}
    if incomplete != KNOWN_INCOMPLETE_CELLS:
        raise DataQualityError(f"Incomplete cells changed: {incomplete}")
    if any(count > 30 for count in cell_counts.tolist()):
        raise DataQualityError("At least one experiment cell contains more than 30 rows")


def validate_parsed_fields(frame: pd.DataFrame, parsed: dict[str, list[Any]]) -> dict[str, Any]:
    """Validate null patterns and judge confidence after JSON decoding."""

    search_aware = pd.Series(parsed["search_aware"], index=frame.index)
    expected_null = frame["condition"].eq("search_off")
    actual_null = search_aware.isna()
    if not actual_null.equals(expected_null):
        raise DataQualityError("search_aware must be null exactly for search_off rows")

    confidence = pd.Series(
        [value.get("confidence_in_extraction") for value in parsed["core"]],
        index=frame.index,
    )
    confidence_counts = confidence.value_counts().sort_index().to_dict()
    if confidence_counts != EXPECTED_CONFIDENCE_COUNTS:
        raise DataQualityError(f"Confidence counts changed: {confidence_counts}")

    return {
        "rows": len(frame),
        "columns_raw": len(frame.columns),
        "record_ids_unique": int(frame["record_id"].nunique()),
        "query_ids": int(frame["query_id"].nunique()),
        "json_decode_errors": 0,
        "category_counts": EXPECTED_CATEGORY_COUNTS,
        "condition_counts": EXPECTED_CONDITION_COUNTS,
        "confidence_counts": EXPECTED_CONFIDENCE_COUNTS,
        "known_incomplete_cells": {
            "|".join(key): value for key, value in KNOWN_INCOMPLETE_CELLS.items()
        },
    }


def prepare_reference_data(config: ReferenceDatasetConfig) -> dict[str, Any]:
    """Download, validate, flatten, and atomically materialize the reference data."""

    dataset = load_dataset(
        config.dataset_id,
        config.config_name,
        split=config.split,
        revision=config.revision,
    )
    source = cast(pd.DataFrame, dataset.to_pandas())
    validate_source_frame(source, config)
    parsed = parse_json_columns(source)
    summary = validate_parsed_fields(source, parsed)
    output = flatten_reference_frame(source, parsed)

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}-",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        output.to_parquet(temporary_path, index=False)
        round_trip = pd.read_parquet(temporary_path)
        if round_trip.shape != output.shape:
            raise DataQualityError(
                f"Parquet round trip changed shape from {output.shape} to {round_trip.shape}"
            )
        os.replace(temporary_path, output_path)
        output_path.chmod(0o644)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    summary.update(
        {
            "columns_output": len(output.columns),
            "dataset_id": config.dataset_id,
            "dataset_revision": config.revision,
            "output_path": str(output_path),
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Project YAML configuration (default: configs/base.yaml)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_reference_config(args.config)
    summary = prepare_reference_data(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
