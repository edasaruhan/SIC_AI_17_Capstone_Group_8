"""Fail-closed validation for the complete 400-row collection."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pyarrow.parquet as pq

from .config import SuiteConfig
from .exporter import EXPORT_COLUMNS
from .judge import judged_path, judgment_is_current
from .records import plan_cells
from .storage import iter_results


def _expected_counts(config: SuiteConfig) -> dict[str, dict[str, int]]:
    cells = plan_cells(config)
    return {
        "experiment_id": dict(Counter(cell.experiment.experiment_id for cell in cells)),
        "model_id": dict(Counter(cell.model.model_id for cell in cells)),
        "condition": dict(Counter(cell.condition for cell in cells)),
        "category": dict(Counter(cell.experiment.category for cell in cells)),
        "query_id": dict(Counter(cell.query.id for cell in cells)),
    }


def validate_dataset(config: SuiteConfig) -> dict[str, Any]:
    issues: list[str] = []
    raw_records = iter_results(config.raw_dir)
    completed = [item for item in raw_records if item.get("status") == "completed"]
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    completed_ids = [str(item.get("record_id")) for item in completed]
    if len(completed) != config.expected_rows:
        issues.append(f"generation: expected {config.expected_rows}, found {len(completed)}")
    duplicates = sorted(key for key, count in Counter(completed_ids).items() if count > 1)
    if duplicates:
        issues.append(f"generation: duplicate record_ids: {duplicates[:5]}")
    missing = sorted(planned_ids - set(completed_ids))
    unexpected = sorted(set(completed_ids) - planned_ids)
    if missing:
        issues.append(f"generation: {len(missing)} planned cells missing")
    if unexpected:
        issues.append(f"generation: {len(unexpected)} unexpected cells")

    for raw in completed:
        condition = raw.get("condition")
        if condition == "search_off" and (raw.get("tool_calls") or raw.get("search_results")):
            issues.append(f"generation: search_off trace is not empty for {raw.get('record_id')}")
        if not judgment_is_current(judged_path(config, raw), raw):
            issues.append(f"judge: missing or stale result for {raw.get('record_id')}")

    expected = _expected_counts(config)
    for key, counts in expected.items():
        actual = dict(Counter(str(item.get(key)) for item in completed))
        if actual != counts:
            issues.append(f"generation: wrong {key} distribution")

    parquet_path = config.processed_dir / "all" / "train.parquet"
    parquet_rows = 0
    if not parquet_path.exists():
        issues.append("export: all/train.parquet is missing")
    else:
        table = pq.read_table(parquet_path)
        parquet_rows = table.num_rows
        if table.column_names != EXPORT_COLUMNS:
            issues.append("export: column order/schema is not reference-compatible")
        if parquet_rows != config.expected_rows:
            issues.append(f"export: expected {config.expected_rows}, found {parquet_rows}")
        frame = table.to_pylist()
        if len({row["record_id"] for row in frame}) != len(frame):
            issues.append("export: duplicate record_ids")
        for row in frame:
            for key in (
                "tool_calls",
                "search_results",
                "core",
                "domain",
                "deterministic",
            ):
                try:
                    json.loads(row[key])
                except (TypeError, json.JSONDecodeError):
                    issues.append(f"export: invalid JSON in {key} for {row['record_id']}")
            if row["condition"] == "search_off" and row["search_aware"] is not None:
                issues.append(f"export: search_off search_aware not null for {row['record_id']}")
            if row["condition"] == "search_on" and row["search_aware"] is None:
                issues.append(f"export: search_on search_aware is null for {row['record_id']}")

    return {
        "ok": not issues,
        "expected_rows": config.expected_rows,
        "completed_generation_rows": len(completed),
        "current_judgment_rows": sum(
            judgment_is_current(judged_path(config, raw), raw) for raw in completed
        ),
        "export_rows": parquet_rows,
        "issues": issues,
    }


def validation_exit_code(result: dict[str, Any]) -> int:
    return 0 if result["ok"] else 1
