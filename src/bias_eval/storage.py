"""Atomic raw-result storage and resumability helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .logging import logger
from .records import CellSpec


def result_path(root: Path, cell: CellSpec) -> Path:
    return (
        root
        / cell.experiment.experiment_id
        / cell.safe_model_id
        / cell.condition
        / f"{cell.query.id}_{cell.run_index:03d}.json"
    )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("json_read_failed path={} error_type={}", path, type(error).__name__)
        return None
    return value if isinstance(value, dict) else None


def record_is_complete(value: dict[str, Any] | None) -> bool:
    if not value or value.get("status") != "completed":
        return False
    response = value.get("final_response")
    return isinstance(response, str) and bool(response.strip())


def is_complete(path: Path) -> bool:
    return record_is_complete(read_json(path))


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
        logger.debug(
            "json_write_complete path={} status={} record_id={}",
            path,
            value.get("status"),
            value.get("record_id"),
        )
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        logger.error("json_write_failed path={}", path)
        raise


def iter_results(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not root.exists():
        return values
    for path in sorted(root.glob("*/*/*/*.json")):
        value = read_json(path)
        if value is not None:
            values.append(value)
    return values


def status_summary(
    root: Path,
    expected: int,
    planned_record_ids: set[str] | None = None,
) -> dict[str, Any]:
    all_values = iter_results(root)
    values = (
        all_values
        if planned_record_ids is None
        else [value for value in all_values if value.get("record_id") in planned_record_ids]
    )
    statuses: Counter[str] = Counter()
    for value in values:
        status = str(value.get("status", "malformed"))
        if status == "completed" and not record_is_complete(value):
            statuses["invalid"] += 1
        else:
            statuses[status] += 1
    result = {
        "expected": expected,
        "files": len(values),
        "superseded_files": len(all_values) - len(values),
        "completed": statuses.get("completed", 0),
        "errors": statuses.get("error", 0),
        "remaining": max(0, expected - statuses.get("completed", 0)),
        "by_status": dict(sorted(statuses.items())),
    }
    logger.info("generation_status summary={}", result)
    return result
