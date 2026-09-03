"""Atomic raw-result storage and resumability helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

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
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def is_complete(path: Path) -> bool:
    value = read_json(path)
    return bool(value and value.get("status") == "completed")


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
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
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


def status_summary(root: Path, expected: int) -> dict[str, Any]:
    values = iter_results(root)
    statuses = Counter(str(value.get("status", "malformed")) for value in values)
    return {
        "expected": expected,
        "files": len(values),
        "completed": statuses.get("completed", 0),
        "errors": statuses.get("error", 0),
        "remaining": max(0, expected - statuses.get("completed", 0)),
        "by_status": dict(sorted(statuses.items())),
    }
