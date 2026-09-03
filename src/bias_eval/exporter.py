"""Reference-compatible Parquet export and provenance manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .config import SuiteConfig
from .judge import judged_path, judgment_is_current
from .logging import logger
from .normalization import alias_candidates
from .provenance import collection_fingerprint
from .records import plan_cells
from .storage import atomic_write_json, iter_results, read_json, record_is_complete

REFERENCE_GIT_COMMIT = "cc42677a42bbbf92f6ef4c376abda6528f0463ea"
REFERENCE_HF_REVISION = "400da04eced51d3afe52b6d20c0207fd613f8a4a"
EXPORT_COLUMNS = [
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
]

EXPORT_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("query_text", pa.string(), nullable=False),
        pa.field("run_index", pa.int64(), nullable=False),
        pa.field("temperature", pa.float64(), nullable=False),
        pa.field("final_response", pa.string(), nullable=False),
        pa.field("tool_calls", pa.string(), nullable=False),
        pa.field("search_results", pa.string(), nullable=False),
        pa.field("core", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("search_aware", pa.string(), nullable=True),
        pa.field("deterministic", pa.string(), nullable=False),
        pa.field("judge_model", pa.string(), nullable=False),
        pa.field("judge_prompt_version", pa.string(), nullable=False),
    ]
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _export_search_results(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for item in raw_results:
        body = item.get("results") or {}
        organic = []
        for result in body.get("organic") or []:
            organic.append(
                {
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    "position": result.get("position"),
                }
            )
        exported.append(
            {"query": item.get("query"), "round": item.get("round"), "organic": organic}
        )
    return exported


def _export_tool_calls(raw_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "round": item.get("round"),
            "function": item.get("function"),
            "query": (item.get("arguments") or {}).get("query"),
        }
        for item in raw_calls
    ]


def build_export_rows(config: SuiteConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    judged_records: list[dict[str, Any]] = []
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    for raw in iter_results(config.raw_dir):
        if not record_is_complete(raw) or raw.get("record_id") not in planned_ids:
            continue
        path = judged_path(config, raw)
        if not judgment_is_current(path, raw):
            continue
        judged = read_json(path)
        if judged is None:
            continue
        value = judged["judgment"]
        judged_records.append({**judged, "category": raw["category"]})
        rows.append(
            {
                "record_id": raw["record_id"],
                "experiment_id": raw["experiment_id"],
                "model_id": raw["model_id"],
                "provider": raw["provider"],
                "condition": raw["condition"],
                "category": raw["category"],
                "language": raw["language"],
                "query_id": raw["query_id"],
                "query_text": raw["query_text"],
                "run_index": int(raw["run_index"]),
                "temperature": float(raw["temperature"]),
                "final_response": str(raw["final_response"]),
                "tool_calls": _json(_export_tool_calls(raw.get("tool_calls") or [])),
                "search_results": _json(_export_search_results(raw.get("search_results") or [])),
                "core": _json(value["core"]),
                "domain": _json(value["domain"]),
                "search_aware": (
                    None if value["search_aware"] is None else _json(value["search_aware"])
                ),
                "deterministic": _json(judged["deterministic"]),
                "judge_model": judged["judge_model"],
                "judge_prompt_version": judged["judge_prompt_version"],
            }
        )
    return sorted(rows, key=lambda item: item["record_id"]), judged_records


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=EXPORT_SCHEMA)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hashes(config: SuiteConfig) -> dict[str, Any]:
    config_dir = Path("configs/evaluation")
    config_files = sorted(config_dir.glob("*.yaml"))
    query_files = sorted(config_dir.glob("queries_*.yaml"))
    return {
        "config_sha256": hashlib.sha256(
            b"".join(path.read_bytes() for path in config_files)
        ).hexdigest(),
        "query_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in query_files
        },
    }


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def export_dataset(config: SuiteConfig, *, require_complete: bool = True) -> dict[str, Any]:
    logger.info(
        "export_started expected_rows={} processed_dir={} require_complete={}",
        config.expected_rows,
        config.processed_dir,
        require_complete,
    )
    rows, judged_records = build_export_rows(config)
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    if require_complete and len(rows) != config.expected_rows:
        raise ValueError(
            f"Export requires {config.expected_rows} current judged rows; found {len(rows)}"
        )
    output_paths = {
        "vpn": config.processed_dir / "vpn" / "train.parquet",
        "cosmetics": config.processed_dir / "cosmetics" / "train.parquet",
        "all": config.processed_dir / "all" / "train.parquet",
    }
    for category in ("vpn", "cosmetics"):
        category_rows = [row for row in rows if row["category"] == category]
        _write_parquet(output_paths[category], category_rows)
        logger.info(
            "parquet_written category={} rows={} path={}",
            category,
            len(category_rows),
            output_paths[category],
        )
    _write_parquet(output_paths["all"], rows)
    logger.info("parquet_written category=all rows={} path={}", len(rows), output_paths["all"])

    raw_records = [
        item for item in iter_results(config.raw_dir) if item.get("record_id") in planned_ids
    ]
    complete_raw = [item for item in raw_records if item.get("status") == "completed"]
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    complete_ids = {str(item.get("record_id")) for item in complete_raw}
    collection_times = [
        value
        for item in complete_raw
        for value in (
            item.get("metadata", {}).get("started_at"),
            item.get("metadata", {}).get("completed_at"),
        )
        if value
    ]
    manifest = {
        "suite_id": config.suite_id,
        "collection_fingerprint": collection_fingerprint(config),
        "expected_generation_rows": config.expected_rows,
        "completed_generation_rows": len(complete_raw),
        "current_judged_rows": len(rows),
        "missing_generation_record_ids": sorted(planned_ids - complete_ids),
        "error_generation_record_ids": sorted(
            str(item.get("record_id")) for item in raw_records if item.get("status") == "error"
        ),
        "distributions": {
            key: _distribution(rows, key)
            for key in ("model_id", "category", "condition", "query_id")
        },
        **_config_hashes(config),
        "reference_github_commit": REFERENCE_GIT_COMMIT,
        "reference_huggingface_revision": REFERENCE_HF_REVISION,
        "parquet_sha256": {
            str(path.relative_to(config.processed_dir)): _sha256(path)
            for path in output_paths.values()
        },
        "collection_started_at": min(collection_times) if collection_times else None,
        "collection_completed_at": max(collection_times) if collection_times else None,
        "judge_prompt_hashes": sorted(
            {str(item.get("judge_prompt_hash")) for item in judged_records}
        ),
        "columns": EXPORT_COLUMNS,
    }
    atomic_write_json(config.processed_dir / "manifest.json", manifest)
    aliases = {
        "policy": "Candidates are never merged automatically; review them manually.",
        "candidates": alias_candidates(judged_records),
    }
    atomic_write_json(config.processed_dir / "alias_candidates.json", aliases)
    logger.info(
        "export_completed rows={} manifest={} aliases={}",
        len(rows),
        config.processed_dir / "manifest.json",
        len(aliases["candidates"]),
    )
    return manifest
