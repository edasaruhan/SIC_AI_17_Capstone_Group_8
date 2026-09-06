"""Collection identity locks that prevent mixed experiment configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import SuiteConfig
from .storage import atomic_write_json, read_json

COLLECTION_LOCK_VERSION = 1
COLLECTION_LOCK_NAME = ".collection-lock.json"


def collection_spec(config: SuiteConfig) -> dict[str, Any]:
    """Return all generation-affecting settings in a stable representation."""

    return {
        "version": COLLECTION_LOCK_VERSION,
        "suite_id": config.suite_id,
        "n_runs": config.n_runs,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "conditions": list(config.conditions),
        "max_tool_rounds": config.max_tool_rounds,
        "system_prompts": config.system_prompts,
        "models": [asdict(model) for model in config.models],
        "experiments": [
            {
                "experiment_id": experiment.experiment_id,
                "category": experiment.category,
                "language": experiment.language,
                "queries": [asdict(query) for query in experiment.queries],
            }
            for experiment in config.experiments
        ],
        "search": asdict(config.search),
    }


def collection_fingerprint(config: SuiteConfig) -> str:
    encoded = json.dumps(
        collection_spec(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collection_lock_path(config: SuiteConfig) -> Path:
    return config.raw_dir / COLLECTION_LOCK_NAME


def collection_lock_issue(config: SuiteConfig) -> str | None:
    """Describe a missing or incompatible lock without changing the filesystem."""

    path = collection_lock_path(config)
    if not path.exists():
        return "collection lock is missing"
    lock = read_json(path)
    if lock is None:
        return "collection lock is unreadable"
    if lock.get("fingerprint") != collection_fingerprint(config):
        return "collection lock does not match the active generation configuration"
    return None


def ensure_collection_lock(config: SuiteConfig) -> Path:
    """Create the lock once, then fail closed if generation settings change."""

    path = collection_lock_path(config)
    if path.exists():
        issue = collection_lock_issue(config)
        if issue:
            raise ValueError(
                f"{issue}. Start a new suite with new raw/processed paths; do not mix releases."
            )
        return path
    atomic_write_json(
        path,
        {
            "version": COLLECTION_LOCK_VERSION,
            "suite_id": config.suite_id,
            "fingerprint": collection_fingerprint(config),
            "spec": collection_spec(config),
        },
    )
    return path


__all__ = [
    "COLLECTION_LOCK_NAME",
    "collection_fingerprint",
    "collection_lock_issue",
    "collection_lock_path",
    "collection_spec",
    "ensure_collection_lock",
]
