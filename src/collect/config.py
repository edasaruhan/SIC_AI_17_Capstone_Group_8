"""Runtime settings for a collection run, loaded from ``configs/collect.yaml``.

This file owns *how* the assistants are called - endpoints, pacing, retries,
spend cap. *What* is asked lives in the shared design files (``queries_tr``,
``brands_tr``, ``variants``, ``protocols``), which stay frozen after Gate 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .providers import ModelConfig


class ConfigError(ValueError):
    """Raised when a configuration file cannot be used as written."""


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff bounds for transient provider failures."""

    max_attempts: int = 5
    initial_seconds: float = 1.0
    max_seconds: float = 60.0
    jitter_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigError("retry.max_attempts must be at least 1")
        if self.initial_seconds <= 0 or self.max_seconds < self.initial_seconds:
            raise ConfigError("retry backoff bounds are inconsistent")


@dataclass(frozen=True)
class CollectionSettings:
    """Everything the runner needs that is not part of the experiment design."""

    raw_dir: Path
    models: tuple[ModelConfig, ...]
    concurrency: int = 4
    daily_spend_cap_usd: float = 20.0
    store_raw_response: bool = True
    seed: int = 42
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if not self.models:
            raise ConfigError("At least one model must be configured")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be at least 1")
        if self.daily_spend_cap_usd <= 0:
            raise ConfigError("daily_spend_cap_usd must be positive")
        keys = [model.key for model in self.models]
        if len(set(keys)) != len(keys):
            raise ConfigError(f"Duplicate model key in configuration: {sorted(keys)}")

    def model(self, key: str) -> ModelConfig:
        for candidate in self.models:
            if candidate.key == key:
                return candidate
        raise ConfigError(f"Unknown model key {key!r}; configured: {[m.key for m in self.models]}")


def read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}") from error
    if not isinstance(content, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")
    return content


def _model_from_mapping(key: str, values: dict[str, Any]) -> ModelConfig:
    allowed = set(ModelConfig.__dataclass_fields__) - {"key"}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"Model {key}: unknown setting(s) {sorted(unknown)}")
    try:
        return ModelConfig(key=key, **values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Model {key}: {error}") from error


def load_collection_settings(path: str | Path) -> CollectionSettings:
    """Read the collection runtime configuration."""

    content = read_yaml(path)
    section = content.get("collection")
    if not isinstance(section, dict):
        raise ConfigError(f"{path} must contain a 'collection' mapping")

    raw_models = section.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError(f"{path}: collection.models must be a non-empty mapping")

    models = tuple(
        _model_from_mapping(str(key), dict(values or {})) for key, values in raw_models.items()
    )
    retry_values = dict(section.get("retry") or {})
    try:
        retry = RetryPolicy(**retry_values)
    except TypeError as error:
        raise ConfigError(f"{path}: invalid retry settings") from error

    return CollectionSettings(
        raw_dir=Path(section.get("raw_dir", "data/raw")),
        models=models,
        concurrency=int(section.get("concurrency", 4)),
        daily_spend_cap_usd=float(section.get("daily_spend_cap_usd", 20.0)),
        store_raw_response=bool(section.get("store_raw_response", True)),
        seed=int(section.get("seed", 42)),
        retry=retry,
    )
