"""Typed configuration for the Turkish evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a suite file is incomplete or inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_id: str
    provider: str
    api_base: str
    api_key_env: str
    max_concurrent: int = 1


@dataclass(frozen=True)
class QueryConfig:
    id: str
    text: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    category: str
    language: str
    queries: tuple[QueryConfig, ...]


@dataclass(frozen=True)
class JudgeConfig:
    model_id: str
    provider: str
    api_base: str
    api_key_env: str
    temperature: float
    reasoning_effort: str
    max_concurrent: int


@dataclass(frozen=True)
class SearchConfig:
    api_base: str
    api_key_env: str
    num_results: int
    country: str
    language: str
    location: str
    max_concurrent: int


@dataclass(frozen=True)
class SuiteConfig:
    suite_id: str
    n_runs: int
    temperature: float
    max_output_tokens: int
    conditions: tuple[str, ...]
    max_tool_rounds: int
    system_prompts: dict[str, str | None]
    models: tuple[ModelConfig, ...]
    experiments: tuple[ExperimentConfig, ...]
    judge: JudgeConfig
    search: SearchConfig
    raw_dir: Path
    judged_dir: Path
    search_cache_dir: Path
    processed_dir: Path

    @property
    def expected_rows(self) -> int:
        return (
            sum(len(exp.queries) for exp in self.experiments)
            * len(self.models)
            * len(self.conditions)
            * self.n_runs
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML: {path}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_suite(path: str | Path = "configs/evaluation/suite.yaml") -> SuiteConfig:
    """Load the suite and its referenced model, query, and judge files."""

    suite_path = Path(path)
    data = _read_yaml(suite_path)
    config_dir = suite_path.parent
    project_root = config_dir.parent.parent

    model_data = _read_yaml(config_dir / str(data["model_file"]))
    models = tuple(
        ModelConfig(key=str(key), **dict(value))
        for key, value in dict(model_data["models"]).items()
    )

    experiments: list[ExperimentConfig] = []
    for item in data["experiments"]:
        query_data = _read_yaml(config_dir / str(item["query_file"]))
        queries = tuple(
            QueryConfig(id=str(q["id"]), text=str(q["text"])) for q in query_data["queries"]
        )
        experiments.append(
            ExperimentConfig(
                experiment_id=str(item["experiment_id"]),
                category=str(query_data["category"]),
                language=str(query_data.get("language", "tr")),
                queries=queries,
            )
        )

    judge_data = _read_yaml(config_dir / str(data["judge_file"]))
    search_data = dict(data["search"])
    paths = dict(data["paths"])
    config = SuiteConfig(
        suite_id=str(data["suite_id"]),
        n_runs=int(data["n_runs"]),
        temperature=float(data["temperature"]),
        max_output_tokens=int(data["max_output_tokens"]),
        conditions=tuple(str(value) for value in data["conditions"]),
        max_tool_rounds=int(data["max_tool_rounds"]),
        system_prompts={str(k): v for k, v in dict(data["system_prompts"]).items()},
        models=models,
        experiments=tuple(experiments),
        judge=JudgeConfig(**judge_data),
        search=SearchConfig(**search_data),
        raw_dir=_resolve(project_root, str(paths["raw_dir"])),
        judged_dir=_resolve(project_root, str(paths["judged_dir"])),
        search_cache_dir=_resolve(project_root, str(paths["search_cache_dir"])),
        processed_dir=_resolve(project_root, str(paths["processed_dir"])),
    )
    _validate(config)
    return config


def _validate(config: SuiteConfig) -> None:
    if config.conditions != ("search_off", "search_on"):
        raise ConfigError("conditions must be exactly search_off, search_on")
    if config.n_runs < 1 or config.max_tool_rounds < 1:
        raise ConfigError("n_runs and max_tool_rounds must be positive")
    if len(config.models) != 4:
        raise ConfigError("the frozen design requires exactly four generation models")
    if len(config.experiments) != 2 or any(len(exp.queries) != 5 for exp in config.experiments):
        raise ConfigError("the frozen design requires two experiments with five queries each")
    ids = [q.id for exp in config.experiments for q in exp.queries]
    if len(ids) != len(set(ids)):
        raise ConfigError("query ids must be unique across the suite")
    if config.expected_rows != 400:
        raise ConfigError(f"frozen design must produce 400 rows, got {config.expected_rows}")
