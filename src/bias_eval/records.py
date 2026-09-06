"""Stable cell identifiers and normalized provider conversation records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig, ModelConfig, QueryConfig, SuiteConfig


@dataclass(frozen=True)
class CellSpec:
    experiment: ExperimentConfig
    model: ModelConfig
    condition: str
    query: QueryConfig
    run_index: int

    @property
    def safe_model_id(self) -> str:
        return self.model.model_id.replace("/", "__")

    @property
    def record_id(self) -> str:
        return f"{self.safe_model_id}_{self.condition}_{self.query.id}_{self.run_index:03d}"


@dataclass(frozen=True)
class ToolCall:
    id: str
    function_name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    raw: dict[str, Any]
    usage: dict[str, int]


@dataclass
class ConversationRecord:
    final_content: str | None
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def plan_cells(config: SuiteConfig) -> list[CellSpec]:
    return [
        CellSpec(exp, model, condition, query, run_index)
        for exp in config.experiments
        for model in config.models
        for condition in config.conditions
        for query in exp.queries
        for run_index in range(config.n_runs)
    ]


def pilot_cells(config: SuiteConfig) -> list[CellSpec]:
    """One production cell for every domain/model/condition combination."""

    return [
        CellSpec(exp, model, condition, exp.queries[0], 0)
        for exp in config.experiments
        for model in config.models
        for condition in config.conditions
    ]


def record_to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def cell_as_dict(cell: CellSpec) -> dict[str, Any]:
    return {
        "record_id": cell.record_id,
        "experiment_id": cell.experiment.experiment_id,
        "category": cell.experiment.category,
        "language": cell.experiment.language,
        "model_key": cell.model.key,
        "model_id": cell.model.model_id,
        "condition": cell.condition,
        "query_id": cell.query.id,
        "query_text": cell.query.text,
        "run_index": cell.run_index,
    }


__all__ = [
    "CellSpec",
    "ConversationRecord",
    "ProviderTurn",
    "ToolCall",
    "cell_as_dict",
    "pilot_cells",
    "plan_cells",
    "record_to_json",
]
