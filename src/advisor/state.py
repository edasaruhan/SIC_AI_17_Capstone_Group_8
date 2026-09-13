"""The graph's state: what each node may read and what it is allowed to add.

One flat dictionary rather than nested objects, because every node's contribution has
to be readable in a receipt and in the final report without a decoder.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def extend(left: list, right: list) -> list:
    """Reducer for fan-out nodes: concurrent branches append, never overwrite."""
    return [*left, *right]


class Observation(TypedDict):
    """One (query, condition) measurement, made by string matching, not by a model."""

    query: str
    condition: str
    rep: int
    mentioned: bool
    first: bool
    named_brands: list[str]
    in_search_results: bool
    best_position: int | None
    n_results_mentioning: int


class AdvisorInput(TypedDict):
    """What the caller must supply; nodes may index these directly."""

    brand: str
    sector: str
    language: str
    max_calls: int


class AdvisorState(AdvisorInput, total=False):
    """Inputs plus everything the nodes add. Added keys are read with ``.get``,
    because a node must not assume a predecessor already ran."""

    # Plan
    queries: list[str]
    rivals: list[str]
    # Fan-out results (reducers keep concurrent branches from clobbering each other)
    search: Annotated[list[dict], extend]
    observations: Annotated[list[Observation], extend]
    calls: Annotated[list[dict], extend]
    # Analysis
    measures: dict[str, Any]
    diagnosis: str
    evidence: list[dict]
    recommendations: list[dict]
    report: str
    notes: Annotated[list[str], extend]
