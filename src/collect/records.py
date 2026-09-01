"""Planned calls and the raw capture rows written to ``data/raw``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

RAW_SCHEMA_VERSION = 1

CONDITIONS = ("search_on", "search_off")
LAYERS = ("A", "B")
PROTOCOLS = ("alpha", "beta", "gamma")


def utc_now_iso() -> str:
    """Timestamp every row so we know which model snapshot produced it."""

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CallSpec:
    """One planned model call, identified stably so a run can resume.

    ``call_id`` hashes every field that changes what is measured. A design edit
    therefore produces new identifiers instead of silently reusing rows that
    were collected under the previous design.
    """

    run_id: str
    layer: str
    protocol: str
    sector: str
    query_id: str
    condition: str
    model_key: str
    repetition: int
    persona: str
    temperature: float
    prompt: str
    candidates: tuple[str, ...] = ()
    variant_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"Unknown layer {self.layer!r}; expected one of {LAYERS}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol {self.protocol!r}; expected one of {PROTOCOLS}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {self.condition!r}; expected one of {CONDITIONS}")
        if self.repetition < 0:
            raise ValueError("repetition must be zero or greater")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"Duplicate candidate brand in {self.query_id}: {self.candidates}")

    @property
    def call_id(self) -> str:
        """Deterministic identity used to skip work already on disk."""

        payload = _canonical_json(
            {
                "run_id": self.run_id,
                "layer": self.layer,
                "protocol": self.protocol,
                "sector": self.sector,
                "query_id": self.query_id,
                "condition": self.condition,
                "model_key": self.model_key,
                "repetition": self.repetition,
                "persona": self.persona,
                "temperature": self.temperature,
                "prompt": self.prompt,
                "candidates": list(self.candidates),
                "variant_ids": list(self.variant_ids),
            }
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CallRecord:
    """One JSONL line: what we asked, what came back, and under which model.

    Records are append-only. A retried call appends a second line with the same
    ``call_id``; downstream stages keep the successful one.
    """

    call_id: str
    run_id: str
    layer: str
    protocol: str
    sector: str
    query_id: str
    condition: str
    model_key: str
    repetition: int
    persona: str
    temperature: float
    prompt: str
    candidates: list[str]
    variant_ids: list[str]
    status: str
    started_at: str
    completed_at: str
    latency_ms: int
    attempts: int
    provider: str | None = None
    model_requested: str | None = None
    model_version: str | None = None
    response_text: str | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None
    schema_version: int = RAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"ok", "error"}:
            raise ValueError(f"Unknown status {self.status!r}; expected 'ok' or 'error'")

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CallRecord:
        known = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        return cls(**known)
