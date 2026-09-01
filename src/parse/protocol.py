"""Read the strict answer line out of a model response.

Each protocol ends its answer with one machine-readable line - ``SIRALAMA:``
for the ranked candidate list, ``ONERI:`` for a single pick, ``SECIM:`` for a
pairwise choice. This module finds that line and validates it.

Nothing is dropped. A response that cannot be parsed comes back as a
``ParseResult`` with ``status="failed"`` and a reason, because a parser that
skips what it cannot read loses data quietly and the loss surfaces weeks later
as a model that mysteriously underperforms.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .turkish import fold

MARKERS: dict[str, str] = {"alpha": "SIRALAMA", "beta": "ONERI", "gamma": "SECIM"}
SINGLE_VALUE_PROTOCOLS = frozenset({"beta", "gamma"})
LABELS = string.ascii_uppercase

NO_RESPONSE = "no_response_text"
MARKER_MISSING = "marker_missing"
EMPTY_VALUE = "empty_value"
UNKNOWN_LABEL = "unknown_label"
DUPLICATE_LABEL = "duplicate_label"
INCOMPLETE_RANKING = "incomplete_ranking"
EXPECTED_SINGLE_VALUE = "expected_single_value"

_TRIM = " \t.;\"'`*[]()<>"


@dataclass(frozen=True)
class ParseResult:
    """The outcome of reading one response - successful or not."""

    status: str
    protocol: str
    marker: str
    labels: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()
    line_index: int | None = None
    on_last_line: bool = False
    failure_reason: str | None = None
    raw_line: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    @property
    def top_choice(self) -> str | None:
        """The brand the model put first, when the response could be read."""

        return self.brands[0] if self.brands else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "protocol": self.protocol,
            "marker": self.marker,
            "labels": list(self.labels),
            "brands": list(self.brands),
            "line_index": self.line_index,
            "on_last_line": self.on_last_line,
            "failure_reason": self.failure_reason,
            "raw_line": self.raw_line,
        }


def marker_for(protocol: str) -> str:
    try:
        return MARKERS[protocol]
    except KeyError as error:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {sorted(MARKERS)}"
        ) from error


def _failure(
    protocol: str, marker: str, reason: str, *, raw_line: str | None = None
) -> ParseResult:
    return ParseResult(
        status="failed",
        protocol=protocol,
        marker=marker,
        failure_reason=reason,
        raw_line=raw_line,
    )


def _find_marker_line(lines: Sequence[str], marker: str) -> tuple[int, str] | None:
    """Return the last line carrying the marker, with its value part."""

    # Models routinely bold or bullet the final line, so leading decoration and
    # emphasis around the marker are tolerated; the value itself stays strict.
    pattern = re.compile(rf"^[\s*_#>\-]*{marker}[\s*_]*[:：][\s*_]*(.*)$")
    for index in range(len(lines) - 1, -1, -1):
        match = pattern.match(fold(lines[index]))
        if match is not None:
            return index, match.group(1)
    return None


def _split_labels(value: str) -> list[str]:
    return [token.strip(_TRIM) for token in value.split(",") if token.strip(_TRIM)]


def parse_response(
    text: str | None,
    *,
    protocol: str,
    candidates: Sequence[str] = (),
) -> ParseResult:
    """Read the answer line for ``protocol`` and map labels back to brand ids.

    ``candidates`` must be in the order the brands were shown, so label ``A``
    maps to ``candidates[0]``. Without it the labels are still validated but no
    brand mapping is produced.
    """

    marker = marker_for(protocol)
    if not text or not text.strip():
        return _failure(protocol, marker, NO_RESPONSE)

    lines = text.splitlines()
    found = _find_marker_line(lines, marker)
    if found is None:
        return _failure(protocol, marker, MARKER_MISSING)

    line_index, value = found
    raw_line = lines[line_index].strip()
    tokens = _split_labels(value)
    if not tokens:
        return _failure(protocol, marker, EMPTY_VALUE, raw_line=raw_line)

    expected_count = len(candidates) if candidates else None
    if protocol in SINGLE_VALUE_PROTOCOLS:
        if len(tokens) != 1:
            return _failure(protocol, marker, EXPECTED_SINGLE_VALUE, raw_line=raw_line)
        expected_count = 1

    allowed = set(LABELS[: len(candidates)]) if candidates else set(LABELS)
    if any(token not in allowed for token in tokens):
        return _failure(protocol, marker, UNKNOWN_LABEL, raw_line=raw_line)

    duplicates = [label for label, count in Counter(tokens).items() if count > 1]
    if duplicates:
        return _failure(protocol, marker, DUPLICATE_LABEL, raw_line=raw_line)

    if expected_count is not None and len(tokens) != expected_count:
        return _failure(protocol, marker, INCOMPLETE_RANKING, raw_line=raw_line)

    brands = tuple(candidates[LABELS.index(label)] for label in tokens) if candidates else ()
    return ParseResult(
        status="ok",
        protocol=protocol,
        marker=marker,
        labels=tuple(tokens),
        brands=brands,
        line_index=line_index,
        on_last_line=line_index == len(lines) - 1,
        raw_line=raw_line,
    )


@dataclass
class ParseReport:
    """Aggregate parse outcomes so a weak model is visible, not hidden."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    failures_by_reason: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0

    def add(self, result: ParseResult, *, model_key: str = "unknown") -> None:
        self.total += 1
        bucket = self.by_model.setdefault(model_key, {"total": 0, "succeeded": 0, "failed": 0})
        bucket["total"] += 1
        if result.succeeded:
            self.succeeded += 1
            bucket["succeeded"] += 1
            return
        self.failed += 1
        bucket["failed"] += 1
        reason = result.failure_reason or "unknown"
        self.failures_by_reason[reason] = self.failures_by_reason.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "failures_by_reason": dict(sorted(self.failures_by_reason.items())),
            "by_model": {
                model: {
                    **counts,
                    "success_rate": (
                        round(counts["succeeded"] / counts["total"], 4) if counts["total"] else 0.0
                    ),
                }
                for model, counts in sorted(self.by_model.items())
            },
        }


def parse_records(records: Iterable[dict[str, Any]]) -> tuple[list[ParseResult], ParseReport]:
    """Parse successful capture rows and report per-model parse success."""

    results: list[ParseResult] = []
    report = ParseReport()
    for record in records:
        if record.get("status") != "ok":
            continue
        result = parse_response(
            record.get("response_text"),
            protocol=str(record.get("protocol", "alpha")),
            candidates=tuple(record.get("candidates") or ()),
        )
        results.append(result)
        report.add(result, model_key=str(record.get("model_key", "unknown")))
    return results, report
