"""Append-only JSONL capture and the resume index built by re-reading it.

JSONL is line based, so an interrupted run leaves every earlier line intact.
``data/raw`` is never rewritten: a retry appends a new line for the same
``call_id`` rather than replacing the failed one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO

from .records import CallRecord


@dataclass(frozen=True)
class ResumeState:
    """What a previous run already produced for a given ``run_id``."""

    completed: frozenset[str]
    failed: frozenset[str]
    total_lines: int
    malformed_lines: tuple[int, ...]

    @property
    def has_malformed_lines(self) -> bool:
        return bool(self.malformed_lines)


def raw_log_path(raw_dir: str | Path, run_id: str) -> Path:
    """Every run writes to one file named after its ``run_id``."""

    if not run_id or any(character in run_id for character in r'/\:*?"<>| '):
        raise ValueError(f"run_id {run_id!r} is not usable as a file name")
    return Path(raw_dir) / f"{run_id}.jsonl"


def scan_raw_log(path: str | Path) -> ResumeState:
    """Index an existing capture file so finished calls are not repeated.

    Malformed lines are counted and reported rather than skipped silently; a
    truncated final line is the normal trace of a hard interruption.
    """

    log_path = Path(path)
    if not log_path.exists():
        return ResumeState(frozenset(), frozenset(), 0, ())

    completed: set[str] = set()
    failed: set[str] = set()
    malformed: list[int] = []
    total = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                malformed.append(number)
                continue
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                malformed.append(number)
                continue
            if payload.get("status") == "ok":
                completed.add(call_id)
            else:
                failed.add(call_id)

    return ResumeState(
        completed=frozenset(completed),
        failed=frozenset(failed - completed),
        total_lines=total,
        malformed_lines=tuple(malformed),
    )


class RawLogWriter:
    """Append records to the run's JSONL file, flushing each line to disk."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> RawLogWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: CallRecord) -> None:
        """Persist one record; the fsync is what makes resume trustworthy."""

        if self._handle is None:
            raise RuntimeError("RawLogWriter must be used as a context manager")
        self._handle.write(f"{record.to_json_line()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
