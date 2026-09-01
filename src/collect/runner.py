"""Drive a collection run: pace it, retry it, record it, and let it resume.

The run is deliberately restartable. Every finished call is on disk before the
next one starts, so an interrupted run - a dropped connection, a closed laptop,
a spend cap - costs at most the calls that were in flight.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import CollectionSettings
from .providers import (
    ModelConfig,
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
    adapter_for,
)
from .rate_limit import TokenBucket
from .records import CallRecord, CallSpec, utc_now_iso
from .storage import RawLogWriter, ResumeState, raw_log_path, scan_raw_log

LOGGER = logging.getLogger("collect.runner")


@dataclass
class RunSummary:
    """The numbers the daily monitoring note is written from."""

    run_id: str
    planned: int = 0
    skipped: int = 0
    succeeded: int = 0
    failed: int = 0
    spend_usd: float = 0.0
    stopped_on_spend_cap: bool = False
    errors_by_type: dict[str, int] = field(default_factory=dict)
    log_path: str = ""

    @property
    def attempted(self) -> int:
        return self.succeeded + self.failed

    @property
    def error_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "planned": self.planned,
            "skipped": self.skipped,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "error_rate": round(self.error_rate, 4),
            "spend_usd": round(self.spend_usd, 4),
            "stopped_on_spend_cap": self.stopped_on_spend_cap,
            "errors_by_type": dict(sorted(self.errors_by_type.items())),
            "log_path": self.log_path,
        }


def _error_record(
    spec: CallSpec,
    model: ModelConfig,
    *,
    error: BaseException,
    attempts: int,
    started_at: str,
    started_monotonic: float,
) -> CallRecord:
    return CallRecord(
        call_id=spec.call_id,
        run_id=spec.run_id,
        layer=spec.layer,
        protocol=spec.protocol,
        sector=spec.sector,
        query_id=spec.query_id,
        condition=spec.condition,
        model_key=spec.model_key,
        repetition=spec.repetition,
        persona=spec.persona,
        temperature=spec.temperature,
        prompt=spec.prompt,
        candidates=list(spec.candidates),
        variant_ids=list(spec.variant_ids),
        status="error",
        started_at=started_at,
        completed_at=utc_now_iso(),
        latency_ms=int((time.monotonic() - started_monotonic) * 1000),
        attempts=attempts,
        provider=model.provider,
        model_requested=model.model,
        error_type=type(error).__name__,
        error_message=str(error)[:1000],
    )


async def execute_call(
    client: httpx.AsyncClient,
    spec: CallSpec,
    *,
    model: ModelConfig,
    settings: CollectionSettings,
    bucket: TokenBucket,
) -> CallRecord:
    """Make one call with pacing and backoff, and always return a record.

    A failure is data too: it is written with its error type so the monitoring
    table can show which model is degrading and why.
    """

    adapter = adapter_for(model)
    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    attempts = 1

    async def attempt_once() -> ProviderResponse:
        await bucket.acquire()
        return await adapter.complete(
            client,
            model,
            system=spec.persona,
            prompt=spec.prompt,
            temperature=spec.temperature,
            use_search=spec.condition == "search_on" and model.supports_search,
        )

    retrying = AsyncRetrying(
        stop=stop_after_attempt(settings.retry.max_attempts),
        wait=wait_exponential_jitter(
            initial=settings.retry.initial_seconds,
            max=settings.retry.max_seconds,
            jitter=settings.retry.jitter_seconds,
        ),
        retry=retry_if_exception_type(RetryableProviderError),
        reraise=True,
    )
    try:
        response = await retrying(attempt_once)
    except (ProviderError, httpx.HTTPError) as error:
        attempts = int(retrying.statistics.get("attempt_number", 1))
        LOGGER.warning("call %s failed after %s attempt(s): %s", spec.call_id, attempts, error)
        return _error_record(
            spec,
            model,
            error=error,
            attempts=attempts,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
    attempts = int(retrying.statistics.get("attempt_number", 1))

    return CallRecord(
        call_id=spec.call_id,
        run_id=spec.run_id,
        layer=spec.layer,
        protocol=spec.protocol,
        sector=spec.sector,
        query_id=spec.query_id,
        condition=spec.condition,
        model_key=spec.model_key,
        repetition=spec.repetition,
        persona=spec.persona,
        temperature=spec.temperature,
        prompt=spec.prompt,
        candidates=list(spec.candidates),
        variant_ids=list(spec.variant_ids),
        status="ok",
        started_at=started_at,
        completed_at=utc_now_iso(),
        latency_ms=int((time.monotonic() - started_monotonic) * 1000),
        attempts=attempts,
        provider=model.provider,
        model_requested=model.model,
        model_version=response.model_version,
        response_text=response.text,
        search_results=response.search_results,
        stop_reason=response.stop_reason,
        usage=response.usage,
        cost_usd=response.cost_usd(model),
        raw_response=response.raw if settings.store_raw_response else None,
    )


def pending_specs(specs: Iterable[CallSpec], resume: ResumeState) -> list[CallSpec]:
    """Drop calls that already succeeded; failed ones are tried again."""

    return [spec for spec in specs if spec.call_id not in resume.completed]


async def run_collection(
    specs: Sequence[CallSpec],
    settings: CollectionSettings,
    *,
    run_id: str,
    client: httpx.AsyncClient | None = None,
) -> RunSummary:
    """Run every pending call and append the results to the run's JSONL file."""

    log_path = raw_log_path(settings.raw_dir, run_id)
    resume = scan_raw_log(log_path)
    if resume.has_malformed_lines:
        LOGGER.warning(
            "%s has %d unreadable line(s) at %s; they are counted, not skipped silently",
            log_path,
            len(resume.malformed_lines),
            resume.malformed_lines[:10],
        )

    pending = pending_specs(specs, resume)
    summary = RunSummary(
        run_id=run_id,
        planned=len(specs),
        skipped=len(specs) - len(pending),
        log_path=str(log_path),
    )
    LOGGER.info(
        "run %s: %d planned, %d already collected, %d to go",
        run_id,
        summary.planned,
        summary.skipped,
        len(pending),
    )
    if not pending:
        return summary

    buckets = {
        model.key: TokenBucket(rate_per_second=model.requests_per_second, capacity=model.burst)
        for model in settings.models
    }
    semaphore = asyncio.Semaphore(settings.concurrency)
    write_lock = asyncio.Lock()
    stop = asyncio.Event()

    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def worker(spec: CallSpec, writer: RawLogWriter) -> None:
        if stop.is_set():
            return
        model = settings.model(spec.model_key)
        async with semaphore:
            if stop.is_set():
                return
            record = await execute_call(
                http_client,
                spec,
                model=model,
                settings=settings,
                bucket=buckets[spec.model_key],
            )
        async with write_lock:
            writer.write(record)
            if record.succeeded:
                summary.succeeded += 1
                summary.spend_usd += record.cost_usd or 0.0
            else:
                summary.failed += 1
                key = record.error_type or "unknown"
                summary.errors_by_type[key] = summary.errors_by_type.get(key, 0) + 1
            if summary.spend_usd >= settings.daily_spend_cap_usd and not stop.is_set():
                summary.stopped_on_spend_cap = True
                stop.set()
                LOGGER.warning(
                    "spend cap reached at $%.2f; no further calls will be scheduled",
                    summary.spend_usd,
                )

    try:
        with RawLogWriter(log_path) as writer:
            await asyncio.gather(*(worker(spec, writer) for spec in pending))
    finally:
        if owns_client:
            await http_client.aclose()

    LOGGER.info("run %s finished: %s", run_id, summary.as_dict())
    return summary


__all__ = ["RunSummary", "execute_call", "pending_specs", "run_collection"]
