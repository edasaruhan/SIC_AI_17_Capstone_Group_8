"""Resumable Cerebras judge pass with strict structured output."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast

import httpx

from .config import JudgeConfig, SuiteConfig
from .features import deterministic_features
from .judge_schema import (
    JUDGE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_judge_input,
    content_hash,
    prompt_hash,
    response_schema,
)
from .logging import logger
from .normalization import normalize_judgment_brands
from .providers import (
    RETRYABLE_STATUS_CODES,
    ProviderError,
    _retry_after,
    _safe_error_text,
    api_key,
)
from .records import plan_cells
from .storage import atomic_write_json, iter_results, read_json, record_is_complete

Sleep = Callable[[float], Awaitable[None]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def judged_path(config: SuiteConfig, raw: dict[str, Any]) -> Path:
    safe_model = str(raw["model_id"]).replace("/", "__")
    return (
        config.judged_dir
        / str(raw["experiment_id"])
        / safe_model
        / str(raw["condition"])
        / f"{raw['query_id']}_{int(raw['run_index']):03d}.json"
    )


def judgment_is_current(path: Path, raw: dict[str, Any]) -> bool:
    value = read_json(path)
    return bool(
        value
        and value.get("status") == "completed"
        and value.get("content_hash") == content_hash(raw)
        and value.get("judge_prompt_hash") == prompt_hash(str(raw["category"]))
    )


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    declared = schema.get("type")
    allowed = declared if isinstance(declared, list) else [declared]
    if value is None and "null" in allowed:
        return
    expected = next((item for item in allowed if item != "null"), None)
    if expected == "object" and not isinstance(value, dict):
        raise ValueError(f"{path} must be {declared}")
    if expected == "array" and not isinstance(value, list):
        raise ValueError(f"{path} must be {declared}")
    if expected == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be {declared}")
    if expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path} must be {declared}")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{path} must be {declared}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has unsupported value {value!r}")
    if expected == "integer":
        integer = cast(int, value)
        if integer < int(schema.get("minimum", integer)):
            raise ValueError(f"{path} is below minimum")
    if expected == "object":
        obj = cast(dict[str, Any], value)
        properties = cast(dict[str, dict[str, Any]], schema.get("properties", {}))
        required = cast(list[str], schema.get("required", []))
        missing = set(required) - set(obj)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(obj) - set(properties)
            if extra:
                raise ValueError(f"{path} has extra fields {sorted(extra)}")
        for key, item in obj.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}")
    if expected == "array":
        item_schema = cast(dict[str, Any], schema.get("items", {}))
        for index, item in enumerate(cast(list[Any], value)):
            _validate_value(item, item_schema, f"{path}[{index}]")


def validate_judgment(value: Any, category: str, condition: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judge response must be an object")
    _validate_value(value, response_schema(category), "judgment")
    if condition == "search_off" and value["search_aware"] is not None:
        raise ValueError("search_off search_aware must be null")
    if condition == "search_on" and value["search_aware"] is None:
        raise ValueError("search_on search_aware must be an object")
    return value


class CerebrasJudgeClient:
    def __init__(
        self,
        config: JudgeConfig,
        http: httpx.AsyncClient,
        *,
        sleep: Sleep = asyncio.sleep,
    ):
        self.config = config
        self.http = http
        self.sleep = sleep
        self._pacing_lock = asyncio.Lock()
        self._last_request_started: float | None = None

    async def _wait_for_request_slot(self, record_id: Any) -> None:
        async with self._pacing_lock:
            now = monotonic()
            delay = 0.0
            if self._last_request_started is not None:
                elapsed = now - self._last_request_started
                delay = max(0.0, self.config.min_request_interval_seconds - elapsed)
            if delay > 0:
                logger.info(
                    "judge_rate_limit_wait record_id={} delay_seconds={:.3f}",
                    record_id,
                    delay,
                )
                await self.sleep(delay)
            self._last_request_started = monotonic()

    async def judge(self, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        category = str(raw["category"])
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_input(raw)},
            ],
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"brand_bias_{category}_extraction",
                    "strict": True,
                    "schema": response_schema(category),
                },
            },
        }
        started = monotonic()
        await self._wait_for_request_slot(raw.get("record_id"))
        logger.debug(
            "judge_request record_id={} model={} category={} condition={}",
            raw.get("record_id"),
            self.config.model_id,
            category,
            raw.get("condition"),
        )
        try:
            response = await self.http.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key(self.config.api_key_env)}"},
                json=payload,
            )
        except httpx.TransportError as error:
            logger.error(
                "judge_transport_error record_id={} duration_seconds={:.3f} error_type={}",
                raw.get("record_id"),
                monotonic() - started,
                type(error).__name__,
            )
            raise ProviderError(
                "Judge transport error",
                retryable=True,
                transport_error=True,
                component="judge",
            ) from error
        if response.status_code >= 400:
            message = _safe_error_text(response)
            lowered = message.casefold()
            quota = response.status_code in {402, 429} and any(
                word in lowered for word in ("quota", "credit", "balance", "billing")
            )
            logger.warning(
                "judge_http_error record_id={} status_code={} duration_seconds={:.3f} "
                "retry_after={} message={}",
                raw.get("record_id"),
                response.status_code,
                monotonic() - started,
                _retry_after(response),
                message,
            )
            raise ProviderError(
                message,
                status_code=response.status_code,
                retry_after=_retry_after(response),
                retryable=response.status_code in RETRYABLE_STATUS_CODES and not quota,
                quota_exhausted=quota,
                component="judge",
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
            if (
                isinstance(result, dict)
                and str(raw["condition"]) == "search_on"
                and result.get("search_aware") is None
                and not raw.get("tool_calls")
                and not raw.get("search_results")
            ):
                result = dict(result)
                result["search_aware"] = {
                    "explicitly_references_search_results": False,
                    "cites_specific_sources": False,
                    "source_names_cited": [],
                    "uses_search_to_justify_top_pick": False,
                }
                logger.warning(
                    "judge_response_repaired record_id={} repair=empty_search_trace",
                    raw.get("record_id"),
                )
            validated = validate_judgment(result, category, str(raw["condition"]))
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            logger.error(
                "judge_invalid_response record_id={} status_code={} duration_seconds={:.3f} "
                "error_type={} validation_error={}",
                raw.get("record_id"),
                response.status_code,
                monotonic() - started,
                type(error).__name__,
                str(error),
            )
            raise ProviderError(
                f"Judge returned invalid structured output: {error}",
                retryable=True,
                component="judge",
            ) from error
        logger.info(
            "judge_response record_id={} status_code={} duration_seconds={:.3f}",
            raw.get("record_id"),
            response.status_code,
            monotonic() - started,
        )
        return validated, body


async def _judge_with_retries(
    client: CerebrasJudgeClient,
    raw: dict[str, Any],
    *,
    sleep: Sleep,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    for attempt in range(1, max_attempts + 1):
        try:
            judgment, body = await client.judge(raw)
            return judgment, body, attempt
        except ProviderError as error:
            error.attempts = attempt
            if not error.retryable or attempt == max_attempts:
                logger.error(
                    "judge_operation_failed record_id={} attempts={} status_code={} message={}",
                    raw.get("record_id"),
                    attempt,
                    error.status_code,
                    error,
                )
                raise
            delay = (
                error.retry_after if error.retry_after is not None else float(2 ** (attempt - 1))
            )
            logger.warning(
                "judge_retry record_id={} attempt={} next_attempt={} delay_seconds={} "
                "status_code={}",
                raw.get("record_id"),
                attempt,
                attempt + 1,
                delay,
                error.status_code,
            )
            await sleep(delay)
    raise AssertionError("retry loop exhausted")


async def judge_suite(
    config: SuiteConfig,
    *,
    http: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    raw_records = [
        item
        for item in iter_results(config.raw_dir)
        if record_is_complete(item) and item.get("record_id") in planned_ids
    ]
    owns_http = http is None
    session = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    client = CerebrasJudgeClient(config.judge, session, sleep=sleep)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for record in raw_records:
        queue.put_nowait(record)
    counts: Counter[str] = Counter()
    quota_exhausted = False
    state_lock = asyncio.Lock()
    logger.info(
        "judge_suite_started eligible={} workers={} model={} judged_dir={} "
        "min_request_interval_seconds={}",
        len(raw_records),
        config.judge.max_concurrent,
        config.judge.model_id,
        config.judged_dir,
        config.judge.min_request_interval_seconds,
    )

    async def worker() -> None:
        nonlocal quota_exhausted
        while not queue.empty():
            raw = await queue.get()
            path = judged_path(config, raw)
            try:
                if judgment_is_current(path, raw):
                    counts["skipped"] += 1
                    logger.debug(
                        "judge_record_skipped record_id={} reason=current", raw.get("record_id")
                    )
                    continue
                async with state_lock:
                    if quota_exhausted:
                        counts["blocked"] += 1
                        logger.debug(
                            "judge_record_blocked record_id={} reason=quota_circuit_open",
                            raw.get("record_id"),
                        )
                        continue
                started = monotonic()
                started_at = _now()
                logger.info(
                    "judge_record_started record_id={} category={} condition={}",
                    raw.get("record_id"),
                    raw.get("category"),
                    raw.get("condition"),
                )
                judgment, provider_body, attempts = await _judge_with_retries(
                    client, raw, sleep=sleep
                )
                normalized = normalize_judgment_brands(judgment, str(raw["category"]))
                atomic_write_json(
                    path,
                    {
                        "record_id": raw["record_id"],
                        "experiment_id": raw["experiment_id"],
                        "category": raw["category"],
                        "condition": raw["condition"],
                        "content_hash": content_hash(raw),
                        "judge_model": config.judge.model_id,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                        "judge_prompt_hash": prompt_hash(str(raw["category"])),
                        "judgment": normalized,
                        "deterministic": deterministic_features(raw),
                        "raw_response": provider_body,
                        "metadata": {
                            "started_at": started_at,
                            "completed_at": _now(),
                            "duration_seconds": round(monotonic() - started, 3),
                            "attempts": attempts,
                        },
                        "status": "completed",
                    },
                )
                counts["completed"] += 1
                logger.info(
                    "judge_record_completed record_id={} duration_seconds={:.3f} attempts={}",
                    raw.get("record_id"),
                    monotonic() - started,
                    attempts,
                )
            except ProviderError as error:
                if error.quota_exhausted:
                    async with state_lock:
                        quota_exhausted = True
                atomic_write_json(
                    path,
                    {
                        "record_id": raw["record_id"],
                        "content_hash": content_hash(raw),
                        "judge_prompt_hash": prompt_hash(str(raw["category"])),
                        "status": "error",
                        "error": {
                            "type": type(error).__name__,
                            "component": error.component,
                            "message": str(error),
                            "status_code": error.status_code,
                            "quota_exhausted": error.quota_exhausted,
                            "transport_error": error.transport_error,
                        },
                        "metadata": {"completed_at": _now(), "attempts": error.attempts},
                    },
                )
                counts["errors"] += 1
                logger.error(
                    "judge_record_failed record_id={} attempts={} status_code={} "
                    "quota_exhausted={} message={}",
                    raw.get("record_id"),
                    error.attempts,
                    error.status_code,
                    error.quota_exhausted,
                    error,
                )
            finally:
                queue.task_done()

    try:
        await asyncio.gather(*(worker() for _ in range(config.judge.max_concurrent)))
    finally:
        if owns_http:
            await session.aclose()
    result = {
        "eligible": len(raw_records),
        "completed": counts["completed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "blocked": counts["blocked"],
        "quota_exhausted": quota_exhausted,
    }
    logger.info("judge_suite_completed summary={}", result)
    return result


def judge_status(config: SuiteConfig) -> dict[str, Any]:
    planned_ids = {cell.record_id for cell in plan_cells(config)}
    raw_records = [
        item
        for item in iter_results(config.raw_dir)
        if record_is_complete(item) and item.get("record_id") in planned_ids
    ]
    current = 0
    stale = 0
    errors = 0
    for raw in raw_records:
        value = read_json(judged_path(config, raw))
        if value and value.get("status") == "error":
            errors += 1
        elif judgment_is_current(judged_path(config, raw), raw):
            current += 1
        elif value:
            stale += 1
    result = {
        "expected": config.expected_rows,
        "eligible_generation_records": len(raw_records),
        "current": current,
        "stale": stale,
        "errors": errors,
        "eligible_remaining": len(raw_records) - current,
        "remaining_for_complete_suite": config.expected_rows - current,
    }
    logger.info("judge_status summary={}", result)
    return result
