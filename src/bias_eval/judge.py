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
from .normalization import normalize_judgment_brands
from .providers import (
    RETRYABLE_STATUS_CODES,
    ProviderError,
    _retry_after,
    _safe_error_text,
    api_key,
)
from .storage import atomic_write_json, iter_results, read_json

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
    def __init__(self, config: JudgeConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

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
        try:
            response = await self.http.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key(self.config.api_key_env)}"},
                json=payload,
            )
        except httpx.TransportError as error:
            raise ProviderError("Judge transport error", retryable=True) from error
        if response.status_code >= 400:
            message = _safe_error_text(response)
            lowered = message.casefold()
            quota = response.status_code in {402, 429} and any(
                word in lowered for word in ("quota", "credit", "balance", "billing")
            )
            raise ProviderError(
                message,
                status_code=response.status_code,
                retry_after=_retry_after(response),
                retryable=response.status_code in RETRYABLE_STATUS_CODES and not quota,
                quota_exhausted=quota,
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            result = json.loads(content) if isinstance(content, str) else content
            validated = validate_judgment(result, category, str(raw["condition"]))
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ProviderError(
                "Judge returned invalid structured output", retryable=True
            ) from error
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
            if not error.retryable or attempt == max_attempts:
                raise
            delay = (
                error.retry_after if error.retry_after is not None else float(2 ** (attempt - 1))
            )
            await sleep(delay)
    raise AssertionError("retry loop exhausted")


async def judge_suite(
    config: SuiteConfig,
    *,
    http: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    raw_records = [
        item for item in iter_results(config.raw_dir) if item.get("status") == "completed"
    ]
    owns_http = http is None
    session = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    client = CerebrasJudgeClient(config.judge, session)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for record in raw_records:
        queue.put_nowait(record)
    counts: Counter[str] = Counter()
    quota_exhausted = False
    state_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal quota_exhausted
        while not queue.empty():
            raw = await queue.get()
            path = judged_path(config, raw)
            try:
                if judgment_is_current(path, raw):
                    counts["skipped"] += 1
                    continue
                async with state_lock:
                    if quota_exhausted:
                        counts["blocked"] += 1
                        continue
                started = monotonic()
                started_at = _now()
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
                            "message": str(error),
                            "status_code": error.status_code,
                            "quota_exhausted": error.quota_exhausted,
                        },
                    },
                )
                counts["errors"] += 1
            finally:
                queue.task_done()

    try:
        await asyncio.gather(*(worker() for _ in range(config.judge.max_concurrent)))
    finally:
        if owns_http:
            await session.aclose()
    return {
        "eligible": len(raw_records),
        "completed": counts["completed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "blocked": counts["blocked"],
        "quota_exhausted": quota_exhausted,
    }


def judge_status(config: SuiteConfig) -> dict[str, Any]:
    raw_records = [
        item for item in iter_results(config.raw_dir) if item.get("status") == "completed"
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
    return {
        "expected": config.expected_rows,
        "eligible_generation_records": len(raw_records),
        "current": current,
        "stale": stale,
        "errors": errors,
        "remaining": len(raw_records) - current,
    }
