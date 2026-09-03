"""Resumable execution of generation cells and the Serper tool loop."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import monotonic
from typing import Any

import httpx

from .config import SuiteConfig
from .logging import logger
from .provenance import collection_fingerprint, ensure_collection_lock
from .providers import WEB_SEARCH_TOOL, OpenAICompatibleClient, ProviderError
from .records import CellSpec, ConversationRecord, pilot_cells, plan_cells
from .search import SerperSearch
from .storage import atomic_write_json, is_complete, result_path

Sleep = Callable[[float], Awaitable[None]]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0))


async def run_conversation(
    client: OpenAICompatibleClient,
    search: SerperSearch,
    cell: CellSpec,
    config: SuiteConfig,
    *,
    thinking: bool | None = None,
) -> ConversationRecord:
    system = config.system_prompts.get(cell.condition)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": cell.query.text})
    use_search = cell.condition == "search_on"
    tools = [WEB_SEARCH_TOOL] if use_search else None
    all_calls: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    rounds = config.max_tool_rounds if use_search else 1
    logger.debug(
        "conversation_started record_id={} model={} condition={} max_rounds={}",
        cell.record_id,
        cell.model.model_id,
        cell.condition,
        rounds,
    )

    for round_index in range(rounds):
        logger.debug(
            "conversation_round record_id={} round={} message_count={}",
            cell.record_id,
            round_index + 1,
            len(messages),
        )
        turn = await client.chat(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            tools=tools,
            thinking=thinking,
        )
        all_raw.append(turn.raw)
        _add_usage(usage, turn.usage)
        if not turn.tool_calls:
            if not isinstance(turn.content, str) or not turn.content.strip():
                raise ProviderError(
                    "Provider returned an empty final response",
                    retryable=True,
                    component="generation",
                )
            messages.append({"role": "assistant", "content": turn.content})
            logger.info(
                "conversation_completed record_id={} rounds={} searches={} search_not_used={}",
                cell.record_id,
                round_index + 1,
                len(all_calls),
                use_search and not all_calls,
            )
            return ConversationRecord(
                final_content=turn.content,
                messages=messages,
                tool_calls=all_calls,
                search_results=all_results,
                raw_responses=all_raw,
                usage=usage,
            )
        if not use_search:
            raise ProviderError("search_off response unexpectedly requested a tool")

        assistant_calls = []
        for call in turn.tool_calls:
            if call.function_name != "web_search":
                raise ProviderError(f"Unsupported tool requested: {call.function_name}")
            query = str(call.arguments.get("query", ""))
            logger.info(
                "tool_call record_id={} round={} tool={} query_length={}",
                cell.record_id,
                round_index + 1,
                call.function_name,
                len(query),
            )
            hit = await search.search(query)
            call_record = {
                "round": round_index,
                "tool_call_id": call.id,
                "function": call.function_name,
                "arguments": call.arguments,
            }
            all_calls.append(call_record)
            all_results.append(
                {
                    "round": round_index,
                    "query": query,
                    "source": hit.source,
                    "results": hit.results,
                }
            )
            assistant_call = dict(call.raw) if call.raw is not None else {}
            assistant_call.update(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function_name,
                        "arguments": __import__("json").dumps(call.arguments, ensure_ascii=False),
                    },
                }
            )
            assistant_calls.append(assistant_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function_name,
                    "content": hit.formatted,
                }
            )
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": assistant_calls,
        }
        if turn.content is not None:
            assistant_message["content"] = turn.content
        messages.insert(len(messages) - len(turn.tool_calls), assistant_message)
    raise ProviderError(f"Model exceeded {config.max_tool_rounds} consecutive search rounds")


async def _with_retries(
    operation: Callable[[], Awaitable[ConversationRecord]],
    *,
    sleep: Sleep,
    max_attempts: int = 3,
) -> tuple[ConversationRecord, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation(), attempt
        except ProviderError as error:
            error.attempts = attempt
            if not error.retryable or attempt == max_attempts:
                logger.error(
                    "operation_failed component={} attempts={} max_attempts={} status_code={} retryable={} "
                    "quota_exhausted={} transport_error={} message={}",
                    error.component,
                    attempt,
                    max_attempts,
                    error.status_code,
                    error.retryable,
                    error.quota_exhausted,
                    error.transport_error,
                    error,
                )
                raise
            delay = (
                error.retry_after if error.retry_after is not None else float(2 ** (attempt - 1))
            )
            logger.warning(
                "operation_retry component={} attempt={} next_attempt={} max_attempts={} status_code={} "
                "delay_seconds={} message={}",
                error.component,
                attempt,
                attempt + 1,
                max_attempts,
                error.status_code,
                delay,
                error,
            )
            await sleep(delay)
    raise AssertionError("retry loop exhausted")


def _completed_payload(
    cell: CellSpec,
    config: SuiteConfig,
    conversation: ConversationRecord,
    *,
    started_at: str,
    duration: float,
    attempts: int,
    thinking: bool | None,
) -> dict[str, Any]:
    return {
        "record_id": cell.record_id,
        "experiment_id": cell.experiment.experiment_id,
        "model_id": cell.model.model_id,
        "provider": cell.model.provider,
        "condition": cell.condition,
        "category": cell.experiment.category,
        "language": cell.experiment.language,
        "query_id": cell.query.id,
        "query_text": cell.query.text,
        "run_index": cell.run_index,
        "temperature": config.temperature,
        "system_prompt": config.system_prompts.get(cell.condition),
        "request_parameters": {
            "max_tokens": config.max_output_tokens,
            "reasoning_effort": cell.model.reasoning_effort,
            "thinking": thinking,
            "request_timeout_seconds": cell.model.request_timeout_seconds,
            "max_attempts": cell.model.max_attempts,
            "tool_choice": "auto" if cell.condition == "search_on" else None,
            "max_tool_rounds": config.max_tool_rounds if cell.condition == "search_on" else 0,
        },
        "final_response": conversation.final_content,
        "tool_calls": conversation.tool_calls,
        "search_results": conversation.search_results,
        "conversation": conversation.messages,
        "raw_responses": conversation.raw_responses,
        "usage": conversation.usage,
        "quality_flags": {
            "search_not_used": cell.condition == "search_on" and not conversation.tool_calls
        },
        "metadata": {
            "started_at": started_at,
            "completed_at": _now(),
            "duration_seconds": round(duration, 3),
            "attempts": attempts,
        },
        "status": "completed",
    }


async def run_suite(
    config: SuiteConfig,
    *,
    pilot: bool = False,
    model_keys: set[str] | None = None,
    http: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
    disable_abliteration_thinking: bool = False,
) -> dict[str, Any]:
    ensure_collection_lock(config)
    cells = pilot_cells(config) if pilot else plan_cells(config)
    available_keys = {model.key for model in config.models}
    selected_keys = model_keys or available_keys
    unknown_keys = selected_keys - available_keys
    if unknown_keys:
        raise ValueError(f"Unknown model keys: {sorted(unknown_keys)}")
    cells = [cell for cell in cells if cell.model.key in selected_keys]
    owns_http = http is None
    session = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    search = SerperSearch(config.search, config.search_cache_dir, session)
    counts: Counter[str] = Counter()
    disabled_models: set[str] = set()
    logger.info(
        "generation_suite_started pilot={} planned={} selected_models={} raw_dir={}",
        pilot,
        len(cells),
        sorted(selected_keys),
        config.raw_dir,
    )

    async def run_model(model_key: str) -> None:
        model_cells = [cell for cell in cells if cell.model.key == model_key]
        client = OpenAICompatibleClient(model_cells[0].model, session)
        thinking = (
            False
            if disable_abliteration_thinking and model_cells[0].model.provider == "abliteration"
            else None
        )
        logger.info(
            "generation_model_started model_key={} model_id={} cells={} max_attempts={} "
            "timeout_seconds={} thinking={}",
            model_key,
            model_cells[0].model.model_id,
            len(model_cells),
            model_cells[0].model.max_attempts,
            model_cells[0].model.request_timeout_seconds,
            thinking,
        )
        for cell in model_cells:
            path = result_path(config.raw_dir, cell)
            if is_complete(path):
                counts["skipped"] += 1
                logger.debug(
                    "generation_cell_skipped record_id={} reason=completed", cell.record_id
                )
                continue
            if model_key in disabled_models:
                counts["blocked"] += 1
                logger.debug(
                    "generation_cell_blocked record_id={} model_key={} reason=circuit_open",
                    cell.record_id,
                    model_key,
                )
                continue
            started_at = _now()
            started = monotonic()
            logger.info(
                "generation_cell_started record_id={} model={} experiment={} condition={} "
                "query_id={} run_index={}",
                cell.record_id,
                cell.model.model_id,
                cell.experiment.experiment_id,
                cell.condition,
                cell.query.id,
                cell.run_index,
            )
            try:
                conversation, attempts = await _with_retries(
                    lambda cell=cell: run_conversation(
                        client, search, cell, config, thinking=thinking
                    ),
                    sleep=sleep,
                    max_attempts=cell.model.max_attempts,
                )
                atomic_write_json(
                    path,
                    _completed_payload(
                        cell,
                        config,
                        conversation,
                        started_at=started_at,
                        duration=monotonic() - started,
                        attempts=attempts,
                        thinking=thinking,
                    ),
                )
                counts["completed"] += 1
                logger.info(
                    "generation_cell_completed record_id={} duration_seconds={:.3f} attempts={} "
                    "searches={} input_tokens={} output_tokens={}",
                    cell.record_id,
                    monotonic() - started,
                    attempts,
                    len(conversation.tool_calls),
                    conversation.usage.get("input_tokens", 0),
                    conversation.usage.get("output_tokens", 0),
                )
            except ProviderError as error:
                should_disable = (
                    error.transport_error
                    or error.quota_exhausted
                    or error.status_code
                    in {
                        401,
                        402,
                        403,
                        404,
                        410,
                        429,
                    }
                )
                if should_disable:
                    disabled_models.add(model_key)
                    logger.error(
                        "generation_circuit_open model_key={} model_id={} trigger_record_id={} "
                        "status_code={} quota_exhausted={} transport_error={}",
                        model_key,
                        cell.model.model_id,
                        cell.record_id,
                        error.status_code,
                        error.quota_exhausted,
                        error.transport_error,
                    )
                duration = monotonic() - started
                atomic_write_json(
                    path,
                    {
                        "record_id": cell.record_id,
                        "experiment_id": cell.experiment.experiment_id,
                        "model_id": cell.model.model_id,
                        "provider": cell.model.provider,
                        "condition": cell.condition,
                        "category": cell.experiment.category,
                        "language": cell.experiment.language,
                        "query_id": cell.query.id,
                        "query_text": cell.query.text,
                        "run_index": cell.run_index,
                        "request_parameters": {
                            "max_tokens": config.max_output_tokens,
                            "reasoning_effort": cell.model.reasoning_effort,
                            "thinking": thinking,
                            "request_timeout_seconds": cell.model.request_timeout_seconds,
                            "max_attempts": cell.model.max_attempts,
                            "tool_choice": ("auto" if cell.condition == "search_on" else None),
                            "max_tool_rounds": (
                                config.max_tool_rounds if cell.condition == "search_on" else 0
                            ),
                        },
                        "status": "error",
                        "error": {
                            "type": type(error).__name__,
                            "component": error.component,
                            "message": str(error),
                            "status_code": error.status_code,
                            "quota_exhausted": error.quota_exhausted,
                            "transport_error": error.transport_error,
                            "provider_disabled": should_disable,
                        },
                        "metadata": {
                            "started_at": started_at,
                            "completed_at": _now(),
                            "duration_seconds": round(duration, 3),
                            "attempts": error.attempts,
                        },
                    },
                )
                counts["errors"] += 1
                logger.error(
                    "generation_cell_failed record_id={} model={} component={} duration_seconds={:.3f} "
                    "attempts={} status_code={} message={}",
                    cell.record_id,
                    cell.model.model_id,
                    error.component,
                    duration,
                    error.attempts,
                    error.status_code,
                    error,
                )
        logger.info(
            "generation_model_finished model_key={} completed={} skipped={} errors={} blocked={}",
            model_key,
            counts["completed"],
            counts["skipped"],
            counts["errors"],
            counts["blocked"],
        )

    try:
        await asyncio.gather(
            *(run_model(model.key) for model in config.models if model.key in selected_keys)
        )
    finally:
        if owns_http:
            await session.aclose()
    result = {
        "planned": len(cells),
        "completed": counts["completed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "blocked": counts["blocked"],
        "disabled_models": sorted(disabled_models),
        "selected_models": sorted(selected_keys),
        "collection_fingerprint": collection_fingerprint(config),
    }
    logger.info("generation_suite_completed summary={}", result)
    return result
