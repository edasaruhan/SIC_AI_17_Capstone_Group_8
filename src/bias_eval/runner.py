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

    for round_index in range(rounds):
        turn = await client.chat(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            tools=tools,
        )
        all_raw.append(turn.raw)
        _add_usage(usage, turn.usage)
        if not turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.content})
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
            assistant_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function_name,
                        "arguments": __import__("json").dumps(call.arguments, ensure_ascii=False),
                    },
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function_name,
                    "content": hit.formatted,
                }
            )
        messages.insert(
            len(messages) - len(turn.tool_calls),
            {"role": "assistant", "content": turn.content, "tool_calls": assistant_calls},
        )
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
            if not error.retryable or attempt == max_attempts:
                raise
            delay = (
                error.retry_after if error.retry_after is not None else float(2 ** (attempt - 1))
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
    http: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, Any]:
    cells = pilot_cells(config) if pilot else plan_cells(config)
    owns_http = http is None
    session = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    search = SerperSearch(config.search, config.search_cache_dir, session)
    counts: Counter[str] = Counter()
    disabled_models: set[str] = set()

    async def run_model(model_key: str) -> None:
        model_cells = [cell for cell in cells if cell.model.key == model_key]
        client = OpenAICompatibleClient(model_cells[0].model, session)
        for cell in model_cells:
            path = result_path(config.raw_dir, cell)
            if is_complete(path):
                counts["skipped"] += 1
                continue
            if model_key in disabled_models:
                counts["blocked"] += 1
                continue
            started_at = _now()
            started = monotonic()
            try:
                conversation, attempts = await _with_retries(
                    lambda cell=cell: run_conversation(client, search, cell, config),
                    sleep=sleep,
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
                    ),
                )
                counts["completed"] += 1
            except ProviderError as error:
                if error.quota_exhausted:
                    disabled_models.add(model_key)
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
                        "status": "error",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "status_code": error.status_code,
                            "quota_exhausted": error.quota_exhausted,
                        },
                        "metadata": {"started_at": started_at, "completed_at": _now()},
                    },
                )
                counts["errors"] += 1

    try:
        await asyncio.gather(*(run_model(model.key) for model in config.models))
    finally:
        if owns_http:
            await session.aclose()
    return {
        "planned": len(cells),
        "completed": counts["completed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "blocked": counts["blocked"],
        "disabled_models": sorted(disabled_models),
    }
