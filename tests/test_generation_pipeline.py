from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from bias_eval.config import load_suite
from bias_eval.preflight import check_connections
from bias_eval.providers import OpenAICompatibleClient, ProviderError, _safe_error_text
from bias_eval.records import ProviderTurn, ToolCall, pilot_cells
from bias_eval.runner import _with_retries, run_conversation, run_suite
from bias_eval.search import SearchHit, SerperSearch
from bias_eval.storage import atomic_write_json, is_complete, read_json, status_summary


class FakeGenerationClient:
    def __init__(self, turns: list[ProviderTurn]):
        self.turns = turns
        self.requests: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        thinking: bool | None = None,
    ) -> ProviderTurn:
        self.requests.append({"messages": list(messages), "tools": tools, "thinking": thinking})
        return self.turns.pop(0)


class FakeSearch:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str) -> SearchHit:
        self.queries.append(query)
        results = {
            "organic": [
                {"title": f"Title {query}", "snippet": "Snippet", "link": "https://example.test"}
            ]
        }
        return SearchHit(
            query, "fixture", results, "1. Title\n   Snippet\n   URL: https://example.test"
        )


def turn(
    content: str | None,
    *calls: ToolCall,
) -> ProviderTurn:
    return ProviderTurn(
        content, calls, "stop", {"fixture": True}, {"input_tokens": 2, "output_tokens": 3}
    )


def test_search_off_sends_only_user_message_and_no_tools(tmp_path: Path) -> None:
    config = replace(load_suite(), search_cache_dir=tmp_path)
    cell = next(cell for cell in pilot_cells(config) if cell.condition == "search_off")
    client = FakeGenerationClient([turn("Yanıt")])

    result = asyncio.run(run_conversation(client, FakeSearch(), cell, config))  # type: ignore[arg-type]

    assert client.requests == [
        {
            "messages": [{"role": "user", "content": cell.query.text}],
            "tools": None,
            "thinking": None,
        }
    ]
    assert result.tool_calls == []
    assert result.search_results == []


def test_empty_final_response_is_rejected_for_retry(tmp_path: Path) -> None:
    config = replace(load_suite(), search_cache_dir=tmp_path)
    cell = next(cell for cell in pilot_cells(config) if cell.condition == "search_off")
    client = FakeGenerationClient([turn(None)])

    with pytest.raises(ProviderError, match="empty final response"):
        asyncio.run(run_conversation(client, FakeSearch(), cell, config))  # type: ignore[arg-type]


def test_search_on_supports_parallel_calls_and_returns_tool_messages(tmp_path: Path) -> None:
    config = replace(load_suite(), search_cache_dir=tmp_path)
    cell = next(cell for cell in pilot_cells(config) if cell.condition == "search_on")
    client = FakeGenerationClient(
        [
            turn(
                None,
                ToolCall(
                    "one",
                    "web_search",
                    {"query": "güncel vpn"},
                    raw={
                        "id": "one",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"güncel vpn"}',
                        },
                        "extra_content": {"google": {"thought_signature": "fixture-signature"}},
                    },
                ),
                ToolCall("two", "web_search", {"query": "vpn testleri"}),
            ),
            turn("Son yanıt"),
        ]
    )
    search = FakeSearch()

    result = asyncio.run(run_conversation(client, search, cell, config))  # type: ignore[arg-type]

    assert search.queries == ["güncel vpn", "vpn testleri"]
    assert len(result.tool_calls) == 2
    second_messages = client.requests[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert "content" not in second_messages[2]
    assert second_messages[2]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "fixture-signature"}
    }
    assert result.final_content == "Son yanıt"


def test_search_on_enforces_five_tool_round_limit(tmp_path: Path) -> None:
    config = replace(load_suite(), search_cache_dir=tmp_path)
    cell = next(cell for cell in pilot_cells(config) if cell.condition == "search_on")
    client = FakeGenerationClient(
        [
            turn(None, ToolCall(str(index), "web_search", {"query": f"q{index}"}))
            for index in range(5)
        ]
    )

    with pytest.raises(ProviderError, match="exceeded 5"):
        asyncio.run(run_conversation(client, FakeSearch(), cell, config))  # type: ignore[arg-type]


@pytest.mark.parametrize("model_index", range(3))
def test_all_provider_adapters_parse_tools_and_reject_malformed_arguments(
    model_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_suite().models[model_index]
    monkeypatch.setenv(config.api_key_env, "fixture-key")
    bodies = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "extra_content": {
                                    "google": {"thought_signature": "fixture-signature"}
                                },
                                "function": {"name": "web_search", "arguments": '{"query":"vpn"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "bad", "function": {"name": "web_search", "arguments": "{"}}
                        ]
                    }
                }
            ]
        },
    ]
    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fixture-key"
        request_payloads.append(json.loads(request.content))
        return httpx.Response(200, json=bodies.pop(0))

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenAICompatibleClient(config, http)
            parsed = await client.chat([], temperature=0.7, max_tokens=10, tools=[])
            assert parsed.tool_calls[0].arguments == {"query": "vpn"}
            assert parsed.tool_calls[0].raw == {
                "id": "call-1",
                "extra_content": {"google": {"thought_signature": "fixture-signature"}},
                "function": {"name": "web_search", "arguments": '{"query":"vpn"}'},
            }
            with pytest.raises(ProviderError, match="malformed"):
                await client.chat([], temperature=0.7, max_tokens=10, tools=[])

    asyncio.run(exercise())
    assert all("reasoning_effort" not in payload for payload in request_payloads)
    assert all("thinking" not in payload for payload in request_payloads)


def test_abliteration_recovery_disables_thinking_and_records_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_suite()
    config = replace(suite, raw_dir=tmp_path / "raw", search_cache_dir=tmp_path / "cache")
    model = next(item for item in config.models if item.key == "glm_5_3_abliteration")
    monkeypatch.setenv(model.api_key_env, "fixture-key")
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Fixture response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    async def exercise() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await run_suite(
                config,
                pilot=True,
                model_keys={model.key},
                http=http,
                disable_abliteration_thinking=True,
            )

    result = asyncio.run(exercise())
    records = [
        record
        for path in (tmp_path / "raw").glob("*/*/*/*.json")
        if (record := read_json(path)) is not None
    ]

    assert result["completed"] == 4
    assert len(payloads) == 4
    assert all(payload["thinking"] is False for payload in payloads)
    assert all(record["request_parameters"]["thinking"] is False for record in records)


def test_serper_cache_is_locale_aware_and_avoids_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_suite()
    monkeypatch.setenv(suite.search.api_key_env, "fixture-serper")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"organic": [{"title": "A", "snippet": "B", "link": "https://example.test"}]},
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            search = SerperSearch(suite.search, tmp_path, http)
            first = await search.search(" En iyi VPN ")
            second = await search.search("En iyi VPN")
            other_locale = SerperSearch(replace(suite.search, language="en"), tmp_path, http)
            assert search.cache_key("vpn") != other_locale.cache_key("vpn")
            assert first.source == "live"
            assert second.source == "cache"

    asyncio.run(exercise())
    assert requests == [
        {"q": "En iyi VPN", "num": 10, "gl": "tr", "hl": "tr", "location": "Turkey"}
    ]


def test_retry_honors_retry_after_without_real_sleep() -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderError("temporary", retryable=True, retry_after=0.25)
        return "done"

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result, used_attempts = asyncio.run(_with_retries(operation, sleep=sleep))
    assert result == "done"
    assert used_attempts == 3
    assert delays == [0.25, 0.25]


def test_quota_error_is_permanent_and_not_retried() -> None:
    attempts = 0

    async def operation() -> Any:
        nonlocal attempts
        attempts += 1
        raise ProviderError("quota exhausted", quota_exhausted=True)

    with pytest.raises(ProviderError, match="quota"):
        asyncio.run(_with_retries(operation, sleep=asyncio.sleep))
    assert attempts == 1


def test_atomic_storage_replaces_broken_or_partial_record(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("broken", encoding="utf-8")
    assert not is_complete(path)

    atomic_write_json(path, {"status": "completed", "secret": None})

    assert read_json(path) == {"secret": None, "status": "completed"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_completed_record_requires_a_nonempty_final_response(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"status": "completed", "final_response": None})
    assert not is_complete(path)

    atomic_write_json(path, {"status": "completed", "final_response": "  "})
    assert not is_complete(path)

    atomic_write_json(path, {"status": "completed", "final_response": "Yanıt"})
    assert is_complete(path)


def test_pilot_resume_uses_mock_transport_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_suite()
    config = replace(
        suite,
        raw_dir=tmp_path / "raw",
        search_cache_dir=tmp_path / "cache",
    )
    for model in config.models:
        monkeypatch.setenv(model.api_key_env, "fixture-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Fixture response"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    async def exercise() -> tuple[dict[str, Any], dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            first = await run_suite(config, pilot=True, http=http)
            second = await run_suite(config, pilot=True, http=http)
            return first, second

    first, second = asyncio.run(exercise())
    assert first["completed"] == 12
    assert second["skipped"] == 12
    assert len(list((tmp_path / "raw").glob("*/*/*/*.json"))) == 12


@pytest.mark.parametrize(
    ("model_key", "status_code", "expected_requests"),
    [
        ("glm_5_3_abliteration", 410, 1),
        ("gemini_3_5_flash_lite", 429, 3),
    ],
)
def test_filtered_run_stops_provider_after_permanent_or_repeated_limit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    status_code: int,
    expected_requests: int,
) -> None:
    suite = load_suite()
    config = replace(suite, raw_dir=tmp_path / "raw", search_cache_dir=tmp_path / "cache")
    model = next(item for item in config.models if item.key == model_key)
    monkeypatch.setenv(model.api_key_env, "fixture-key")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(status_code, json={})

    async def no_sleep(delay: float) -> None:
        return None

    async def exercise() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await run_suite(
                config,
                model_keys={model_key},
                http=http,
                sleep=no_sleep,
            )

    result = asyncio.run(exercise())
    assert result["planned"] == 100
    assert result["errors"] == 1
    assert result["blocked"] == 99
    assert result["disabled_models"] == [model_key]
    assert requests == expected_requests


def test_status_excludes_superseded_model_files(tmp_path: Path) -> None:
    atomic_write_json(
        tmp_path / "a" / "b" / "c" / "active.json",
        {
            "record_id": "active",
            "status": "completed",
            "final_response": "Fixture response",
        },
    )
    atomic_write_json(
        tmp_path / "a" / "b" / "c" / "old.json",
        {
            "record_id": "old-preview-model",
            "status": "error",
        },
    )

    result = status_summary(tmp_path, 300, {"active"})

    assert result["completed"] == 1
    assert result["errors"] == 0
    assert result["superseded_files"] == 1


def test_status_reports_empty_completed_response_as_invalid(tmp_path: Path) -> None:
    atomic_write_json(
        tmp_path / "a" / "b" / "c" / "invalid.json",
        {"record_id": "invalid", "status": "completed", "final_response": None},
    )

    result = status_summary(tmp_path, 1, {"invalid"})

    assert result["completed"] == 0
    assert result["remaining"] == 1
    assert result["by_status"] == {"invalid": 1}


def test_filtered_preflight_checks_exact_model_ids_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite()
    selected = {"gemini_3_5_flash_lite", "glm_5_3_abliteration"}
    for model in suite.models:
        if model.key in selected:
            monkeypatch.setenv(model.api_key_env, "fixture-key")
    monkeypatch.setenv(suite.search.api_key_env, "fixture-serper")
    available_ids = [
        f"models/{model.model_id}" if model.provider == "google" else model.model_id
        for model in suite.models
        if model.key in selected
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": value} for value in available_ids]})

    async def exercise() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await check_connections(
                suite,
                model_keys=selected,
                include_judge=False,
                http=http,
            )

    result = asyncio.run(exercise())
    assert set(result) == {*selected, "serper"}
    assert all(result[key]["model_available"] for key in selected)


def test_provider_error_does_not_echo_authorization_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_suite().models[0]
    secret = "do-not-leak-this-key"
    monkeypatch.setenv(config.api_key_env, secret)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    async def exercise() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(ProviderError) as captured:
                await OpenAICompatibleClient(config, http).chat(
                    [], temperature=0.7, max_tokens=10, tools=None
                )
            return str(captured.value)

    assert secret not in asyncio.run(exercise())


def test_safe_error_text_parses_gemini_list_error_body() -> None:
    response = httpx.Response(
        400,
        json=[
            {
                "error": {
                    "code": 400,
                    "message": "Expected string or list of content parts, got: null",
                    "status": "INVALID_ARGUMENT",
                }
            }
        ],
    )

    assert _safe_error_text(response) == (
        "HTTP 400: Expected string or list of content parts, got: null"
    )
