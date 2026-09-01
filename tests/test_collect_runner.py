import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from collect.config import CollectionSettings
from collect.providers import FatalProviderError, ModelConfig, adapter_for
from collect.rate_limit import TokenBucket
from collect.records import CallRecord, CallSpec
from collect.runner import execute_call, run_collection
from collect.storage import RawLogWriter, raw_log_path, scan_raw_log


def make_spec(index: int, *, run_id: str = "test-run", condition: str = "search_off") -> CallSpec:
    return CallSpec(
        run_id=run_id,
        layer="A",
        protocol="alpha",
        sector="kozmetik",
        query_id=f"kozmetik_{index:02d}",
        condition=condition,
        model_key="test_model",
        repetition=0,
        persona="Sen bir alışveriş danışmanısın.",
        temperature=0.0,
        prompt=f"Soru {index}\nSon satırda SIRALAMA yaz.",
        candidates=("koz_m01", "koz_m02", "koz_m03"),
    )


def anthropic_reply(text: str = "SIRALAMA: [A, B, C]") -> dict[str, object]:
    return {
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "test-key-not-real")


def run(specs, settings: CollectionSettings, handler, *, run_id: str = "test-run"):
    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await run_collection(specs, settings, run_id=run_id, client=client)

    return asyncio.run(scenario())


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- storage -----------------------------------------------------------------


def test_written_records_are_readable_and_indexed(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    spec = make_spec(1)
    record = CallRecord(
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
        variant_ids=[],
        status="ok",
        started_at="2026-09-01T10:00:00.000+00:00",
        completed_at="2026-09-01T10:00:01.000+00:00",
        latency_ms=1000,
        attempts=1,
        response_text="SIRALAMA: [A, B, C]",
    )

    with RawLogWriter(path) as writer:
        writer.write(record)

    state = scan_raw_log(path)
    assert state.completed == frozenset({spec.call_id})
    assert state.total_lines == 1
    assert CallRecord.from_dict(read_lines(path)[0]).response_text == "SIRALAMA: [A, B, C]"


def test_a_truncated_final_line_is_counted_not_ignored(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"call_id": "aaa", "status": "ok"}\n{"call_id": "bbb", "sta',
        encoding="utf-8",
    )

    state = scan_raw_log(path)

    assert state.completed == frozenset({"aaa"})
    assert state.malformed_lines == (2,)
    assert state.has_malformed_lines


def test_failed_calls_are_offered_for_retry(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        '{"call_id": "aaa", "status": "error"}\n{"call_id": "bbb", "status": "ok"}\n',
        encoding="utf-8",
    )

    state = scan_raw_log(path)

    assert state.failed == frozenset({"aaa"})
    assert state.completed == frozenset({"bbb"})


def test_run_id_must_be_a_usable_file_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file name"):
        raw_log_path(tmp_path, "pilot 01/A")


# --- one call ----------------------------------------------------------------


def test_successful_call_records_version_cost_and_timestamps(
    settings: CollectionSettings, model: ModelConfig
) -> None:
    async def scenario() -> CallRecord:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=anthropic_reply()))
        async with httpx.AsyncClient(transport=transport) as client:
            return await execute_call(
                client,
                make_spec(1),
                model=model,
                settings=settings,
                bucket=TokenBucket(rate_per_second=1000.0),
            )

    record = asyncio.run(scenario())

    assert record.succeeded
    assert record.model_version == "claude-haiku-4-5-20251001"
    assert record.response_text == "SIRALAMA: [A, B, C]"
    assert record.cost_usd == pytest.approx(100 / 1e6 * 1.0 + 20 / 1e6 * 5.0)
    assert record.started_at.endswith("+00:00") and record.completed_at.endswith("+00:00")
    assert record.attempts == 1


def test_transient_failures_are_retried_then_succeed(
    settings: CollectionSettings, model: ModelConfig
) -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(200, json=anthropic_reply())

    async def scenario() -> CallRecord:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_call(
                client,
                make_spec(1),
                model=model,
                settings=settings,
                bucket=TokenBucket(rate_per_second=1000.0),
            )

    record = asyncio.run(scenario())

    assert record.succeeded
    assert record.attempts == 3


def test_a_bad_request_is_recorded_without_retrying(
    settings: CollectionSettings, model: ModelConfig
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad model"})

    async def scenario() -> CallRecord:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_call(
                client,
                make_spec(1),
                model=model,
                settings=settings,
                bucket=TokenBucket(rate_per_second=1000.0),
            )

    record = asyncio.run(scenario())

    assert record.status == "error"
    assert record.error_type == "FatalProviderError"
    assert calls["count"] == 1, "a 400 must not be retried"
    assert record.response_text is None


def test_a_missing_api_key_fails_loudly(
    model: ModelConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    with pytest.raises(FatalProviderError, match="TEST_API_KEY"):
        model.api_key()


def test_search_tool_is_sent_only_for_the_search_on_condition(
    settings: CollectionSettings, model: ModelConfig
) -> None:
    searching = replace(model, supports_search=True)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=anthropic_reply())

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for condition in ("search_off", "search_on"):
                await execute_call(
                    client,
                    make_spec(1, condition=condition),
                    model=searching,
                    settings=settings,
                    bucket=TokenBucket(rate_per_second=1000.0),
                )

    asyncio.run(scenario())

    assert "tools" not in seen[0]
    assert seen[1]["tools"][0]["name"] == "web_search"
    assert seen[0]["temperature"] == 0.0


def test_a_paused_server_tool_turn_is_continued(
    settings: CollectionSettings, model: ModelConfig
) -> None:
    replies = [
        {
            "model": "claude-haiku-4-5",
            "stop_reason": "pause_turn",
            "content": [{"type": "text", "text": "Arama yapıyorum."}],
            "usage": {"input_tokens": 50, "output_tokens": 5},
        },
        anthropic_reply(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=replies.pop(0))

    async def scenario() -> CallRecord:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_call(
                client,
                make_spec(1, condition="search_on"),
                model=replace(model, supports_search=True),
                settings=settings,
                bucket=TokenBucket(rate_per_second=1000.0),
            )

    record = asyncio.run(scenario())

    assert record.succeeded
    assert "SIRALAMA: [A, B, C]" in (record.response_text or "")
    assert record.usage["input_tokens"] == 150, "usage from both turns must be billed"


def test_unknown_provider_is_refused(model: ModelConfig) -> None:
    with pytest.raises(FatalProviderError, match="unknown provider"):
        adapter_for(replace(model, provider="mistral"))


# --- a whole run -------------------------------------------------------------


def test_run_writes_one_line_per_call(settings: CollectionSettings) -> None:
    specs = [make_spec(index) for index in range(5)]

    summary = run(specs, settings, lambda request: httpx.Response(200, json=anthropic_reply()))

    assert summary.succeeded == 5 and summary.failed == 0
    assert len(read_lines(Path(summary.log_path))) == 5


def test_an_interrupted_run_resumes_where_it_stopped(settings: CollectionSettings) -> None:
    specs = [make_spec(index) for index in range(10)]
    handler = lambda request: httpx.Response(200, json=anthropic_reply())  # noqa: E731

    first = run(specs[:4], settings, handler)
    second = run(specs, settings, handler)

    assert first.succeeded == 4
    assert second.skipped == 4, "already collected calls must not be paid for twice"
    assert second.succeeded == 6
    lines = read_lines(Path(second.log_path))
    assert len(lines) == 10
    assert len({line["call_id"] for line in lines}) == 10


def test_lines_written_before_a_crash_survive_and_are_reused(
    settings: CollectionSettings,
) -> None:
    specs = [make_spec(index) for index in range(5)]
    serial = replace(settings, concurrency=1)
    seen = {"count": 0}

    def crashing(request: httpx.Request) -> httpx.Response:
        seen["count"] += 1
        if seen["count"] > 3:
            raise RuntimeError("simulated hard interruption")
        return httpx.Response(200, json=anthropic_reply())

    with pytest.raises(RuntimeError, match="simulated hard interruption"):
        run(specs, serial, crashing)

    path = raw_log_path(serial.raw_dir, "test-run")
    assert len(read_lines(path)) == 3, "each finished call must already be on disk"

    resumed = run(specs, serial, lambda request: httpx.Response(200, json=anthropic_reply()))

    assert resumed.skipped == 3
    assert len(read_lines(path)) == 5


def test_failures_are_recorded_with_their_type(settings: CollectionSettings) -> None:
    specs = [make_spec(index) for index in range(3)]

    summary = run(specs, settings, lambda request: httpx.Response(400, json={"error": "nope"}))

    assert summary.failed == 3
    assert summary.errors_by_type == {"FatalProviderError": 3}
    assert summary.error_rate == 1.0
    assert all(line["status"] == "error" for line in read_lines(Path(summary.log_path)))


def test_the_spend_cap_stops_the_run(settings: CollectionSettings) -> None:
    capped = replace(settings, daily_spend_cap_usd=0.0005, concurrency=1)
    specs = [make_spec(index) for index in range(20)]

    summary = run(specs, capped, lambda request: httpx.Response(200, json=anthropic_reply()))

    assert summary.stopped_on_spend_cap
    assert summary.succeeded < 20
    assert summary.spend_usd >= 0.0005


def test_a_fully_collected_run_makes_no_calls(settings: CollectionSettings) -> None:
    specs = [make_spec(index) for index in range(3)]
    run(specs, settings, lambda request: httpx.Response(200, json=anthropic_reply()))

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call should be made on a completed run")

    summary = run(specs, settings, refuse)

    assert summary.skipped == 3 and summary.succeeded == 0


def test_attempt_counts_stay_correct_under_concurrency(settings: CollectionSettings) -> None:
    specs = [make_spec(index) for index in range(4)]
    retried = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "Soru 1\n" in payload["messages"][0]["content"] and retried["count"] < 2:
            retried["count"] += 1
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(200, json=anthropic_reply())

    summary = run(specs, settings, handler)

    attempts = {line["query_id"]: line["attempts"] for line in read_lines(Path(summary.log_path))}
    assert summary.succeeded == 4
    assert attempts["kozmetik_01"] == 3, "retries must be attributed to the call that retried"
    assert [attempts[key] for key in ("kozmetik_00", "kozmetik_02", "kozmetik_03")] == [1, 1, 1]
