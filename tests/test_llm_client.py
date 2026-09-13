import asyncio

import httpx
import pytest

from visibility import llm

PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "q"}]}


def _completion(finish: str = "stop", text: str = "Use Mullvad.") -> dict:
    return {
        "choices": [{"finish_reason": finish, "message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _call(responses: list[httpx.Response], monkeypatch) -> tuple[object, list[httpx.Request]]:
    monkeypatch.setenv(llm.KEY_ENV, "test-key")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[len(seen) - 1]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await llm.GeminiClient(http, waits=(0.0, 0.0))("gemini", PAYLOAD)

    try:
        return asyncio.run(run()), seen
    except ValueError as exc:
        return exc, seen


def test_rate_limits_are_waited_out_and_usage_is_kept(monkeypatch) -> None:
    result, seen = _call(
        [httpx.Response(429), httpx.Response(200, json=_completion())], monkeypatch
    )
    assert result == {
        "text": "Use Mullvad.",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    assert len(seen) == 2
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert str(seen[0].url).endswith("/v1beta/openai/chat/completions")


def test_billing_errors_raise_once_without_leaking_the_body(monkeypatch) -> None:
    result, seen = _call([httpx.Response(402, json={"error": "secret detail"})], monkeypatch)
    assert isinstance(result, ValueError) and "HTTP 402" in str(result)
    assert "secret detail" not in str(result) and "test-key" not in str(result)
    assert len(seen) == 1


def test_persistent_rate_limit_and_truncation_fail(monkeypatch) -> None:
    result, seen = _call([httpx.Response(429)] * 3, monkeypatch)
    assert isinstance(result, ValueError) and "HTTP 429" in str(result) and len(seen) == 3
    result, _ = _call([httpx.Response(200, json=_completion(finish="length"))], monkeypatch)
    assert isinstance(result, ValueError) and "kesilmiş" in str(result)


def test_missing_key_and_unknown_service_are_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(llm.KEY_ENV, raising=False)
    with pytest.raises(ValueError, match="Eksik anahtar"):
        llm.load_key(path=tmp_path / "missing.env")
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nexport GEMINI_API_KEY="abc"\n')
    llm.load_key(path=env)
    assert llm.os.environ[llm.KEY_ENV] == "abc"

    async def run():
        async with httpx.AsyncClient() as http:
            await llm.GeminiClient(http)("minimax", PAYLOAD)

    with pytest.raises(ValueError, match="Unknown service"):
        asyncio.run(run())
