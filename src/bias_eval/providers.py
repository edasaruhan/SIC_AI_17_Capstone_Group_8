"""OpenAI-compatible generation client shared by the four providers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .config import ModelConfig
from .records import ProviderTurn, ToolCall

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable
        self.quota_exhausted = quota_exhausted


def api_key(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ProviderError(f"Missing environment variable: {env_name}")
    return value


def _safe_error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return f"HTTP {response.status_code}"
    error = body.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or "provider error")
    else:
        message = str(error or body.get("message") or "provider error")
    return f"HTTP {response.status_code}: {message[:300]}"


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> ProviderTurn:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = await self.http.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key(self.config.api_key_env)}"},
                json=payload,
            )
        except httpx.TransportError as error:
            raise ProviderError("Provider transport error", retryable=True) from error
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
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "Provider returned an invalid completion", retryable=True
            ) from error
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            try:
                function = item["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")
                calls.append(ToolCall(str(item["id"]), str(function["name"]), arguments))
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ProviderError("Provider returned malformed tool arguments") from error
        usage = body.get("usage") or {}
        return ProviderTurn(
            content=message.get("content"),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            raw=body,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Güncel bilgiler için Google'da arama yapar.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
