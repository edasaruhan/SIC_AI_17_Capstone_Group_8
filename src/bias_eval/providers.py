"""OpenAI-compatible generation client shared by the four providers."""

from __future__ import annotations

import json
import os
from time import monotonic
from typing import Any

import httpx

from .config import ModelConfig
from .logging import logger, sanitize_for_log
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
        transport_error: bool = False,
        component: str = "generation",
    ) -> None:
        super().__init__(sanitize_for_log(message))
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable
        self.quota_exhausted = quota_exhausted
        self.transport_error = transport_error
        self.component = component
        self.attempts = 1


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
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]
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
        thinking: bool | None = None,
    ) -> ProviderTurn:
        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if thinking is not None:
            if self.config.provider != "abliteration":
                raise ValueError("thinking override is supported only for Abliteration")
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        started = monotonic()
        logger.debug(
            "generation_request provider={} model={} messages={} tools={} timeout_seconds={} "
            "max_tokens={} reasoning_effort={} thinking={}",
            self.config.provider,
            self.config.model_id,
            len(messages),
            bool(tools),
            self.config.request_timeout_seconds,
            max_tokens,
            self.config.reasoning_effort,
            thinking,
        )
        try:
            response = await self.http.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key(self.config.api_key_env)}"},
                json=payload,
                timeout=httpx.Timeout(self.config.request_timeout_seconds),
            )
        except httpx.TransportError as error:
            logger.error(
                "generation_transport_error provider={} model={} duration_seconds={:.3f} "
                "error_type={}",
                self.config.provider,
                self.config.model_id,
                monotonic() - started,
                type(error).__name__,
            )
            raise ProviderError(
                "Provider transport error",
                retryable=True,
                transport_error=True,
            ) from error
        if response.status_code >= 400:
            message = _safe_error_text(response)
            lowered = message.casefold()
            quota = response.status_code in {402, 429} and any(
                word in lowered for word in ("quota", "credit", "balance", "billing")
            )
            logger.warning(
                "generation_http_error provider={} model={} status_code={} duration_seconds={:.3f} "
                "retry_after={} message={}",
                self.config.provider,
                self.config.model_id,
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
            )
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            logger.error(
                "generation_invalid_response provider={} model={} status_code={} "
                "duration_seconds={:.3f} error_type={}",
                self.config.provider,
                self.config.model_id,
                response.status_code,
                monotonic() - started,
                type(error).__name__,
            )
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
                calls.append(
                    ToolCall(
                        str(item["id"]),
                        str(function["name"]),
                        arguments,
                        raw=dict(item),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                logger.error(
                    "generation_malformed_tool_call provider={} model={} error_type={}",
                    self.config.provider,
                    self.config.model_id,
                    type(error).__name__,
                )
                raise ProviderError("Provider returned malformed tool arguments") from error
        usage = body.get("usage") or {}
        turn = ProviderTurn(
            content=message.get("content"),
            tool_calls=tuple(calls),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            raw=body,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )
        logger.info(
            "generation_response provider={} model={} status_code={} duration_seconds={:.3f} "
            "finish_reason={} tool_calls={} input_tokens={} output_tokens={}",
            self.config.provider,
            self.config.model_id,
            response.status_code,
            monotonic() - started,
            turn.finish_reason,
            len(turn.tool_calls),
            turn.usage["input_tokens"],
            turn.usage["output_tokens"],
        )
        return turn


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
