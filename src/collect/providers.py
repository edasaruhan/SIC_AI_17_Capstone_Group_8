"""Provider adapters that turn one ``CallSpec`` into one normalised response.

Every provider is called through the same httpx client, rate limiter and retry
policy. Treating the three assistants identically is a measurement requirement:
a difference in results has to come from the model, not from how we called it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

ANTHROPIC_VERSION = "2023-06-01"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class ProviderError(RuntimeError):
    """Base class for a failed provider call."""


class RetryableProviderError(ProviderError):
    """Transient failure - rate limit, timeout or server error."""


class FatalProviderError(ProviderError):
    """Permanent failure - bad request, bad credentials, unknown model."""


@dataclass(frozen=True)
class ModelConfig:
    """One measured assistant, including how fast and how dearly we may call it."""

    key: str
    provider: str
    model: str
    api_key_env: str
    base_url: str
    max_output_tokens: int = 1024
    requests_per_second: float = 1.0
    burst: float = 4.0
    timeout_seconds: float = 120.0
    supports_search: bool = False
    search_tool_type: str = "web_search_20250305"
    max_search_uses: int = 5
    max_continuations: int = 3
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    search_cost_per_request: float = 0.0

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            raise ValueError(f"{self.key}: requests_per_second must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError(f"{self.key}: max_output_tokens must be positive")

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise FatalProviderError(
                f"{self.key}: environment variable {self.api_key_env} is unset"
            )
        return key


@dataclass(frozen=True)
class ProviderResponse:
    """The provider-independent view the rest of the pipeline works from."""

    text: str
    model_version: str
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self, model: ModelConfig) -> float:
        input_tokens = int(self.usage.get("input_tokens", 0) or 0)
        output_tokens = int(self.usage.get("output_tokens", 0) or 0)
        search_requests = int(self.usage.get("search_requests", 0) or 0)
        return (
            input_tokens / 1_000_000 * model.input_cost_per_million
            + output_tokens / 1_000_000 * model.output_cost_per_million
            + search_requests * model.search_cost_per_request
        )


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST once and classify the outcome as retryable or fatal."""

    try:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.TimeoutException as error:
        raise RetryableProviderError(f"Timeout calling {url}") from error
    except httpx.TransportError as error:
        raise RetryableProviderError(f"Transport error calling {url}: {error}") from error

    if response.status_code in RETRYABLE_STATUS_CODES:
        raise RetryableProviderError(
            f"HTTP {response.status_code} from {url}: {response.text[:400]}"
        )
    if response.status_code >= 400:
        raise FatalProviderError(f"HTTP {response.status_code} from {url}: {response.text[:400]}")

    try:
        body = response.json()
    except ValueError as error:
        raise RetryableProviderError(f"Non-JSON response from {url}") from error
    if not isinstance(body, dict):
        raise RetryableProviderError(f"Unexpected JSON shape from {url}: {type(body).__name__}")
    return body


class ProviderAdapter(ABC):
    """Builds the request for one provider and normalises its response."""

    name: ClassVar[str]

    @abstractmethod
    async def complete(
        self,
        client: httpx.AsyncClient,
        model: ModelConfig,
        *,
        system: str,
        prompt: str,
        temperature: float,
        use_search: bool,
    ) -> ProviderResponse: ...


class AnthropicAdapter(ProviderAdapter):
    """Messages API, including the server-side web search tool."""

    name = "anthropic"

    async def complete(
        self,
        client: httpx.AsyncClient,
        model: ModelConfig,
        *,
        system: str,
        prompt: str,
        temperature: float,
        use_search: bool,
    ) -> ProviderResponse:
        base = model.base_url.rstrip("/")
        url = f"{base}/v1/messages"
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": model.model,
            "max_tokens": model.max_output_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        if use_search:
            payload["tools"] = [
                {
                    "type": model.search_tool_type,
                    "name": "web_search",
                    "max_uses": model.max_search_uses,
                }
            ]

        headers = {
            "x-api-key": model.api_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        blocks: list[dict[str, Any]] = []
        body: dict[str, Any] = {}
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "search_requests": 0}

        # A server-tool turn can pause; continue it instead of storing half an answer.
        for _ in range(model.max_continuations + 1):
            body = await post_json(
                client,
                url,
                headers=headers,
                payload=payload,
                timeout=model.timeout_seconds,
            )
            content = body.get("content")
            if not isinstance(content, list):
                raise RetryableProviderError("Anthropic response has no content list")
            blocks.extend(block for block in content if isinstance(block, dict))

            usage = body.get("usage") or {}
            usage_totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            usage_totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            server_tool_use = usage.get("server_tool_use") or {}
            usage_totals["search_requests"] += int(
                server_tool_use.get("web_search_requests", 0) or 0
            )

            if body.get("stop_reason") != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": content}]
            payload = {**payload, "messages": messages}

        text = "\n".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )
        return ProviderResponse(
            text=text.strip(),
            model_version=str(body.get("model") or model.model),
            stop_reason=body.get("stop_reason"),
            usage=usage_totals,
            search_results=self._search_results(blocks),
            raw=body,
        )

    def _search_results(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") != "web_search_tool_result":
                continue
            content = block.get("content")
            # An errored server tool returns an object here instead of a list.
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "web_search_result":
                    continue
                results.append(
                    {
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "snippet": item.get("page_age"),
                    }
                )
        return results


class OpenAIAdapter(ProviderAdapter):
    """Chat Completions, with hosted web search when the condition requires it."""

    name = "openai"

    async def complete(
        self,
        client: httpx.AsyncClient,
        model: ModelConfig,
        *,
        system: str,
        prompt: str,
        temperature: float,
        use_search: bool,
    ) -> ProviderResponse:
        base = model.base_url.rstrip("/")
        payload: dict[str, Any] = {
            "model": model.model,
            "temperature": temperature,
            "max_tokens": model.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if use_search:
            payload["web_search_options"] = {}

        body = await post_json(
            client,
            f"{base}/v1/chat/completions",
            headers={
                "authorization": f"Bearer {model.api_key()}",
                "content-type": "application/json",
            },
            payload=payload,
            timeout=model.timeout_seconds,
        )

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RetryableProviderError("OpenAI response has no choices")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}

        return ProviderResponse(
            text=str(message.get("content") or "").strip(),
            model_version=str(body.get("model") or model.model),
            stop_reason=choices[0].get("finish_reason"),
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                "search_requests": 1 if use_search else 0,
            },
            search_results=self._search_results(message),
            raw=body,
        )

    def _search_results(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        annotations = message.get("annotations")
        if not isinstance(annotations, list):
            return []
        results: list[dict[str, Any]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                continue
            citation = annotation.get("url_citation") or {}
            results.append(
                {
                    "url": citation.get("url"),
                    "title": citation.get("title"),
                    "snippet": citation.get("content"),
                }
            )
        return results


class GeminiAdapter(ProviderAdapter):
    """generateContent, with Google Search grounding when the condition requires it."""

    name = "gemini"

    async def complete(
        self,
        client: httpx.AsyncClient,
        model: ModelConfig,
        *,
        system: str,
        prompt: str,
        temperature: float,
        use_search: bool,
    ) -> ProviderResponse:
        base = model.base_url.rstrip("/")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": model.max_output_tokens,
            },
        }
        if use_search:
            payload["tools"] = [{"google_search": {}}]

        body = await post_json(
            client,
            f"{base}/v1beta/models/{model.model}:generateContent",
            headers={
                "x-goog-api-key": model.api_key(),
                "content-type": "application/json",
            },
            payload=payload,
            timeout=model.timeout_seconds,
        )

        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RetryableProviderError("Gemini response has no candidates")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        usage = body.get("usageMetadata") or {}

        text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        return ProviderResponse(
            text=text.strip(),
            model_version=str(body.get("modelVersion") or model.model),
            stop_reason=candidate.get("finishReason"),
            usage={
                "input_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "search_requests": 1 if use_search else 0,
            },
            search_results=self._search_results(candidate),
            raw=body,
        )

    def _search_results(self, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        grounding = candidate.get("groundingMetadata") or {}
        chunks = grounding.get("groundingChunks")
        if not isinstance(chunks, list):
            return []
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            web = chunk.get("web") if isinstance(chunk, dict) else None
            if not isinstance(web, dict):
                continue
            results.append(
                {
                    "url": web.get("uri"),
                    "title": web.get("title"),
                    "snippet": web.get("snippet"),
                }
            )
        return results


ADAPTERS: dict[str, ProviderAdapter] = {
    adapter.name: adapter for adapter in (AnthropicAdapter(), OpenAIAdapter(), GeminiAdapter())
}


def adapter_for(model: ModelConfig) -> ProviderAdapter:
    try:
        return ADAPTERS[model.provider]
    except KeyError as error:
        raise FatalProviderError(
            f"{model.key}: unknown provider {model.provider!r}; expected one of {sorted(ADAPTERS)}"
        ) from error
