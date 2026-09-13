"""Gemini client for the controlled test, over the OpenAI-compatible endpoint.

Same endpoint and key variable as the collection pipeline
(``configs/evaluation/models.yaml``). Mirrors ``brand_demo.clients.LiveClient``: no
credential or provider body ever enters a message. Rate-limit answers (429) are not
billed, so they are waited out a few times inside one call; every other failure is
raised at once and recorded by the step receipts, never retried silently.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
KEY_ENV = "GEMINI_API_KEY"
RATE_LIMIT_WAITS = (20.0, 40.0, 60.0)


def load_key(name: str = KEY_ENV, path: Path = Path(".env")) -> None:
    """Read one literal value; never execute shell expressions from .env."""
    if not os.environ.get(name, "").strip() and path.exists():
        for line in path.read_text().splitlines():
            key, separator, value = line.removeprefix("export ").partition("=")
            if separator and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ[name] = value
    if not os.environ.get(name, "").strip():
        raise ValueError(f"Eksik anahtar: {name}")


def _failure(response: httpx.Response) -> ValueError:
    retry = response.headers.get("retry-after", "")
    wait = retry if retry.isdigit() else "bilinmiyor"
    return ValueError(
        f"gemini HTTP {response.status_code}; Retry-After={wait}. "
        "Kota/anahtar/paneli kontrol edin; otomatik tekrar yok."
    )


class GeminiClient:
    def __init__(self, http: httpx.AsyncClient, *, waits: tuple[float, ...] = RATE_LIMIT_WAITS):
        self.http = http
        self.waits = waits

    async def __call__(self, service: str, payload: dict) -> dict:
        if service != "gemini":
            raise ValueError("Unknown service")
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise ValueError(f"Eksik anahtar: {KEY_ENV}")
        for attempt in range(len(self.waits) + 1):
            try:
                response = await self.http.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
            except httpx.TransportError as exc:
                raise ValueError(
                    "gemini bağlantı/timeout hatası; otomatik tekrar yok. "
                    "İstek ücretlendirilmiş olabilir."
                ) from exc
            if response.status_code == 429 and attempt < len(self.waits):
                retry = response.headers.get("retry-after", "")
                await asyncio.sleep(float(retry) if retry.isdigit() else self.waits[attempt])
                continue
            if response.status_code != 200:
                raise _failure(response)
            return self._parse(response)
        raise ValueError("gemini hız sınırı aşılamadı")  # unreachable: the loop returns or raises

    @staticmethod
    def _parse(response: httpx.Response) -> dict:
        try:
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("truncated")
            text = str(choice["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty")
            usage = body.get("usage") or {}
            return {
                "text": text,
                "usage": {
                    k: int(usage.get(k) or 0)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(
                "gemini yanıtı eksik/geçersiz/kesilmiş; otomatik tamir çağrısı yok."
            ) from exc
