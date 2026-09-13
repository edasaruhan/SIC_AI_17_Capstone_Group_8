"""Cerebras client, and a collector that records which service answered.

``visibility.intervention.collect`` is hash-pinned to the Gemini test and labels every
receipt "gemini"; this copy takes the service name so receipts stay honest. Gemini calls
still pass "gemini", so the request hashes of round one's receipts keep matching.

Like the Gemini client, no credential or provider body ever enters a message. Rate-limit
answers (429) are waited out a few times inside one call; every other failure is raised
at once and recorded by the step receipts, never retried silently.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

API_BASE = "https://api.cerebras.ai/v1"
KEY_ENV = "CEREBRAS_API_KEY"
# Free tier: 5 requests a minute and 150 an hour. One request every 25 s stays under both.
MIN_INTERVAL = 25.0
RATE_LIMIT_WAITS = (30.0, 60.0, 120.0)
STOP_AFTER_CONSECUTIVE_FAILURES = 5


def _seconds(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


class CerebrasClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        min_interval: float = MIN_INTERVAL,
        waits: tuple[float, ...] = RATE_LIMIT_WAITS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.http, self.min_interval, self.waits = http, min_interval, waits
        self.clock, self.sleep = clock, sleep
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def _post(self, key: str, payload: dict) -> httpx.Response:
        async with self._lock:
            if self._last is not None:
                wait = self.min_interval - (self.clock() - self._last)
                if wait > 0:
                    await self.sleep(wait)
            try:
                return await self.http.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
            finally:
                self._last = self.clock()

    async def __call__(self, service: str, payload: dict) -> dict:
        if service != "cerebras":
            raise ValueError("Unknown service")
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise ValueError(f"Eksik anahtar: {KEY_ENV}")
        for attempt in range(len(self.waits) + 1):
            try:
                response = await self._post(key, payload)
            except httpx.TransportError as exc:
                raise ValueError("cerebras bağlantı/timeout hatası; otomatik tekrar yok.") from exc
            retry = _seconds(response.headers.get("retry-after", ""))
            if response.status_code == 429 and attempt < len(self.waits):
                await self.sleep(retry if retry is not None else self.waits[attempt])
                continue
            if response.status_code != 200:
                raise ValueError(
                    f"cerebras HTTP {response.status_code}; "
                    f"Retry-After={retry if retry is not None else 'bilinmiyor'}. "
                    "Kota/anahtarı kontrol edin; otomatik tekrar yok."
                )
            return _parse(response)
        raise ValueError("cerebras hız sınırı aşılamadı")  # unreachable


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
        raise ValueError("cerebras yanıtı eksik/geçersiz/kesilmiş; otomatik tamir yok.") from exc


async def collect(
    jobs: list[dict],
    folder: Path,
    client: Callable[[str, dict], Awaitable[dict]],
    service: str,
    *,
    concurrency: int,
    retry_failed: bool,
) -> list[dict]:
    from brand_demo.workflow import Receipts

    receipts = Receipts(folder, client, retry_failed)
    gate = asyncio.Semaphore(concurrency)
    failures: list[dict] = []
    streak = {"failures": 0}

    async def one(entry: dict) -> None:
        async with gate:
            if streak["failures"] >= STOP_AFTER_CONSECUTIVE_FAILURES:
                failures.append({"key": entry["key"], "error": "skipped_after_failures"})
                return
            try:
                await receipts.call(entry["key"], service, entry["payload"])
                streak["failures"] = 0
            except Exception as exc:  # noqa: BLE001 - every failure is recorded, none retried
                streak["failures"] += 1
                failures.append(
                    {"key": entry["key"], "error": type(exc).__name__, "message": str(exc)[:300]}
                )

    await asyncio.gather(*(one(entry) for entry in jobs))
    return failures
