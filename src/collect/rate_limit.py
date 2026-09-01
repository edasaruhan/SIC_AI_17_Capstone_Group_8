"""Token-bucket pacing so a collection run stays inside each provider's limit."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class TokenBucket:
    """Pace concurrent callers to a sustained rate with a bounded burst.

    The bucket refills continuously at ``rate_per_second`` and never holds more
    than ``capacity`` tokens. ``acquire`` serialises waiters behind a lock, so
    the sustained rate is respected no matter how many tasks are queued.
    """

    def __init__(
        self,
        *,
        rate_per_second: float,
        capacity: float | None = None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        resolved_capacity = rate_per_second if capacity is None else capacity
        if resolved_capacity <= 0:
            raise ValueError("capacity must be positive")

        self._rate = float(rate_per_second)
        self._capacity = float(resolved_capacity)
        self._tokens = float(resolved_capacity)
        self._now = now
        self._sleep = sleep
        self._updated_at = now()
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    def _refill(self) -> None:
        current = self._now()
        elapsed = max(0.0, current - self._updated_at)
        self._updated_at = current
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until ``tokens`` are available and return the seconds waited."""

        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self._capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens from a bucket of {self._capacity}")

        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                delay = (tokens - self._tokens) / self._rate
                waited += delay
                await self._sleep(delay)
