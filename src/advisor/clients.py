"""One client for the two services the advisor uses, reusing the existing ones.

``GeminiClient`` is the client the controlled test ran on, so the advisor asks the
same assistant whose effect sizes it quotes. ``LiveClient`` already speaks Serper and
already refuses to let a provider body or a key reach a message.
"""

from __future__ import annotations

import httpx

from brand_demo.clients import LiveClient
from visibility.llm import GeminiClient, load_key


class AdvisorClient:
    def __init__(self, http: httpx.AsyncClient):
        self.gemini = GeminiClient(http)
        self.serper = LiveClient(http)

    async def __call__(self, service: str, payload: dict) -> dict:
        if service == "gemini":
            return await self.gemini("gemini", payload)
        if service == "serper":
            return await self.serper("serper", payload)
        raise ValueError(f"Bilinmeyen servis: {service}")


def require_keys() -> None:
    """Fail before any paid call if a key is missing, not halfway through."""
    load_key("GEMINI_API_KEY")
    load_key("SERPER_API_KEY")
