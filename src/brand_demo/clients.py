"""Explicit single attempts; credentials and hidden reasoning are never persisted."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .core import final_text


class LiveClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    async def __call__(self, service: str, payload: dict) -> dict:
        if service not in {"minimax", "serper"}:
            raise ValueError("Unknown service")
        env = "MINIMAX_API_KEY" if service == "minimax" else "SERPER_API_KEY"
        key = os.environ.get(env, "").strip()
        if not key:
            raise ValueError(f"Eksik anahtar: {env}")
        url = (
            "https://api.minimax.io/v1/chat/completions"
            if service == "minimax"
            else "https://google.serper.dev/search"
        )
        headers = {"Authorization": f"Bearer {key}"} if service == "minimax" else {"X-API-KEY": key}
        try:
            response = await self.http.post(url, headers=headers, json=payload)
        except httpx.TransportError as exc:
            raise ValueError(
                "Bağlantı/timeout hatası; otomatik tekrar yok. İstek ücretlendirilmiş olabilir."
            ) from exc
        if response.status_code != 200:
            retry_after = response.headers.get("retry-after", "bilinmiyor")
            # Never propagate a provider body, request header or unsanitized response text.
            wait = retry_after if retry_after.isdigit() else "bilinmiyor"
            raise ValueError(
                f"{service} HTTP {response.status_code}; Retry-After={wait}. Kota/anahtar/paneli kontrol edin; otomatik tekrar yok."
            )
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("non-object")
            if service == "serper":
                if not isinstance(body.get("organic", []), list) or body.get("error"):
                    raise ValueError("invalid search result")
                return {"organic": body.get("organic", [])}
            if body.get("base_resp", {}).get("status_code", 0) != 0:
                raise ValueError("provider status error")
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("tool_calls"):
                raise ValueError("truncated or unexpected tools")
            return {
                "text": final_text(choice["message"].get("content")),
                "usage": {
                    k: int(body.get("usage", {}).get(k, 0))
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(
                f"{service} yanıtı eksik/geçersiz/kesilmiş; otomatik tamir çağrısı yok."
            ) from exc


class FixtureClient:
    """Deliberately synthetic data for exercising the CLI with zero keys/network."""

    def __init__(self, brand: str):
        self.brand = brand

    async def __call__(self, service: str, payload: dict) -> dict[str, Any]:
        if service == "serper":
            return {
                "organic": [
                    {
                        "title": f"{self.brand} — sentetik örnek",
                        "snippet": f"{self.brand} için test tarihi ve kapsamı bu örnek metinde belirtilmiştir.",
                        "link": "https://example.org/synthetic-brand-demo",
                        "position": 1,
                    }
                ]
            }
        if payload["messages"][0]["content"].startswith("Türkçe marka analizi yardımcısısın"):
            context = json.loads(payload["messages"][1]["content"])
            source = context["sources"][0]
            return {
                "text": json.dumps(
                    {
                        "actions": [
                            {
                                "source_id": source["source_id"],
                                "quote": source["snippet"],
                                "suggestion": "Gerçek bir test raporu varsa tarih ve kapsamını kaynak bağlantısıyla doğrulayın; bu sentetik örnek bir performans iddiası değildir.",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                "usage": {},
            }
        return {
            "text": f"SENTETİK YANIT: {self.brand} bu örnekte anılıyor. Bu gerçek model çıktısı değildir.",
            "usage": {},
        }
