"""Serper Google Search client with locale-aware persistent caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from .config import SearchConfig
from .logging import logger
from .providers import ProviderError, _retry_after, _safe_error_text, api_key
from .storage import atomic_write_json, read_json


@dataclass(frozen=True)
class SearchHit:
    query: str
    source: str
    results: dict[str, Any]
    formatted: str


class SerperSearch:
    def __init__(
        self,
        config: SearchConfig,
        cache_dir: Path,
        http: httpx.AsyncClient,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.http = http
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    def cache_key(self, query: str) -> str:
        value = json.dumps(
            {
                "q": query.strip().casefold(),
                "num": self.config.num_results,
                "gl": self.config.country,
                "hl": self.config.language,
                "location": self.config.location,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    async def search(self, query: str) -> SearchHit:
        query = query.strip()
        if not query:
            raise ProviderError(
                "Model requested web_search with an empty query", component="serper"
            )
        path = self.cache_dir / f"{self.cache_key(query)}.json"
        query_key = path.stem
        cached = read_json(path)
        if cached and isinstance(cached.get("results"), dict):
            results = cached["results"]
            logger.info(
                "serper_cache_hit query_key={} organic_results={}",
                query_key,
                len(results.get("organic") or []),
            )
            return SearchHit(query, "cache", results, self.format_results(results))

        started = monotonic()
        logger.info(
            "serper_request query_key={} query_length={} locale={}/{} num={}",
            query_key,
            len(query),
            self.config.country,
            self.config.language,
            self.config.num_results,
        )
        async with self._semaphore:
            try:
                response = await self.http.post(
                    self.config.api_base,
                    headers={"X-API-KEY": api_key(self.config.api_key_env)},
                    json={
                        "q": query,
                        "num": self.config.num_results,
                        "gl": self.config.country,
                        "hl": self.config.language,
                        "location": self.config.location,
                    },
                )
            except httpx.TransportError as error:
                logger.error(
                    "serper_transport_error query_key={} duration_seconds={:.3f} error_type={}",
                    query_key,
                    monotonic() - started,
                    type(error).__name__,
                )
                raise ProviderError(
                    "Serper transport error", retryable=True, component="serper"
                ) from error
        if response.status_code >= 400:
            logger.warning(
                "serper_http_error query_key={} status_code={} duration_seconds={:.3f} "
                "retry_after={}",
                query_key,
                response.status_code,
                monotonic() - started,
                _retry_after(response),
            )
            raise ProviderError(
                _safe_error_text(response),
                status_code=response.status_code,
                retry_after=_retry_after(response),
                retryable=response.status_code in {408, 429, 500, 502, 503, 504},
                quota_exhausted=response.status_code in {402, 429},
                component="serper",
            )
        try:
            results = response.json()
        except ValueError as error:
            raise ProviderError(
                "Serper returned invalid JSON", retryable=True, component="serper"
            ) from error
        if not isinstance(results, dict):
            raise ProviderError(
                "Serper returned a non-object response", retryable=True, component="serper"
            )
        atomic_write_json(
            path,
            {
                "query": query,
                "fetched_at": datetime.now(UTC).isoformat(),
                "locale": {
                    "gl": self.config.country,
                    "hl": self.config.language,
                    "location": self.config.location,
                    "num": self.config.num_results,
                },
                "results": results,
            },
        )
        logger.info(
            "serper_response query_key={} status_code={} duration_seconds={:.3f} "
            "organic_results={} cache_file={}",
            query_key,
            response.status_code,
            monotonic() - started,
            len(results.get("organic") or []),
            path,
        )
        return SearchHit(query, "live", results, self.format_results(results))

    @staticmethod
    def format_results(results: dict[str, Any]) -> str:
        parts = []
        for index, item in enumerate(results.get("organic") or [], start=1):
            if not isinstance(item, dict):
                continue
            parts.append(
                f"{index}. {item.get('title', '')}\n"
                f"   {item.get('snippet', '')}\n"
                f"   URL: {item.get('link', '')}"
            )
        return (
            "Arama sonucu bulunamadı." if not parts else "Arama sonuçları:\n\n" + "\n\n".join(parts)
        )
