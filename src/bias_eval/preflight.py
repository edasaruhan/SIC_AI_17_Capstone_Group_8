"""Credential presence and non-generating provider connectivity checks."""

from __future__ import annotations

import asyncio
import os
from time import monotonic
from typing import Any

import httpx

from .config import SuiteConfig
from .logging import logger


def credential_status(config: SuiteConfig) -> dict[str, bool]:
    names = [model.api_key_env for model in config.models]
    names.extend((config.search.api_key_env, config.judge.api_key_env))
    return {name: bool(os.environ.get(name, "").strip()) for name in names}


async def check_connections(
    config: SuiteConfig,
    *,
    model_keys: set[str] | None = None,
    include_judge: bool = True,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """GET model catalogs; this does not request a generation or a Serper search."""

    credentials = credential_status(config)
    available_keys = {model.key for model in config.models}
    selected_keys = model_keys or available_keys
    unknown_keys = selected_keys - available_keys
    if unknown_keys:
        raise ValueError(f"Unknown model keys: {sorted(unknown_keys)}")
    targets = [
        (model.key, model.api_base, model.api_key_env, model.model_id)
        for model in config.models
        if model.key in selected_keys
    ]
    if include_judge:
        targets.append(
            ("judge", config.judge.api_base, config.judge.api_key_env, config.judge.model_id)
        )
    results: dict[str, Any] = {}
    owns_http = http is None
    session = http or httpx.AsyncClient(timeout=20.0)
    logger.info(
        "preflight_started models={} include_judge={}", sorted(selected_keys), include_judge
    )

    async def check(name: str, base: str, env_name: str, model_id: str) -> None:
        if not credentials[env_name]:
            results[name] = {"ok": False, "error": f"missing {env_name}"}
            logger.error("preflight_missing_credential target={} env={}", name, env_name)
            return
        started = monotonic()
        try:
            response = await session.get(
                f"{base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {os.environ[env_name]}"},
            )
            model_available: bool | None = None
            if response.status_code < 400:
                try:
                    body = response.json()
                    entries = body.get("data") if isinstance(body, dict) else None
                    if isinstance(entries, list):
                        model_ids = {
                            str(item.get("id")) for item in entries if isinstance(item, dict)
                        }
                        accepted_ids = {model_id, f"models/{model_id}"}
                        model_available = bool(model_ids & accepted_ids)
                except ValueError:
                    pass
            results[name] = {
                "ok": response.status_code < 400 and model_available is not False,
                "status_code": response.status_code,
                "model_id": model_id,
                "model_available": model_available,
            }
            logger.info(
                "preflight_result target={} model={} status_code={} model_available={} "
                "duration_seconds={:.3f}",
                name,
                model_id,
                response.status_code,
                model_available,
                monotonic() - started,
            )
        except httpx.HTTPError as error:
            results[name] = {"ok": False, "error": type(error).__name__}
            logger.error(
                "preflight_transport_error target={} model={} duration_seconds={:.3f} "
                "error_type={}",
                name,
                model_id,
                monotonic() - started,
                type(error).__name__,
            )

    try:
        await asyncio.gather(*(check(*target) for target in targets))
    finally:
        if owns_http:
            await session.aclose()
    results["serper"] = {
        "ok": credentials[config.search.api_key_env],
        "note": "key presence only; use search-test for one live query",
    }
    logger.info(
        "preflight_completed targets={} all_ok={}",
        len(results),
        all(v["ok"] for v in results.values()),
    )
    return results
