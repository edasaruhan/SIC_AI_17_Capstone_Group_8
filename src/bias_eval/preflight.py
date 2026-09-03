"""Credential presence and non-generating provider connectivity checks."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from .config import SuiteConfig


def credential_status(config: SuiteConfig) -> dict[str, bool]:
    names = [model.api_key_env for model in config.models]
    names.extend((config.search.api_key_env, config.judge.api_key_env))
    return {name: bool(os.environ.get(name, "").strip()) for name in names}


async def check_connections(config: SuiteConfig) -> dict[str, Any]:
    """GET model catalogs; this does not request a generation or a Serper search."""

    credentials = credential_status(config)
    targets = [(model.key, model.api_base, model.api_key_env) for model in config.models] + [
        ("judge", config.judge.api_base, config.judge.api_key_env)
    ]
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=20.0) as http:

        async def check(name: str, base: str, env_name: str) -> None:
            if not credentials[env_name]:
                results[name] = {"ok": False, "error": f"missing {env_name}"}
                return
            try:
                response = await http.get(
                    f"{base.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {os.environ[env_name]}"},
                )
                results[name] = {
                    "ok": response.status_code < 400,
                    "status_code": response.status_code,
                }
            except httpx.HTTPError as error:
                results[name] = {"ok": False, "error": type(error).__name__}

        await asyncio.gather(*(check(*target) for target in targets))
    results["serper"] = {
        "ok": credentials[config.search.api_key_env],
        "note": "key presence only; use search-test for one live query",
    }
    return results
