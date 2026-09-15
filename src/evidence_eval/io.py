"""Atomic, finite JSON and immutable experiment identity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(
        path,
        json.dumps(clean(value), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_url(value: str) -> bool:
    from urllib.parse import urlsplit

    try:
        url = urlsplit(value)
        return url.scheme in {"http", "https"} and bool(url.hostname) and not url.username
    except ValueError:
        return False
