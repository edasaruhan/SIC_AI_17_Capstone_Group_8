"""Central, secret-safe Loguru configuration for CLI and library runs."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_LOG_FILE = Path("logs/bias-eval.log")
_SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")


def _secret_values() -> tuple[str, ...]:
    values = []
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in _SECRET_NAME_MARKERS) and len(value) >= 6:
            values.append(value)
    return tuple(sorted(set(values), key=len, reverse=True))


def sanitize_for_log(value: Any) -> str:
    """Return a bounded message with credentials removed."""

    message = str(value).replace("\r", " ").replace("\n", " ")
    message = _BEARER_PATTERN.sub("Bearer [REDACTED]", message)
    for secret in _secret_values():
        message = message.replace(secret, "[REDACTED]")
    return message[:1000]


def _redact_record(record: Any) -> None:
    record["message"] = sanitize_for_log(record["message"])


def configure_logging(
    *,
    log_file: str | Path | None = None,
    level: str | None = None,
) -> Path:
    """Configure readable stderr output and a detailed rotating debug log."""

    selected_file = Path(log_file or os.environ.get("BIAS_EVAL_LOG_FILE", DEFAULT_LOG_FILE))
    selected_level = (level or os.environ.get("BIAS_EVAL_LOG_LEVEL", "INFO")).upper()
    selected_file.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(patcher=_redact_record)
    logger.add(
        sys.stderr,
        level=selected_level,
        colorize=sys.stderr.isatty(),
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        selected_file,
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=False,
        backtrace=True,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{process.id}:{thread.id} | {name}:{function}:{line} | {message}"
        ),
    )
    logger.info("logging_ready file={} console_level={}", selected_file, selected_level)
    return selected_file


__all__ = ["configure_logging", "logger", "sanitize_for_log"]
