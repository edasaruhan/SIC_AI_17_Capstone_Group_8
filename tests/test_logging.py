from __future__ import annotations

from pathlib import Path
from typing import Any

from bias_eval.logging import configure_logging, logger, sanitize_for_log


def test_log_file_contains_context_but_redacts_secrets(tmp_path: Path, monkeypatch: Any) -> None:
    secret = "fixture-super-secret-api-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    path = configure_logging(log_file=tmp_path / "pipeline.log", level="ERROR")

    logger.info("safe_context record_id={} credential={}", "record-123", secret)
    logger.complete()

    content = path.read_text(encoding="utf-8")
    assert "record-123" in content
    assert secret not in content
    assert "[REDACTED]" in content


def test_sanitize_for_log_redacts_bearer_and_bounds_message(monkeypatch: Any) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "fixture-serper-secret")
    value = sanitize_for_log("Bearer explicit-token fixture-serper-secret " + "x" * 2000)

    assert "explicit-token" not in value
    assert "fixture-serper-secret" not in value
    assert len(value) == 1000
