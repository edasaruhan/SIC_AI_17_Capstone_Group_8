from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bias_eval.config import load_suite
from bias_eval.provenance import (
    collection_fingerprint,
    collection_lock_issue,
    ensure_collection_lock,
)


def test_collection_lock_rejects_changed_generation_settings(tmp_path: Path) -> None:
    suite = load_suite()
    config = replace(suite, raw_dir=tmp_path / "raw")

    path = ensure_collection_lock(config)

    assert path.is_file()
    assert collection_lock_issue(config) is None
    changed = replace(config, temperature=0.2)
    assert collection_fingerprint(changed) != collection_fingerprint(config)
    with pytest.raises(ValueError, match="does not match"):
        ensure_collection_lock(changed)
