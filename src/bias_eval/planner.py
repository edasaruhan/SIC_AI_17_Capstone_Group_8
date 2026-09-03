"""Pure, offline expansion of the frozen experiment matrix."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_suite
from .records import CellSpec, plan_cells


def summarize(cells: list[CellSpec]) -> dict[str, Any]:
    return {
        "total": len(cells),
        "by_experiment": dict(sorted(Counter(c.experiment.experiment_id for c in cells).items())),
        "by_domain": dict(sorted(Counter(c.experiment.category for c in cells).items())),
        "by_model": dict(sorted(Counter(c.model.key for c in cells).items())),
        "by_condition": dict(sorted(Counter(c.condition for c in cells).items())),
        "by_query": dict(sorted(Counter(c.query.id for c in cells).items())),
        "unique_record_ids": len({c.record_id for c in cells}),
    }


def render_plan(config_path: str | Path) -> str:
    config = load_suite(config_path)
    return json.dumps(
        {"suite_id": config.suite_id, **summarize(plan_cells(config))},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
