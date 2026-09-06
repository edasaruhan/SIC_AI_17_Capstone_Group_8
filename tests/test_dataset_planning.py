from __future__ import annotations

from collections import Counter

from bias_eval.config import load_suite
from bias_eval.planner import summarize
from bias_eval.records import pilot_cells, plan_cells


def test_frozen_plan_has_exactly_300_unique_cells() -> None:
    config = load_suite()
    cells = plan_cells(config)

    assert len(cells) == 300
    assert len({cell.record_id for cell in cells}) == 300
    assert Counter(cell.experiment.category for cell in cells) == {"vpn": 150, "cosmetics": 150}
    assert Counter(cell.condition for cell in cells) == {"search_off": 150, "search_on": 150}
    assert set(Counter(cell.model.key for cell in cells).values()) == {100}
    assert set(Counter(cell.query.id for cell in cells).values()) == {30}
    assert summarize(cells)["total"] == 300
    gemini = next(model for model in config.models if model.key == "gemini_3_5_flash_lite")
    assert gemini.model_id == "gemini-3.5-flash-lite"
    assert config.judge.max_concurrent == 1
    assert config.judge.min_request_interval_seconds == 20.0


def test_pilot_is_one_real_cell_per_domain_model_condition() -> None:
    cells = pilot_cells(load_suite())

    assert len(cells) == 12
    assert len({cell.record_id for cell in cells}) == 12
    assert {(cell.query.id, cell.run_index) for cell in cells} == {
        ("vpn_tr_01", 0),
        ("cosmetics_tr_01", 0),
    }
