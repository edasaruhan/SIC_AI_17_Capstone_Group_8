from __future__ import annotations

from collections import Counter

from bias_eval.config import load_suite
from bias_eval.planner import summarize
from bias_eval.records import pilot_cells, plan_cells


def test_frozen_plan_has_exactly_400_unique_cells() -> None:
    config = load_suite()
    cells = plan_cells(config)

    assert len(cells) == 400
    assert len({cell.record_id for cell in cells}) == 400
    assert Counter(cell.experiment.category for cell in cells) == {"vpn": 200, "cosmetics": 200}
    assert Counter(cell.condition for cell in cells) == {"search_off": 200, "search_on": 200}
    assert set(Counter(cell.model.key for cell in cells).values()) == {100}
    assert set(Counter(cell.query.id for cell in cells).values()) == {40}
    assert summarize(cells)["total"] == 400


def test_pilot_is_one_real_cell_per_domain_model_condition() -> None:
    cells = pilot_cells(load_suite())

    assert len(cells) == 16
    assert len({cell.record_id for cell in cells}) == 16
    assert {(cell.query.id, cell.run_index) for cell in cells} == {
        ("vpn_tr_01", 0),
        ("cosmetics_tr_01", 0),
    }
