import json

import pandas as pd

from visibility import intervention_report as report

RESULTS = [{"title": "t", "snippet": "s", "link": "https://site.com/x"}]
PLAN = {
    "model": "gemini-3.5-flash-lite",
    "temperature": 0.7,
    "reps": 1,
    "contexts": [
        {"sector": "vpn", "query_id": "vpn_01", "query_text": "q", "results": RESULTS},
    ],
    "targets": {"vpn": ["Windscribe"]},
    "peers": {"vpn": ["Proton VPN", "Mullvad"]},
}


def test_report_shows_arms_peers_pilot_note_and_token_totals(tmp_path) -> None:
    steps = tmp_path / "steps"
    steps.mkdir()
    done = {
        "status": "completed",
        "result": {"usage": {"prompt_tokens": 100, "completion_tokens": 40}},
    }
    (steps / "a.json").write_text(json.dumps(done))
    (steps / "b.json").write_text(json.dumps({"status": "error"}))
    outcomes = pd.DataFrame(
        [
            {"arm": "control", "mentioned": 0, "first": 0},
            {"arm": "neutral_p1", "mentioned": 1, "first": 0},
        ]
    )
    effects = pd.DataFrame(
        [
            {
                "scope": "ALL",
                "contrast": "Sıra: 5 → 1",
                "metric": "mentioned",
                "effect": 0.25,
                "effect_lo": 0.1,
                "effect_hi": 0.4,
                "cells": 2,
            }
        ]
    )
    text = report.render(PLAN, outcomes, effects, tmp_path)
    assert "Tamamlanan çağrı: 1/5" in text  # 1 control + 1 target x 4 arms
    assert "100 girdi + 40 çıktı" in text
    assert "Proton VPN, Mullvad" in text and "Pilot:" in text
    assert "+1 karşılaştırma sayfası, 1. sıra | %100.0 | %0.0 | 1 |" in text
    assert "| Tümü | Sıra: 5 → 1 | Anılma | +25.0 | [+10.0, +40.0] | 2 |" in text
