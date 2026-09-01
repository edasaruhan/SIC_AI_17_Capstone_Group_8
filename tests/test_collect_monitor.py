import json
from pathlib import Path

from collect.monitor import build_report, render_daily_line, render_table


def line(**overrides: object) -> str:
    record = {
        "call_id": "id",
        "model_key": "claude_haiku",
        "model_version": "claude-haiku-4-5-20251001",
        "protocol": "alpha",
        "candidates": ["a", "b", "c"],
        "status": "ok",
        "started_at": "2026-09-01T10:00:00.000+00:00",
        "cost_usd": 0.001,
        "response_text": "SIRALAMA: [A, B, C]",
    }
    record.update(overrides)
    return json.dumps(record, ensure_ascii=False)


def write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_report_separates_call_errors_from_parse_failures(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "run.jsonl",
        [
            line(),
            line(response_text="Sıralama yapamıyorum."),
            line(status="error", error_type="RetryableProviderError", cost_usd=None),
            line(model_key="gpt_mini", cost_usd=0.002),
        ],
    )

    report = build_report(log, run_id="test-run", spend_cap_usd=10.0)

    haiku = report["by_model"]["claude_haiku"]
    assert haiku["calls"] == 3
    assert haiku["failed"] == 1, "an API error is a call failure"
    assert haiku["unparsed"] == 1, "a readable call with no marker is a parse failure"
    assert haiku["parse_rate"] == 0.5
    assert haiku["errors"] == {"RetryableProviderError": 1, "parse:marker_missing": 1}
    assert report["totals"]["calls"] == 4
    assert report["totals"]["spend_usd"] == 0.004
    assert report["spend_cap_exceeded"] is False


def test_model_versions_are_tracked_so_drift_is_visible(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "run.jsonl",
        [line(), line(model_version="claude-haiku-4-5-20260401")],
    )

    report = build_report(log, run_id="test-run")

    assert report["by_model"]["claude_haiku"]["model_versions"] == [
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5-20260401",
    ]


def test_spend_cap_breach_is_flagged_for_the_day(tmp_path: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [line(cost_usd=6.0), line(cost_usd=6.0)])

    report = build_report(log, run_id="test-run", spend_cap_usd=10.0)

    assert report["latest_day_spend_usd"] == 12.0
    assert report["spend_cap_exceeded"] is True
    assert "TAVAN AŞILDI" in render_daily_line(report)


def test_malformed_lines_are_counted_in_the_daily_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(line() + "\n{broken\n", encoding="utf-8")

    report = build_report(path, run_id="test-run")

    assert report["malformed_lines"] == 1
    assert "1 bozuk satır" in render_daily_line(report)


def test_table_has_one_row_per_model_plus_a_total(tmp_path: Path) -> None:
    log = write_log(tmp_path / "run.jsonl", [line(), line(model_key="gpt_mini")])

    table = render_table(build_report(log, run_id="test-run"))

    assert table.count("\n") == 4, "header, separator, two models, total"
    assert "claude_haiku" in table and "gpt_mini" in table and "**toplam**" in table


def test_a_missing_log_reports_zeroes_instead_of_crashing(tmp_path: Path) -> None:
    report = build_report(tmp_path / "absent.jsonl", run_id="test-run", spend_cap_usd=10.0)

    assert report["totals"]["calls"] == 0
    assert report["totals"]["parse_rate"] == 0.0
    assert report["spend_cap_exceeded"] is False
