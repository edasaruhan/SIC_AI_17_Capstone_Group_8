import pytest

from parse.protocol import (
    DUPLICATE_LABEL,
    EMPTY_VALUE,
    EXPECTED_SINGLE_VALUE,
    INCOMPLETE_RANKING,
    MARKER_MISSING,
    NO_RESPONSE,
    UNKNOWN_LABEL,
    ParseReport,
    marker_for,
    parse_records,
    parse_response,
)
from parse.turkish import fold, turkish_lower, turkish_upper

CANDIDATES = ("marka_a", "marka_b", "marka_c")


def test_turkish_case_helpers_round_trip() -> None:
    assert turkish_upper("sıralama") == "SIRALAMA"
    assert turkish_upper("ilk") == "İLK"
    assert turkish_lower("SIRALAMA") == "sıralama"
    assert turkish_lower("İLK") == "ilk"


def test_python_default_lower_is_why_folding_exists() -> None:
    # The trap the parser has to survive: str.lower does not round-trip here.
    assert "SIRALAMA".lower() != "sıralama"
    assert fold("SIRALAMA") == fold("sıralama") == fold("Sıralama") == fold("SİRALAMA")


def test_marker_for_rejects_unknown_protocol() -> None:
    assert marker_for("alpha") == "SIRALAMA"
    with pytest.raises(ValueError, match="delta"):
        marker_for("delta")


def test_parses_a_well_formed_ranking() -> None:
    text = "Kısa bir gerekçe.\nSIRALAMA: [B, A, C]"

    result = parse_response(text, protocol="alpha", candidates=CANDIDATES)

    assert result.succeeded
    assert result.labels == ("B", "A", "C")
    assert result.brands == ("marka_b", "marka_a", "marka_c")
    assert result.top_choice == "marka_b"
    assert result.on_last_line


@pytest.mark.parametrize(
    "line",
    [
        "SIRALAMA: [B, A, C]",
        "sıralama: [B, A, C]",
        "Sıralama : B, A, C",
        "SİRALAMA: [b, a, c]",
        "SIRALAMA:[B,A,C].",
        "**SIRALAMA:** [B, A, C]",
    ],
)
def test_accepts_turkish_and_formatting_variants(line: str) -> None:
    result = parse_response(line, protocol="alpha", candidates=CANDIDATES)

    assert result.succeeded, result.failure_reason
    assert result.brands == ("marka_b", "marka_a", "marka_c")


def test_uses_the_last_marker_line_and_flags_position() -> None:
    text = "SIRALAMA: [A, B, C]\nDüzeltme yapıyorum.\nSIRALAMA: [C, B, A]\nTeşekkürler."

    result = parse_response(text, protocol="alpha", candidates=CANDIDATES)

    assert result.labels == ("C", "B", "A")
    assert result.on_last_line is False


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (None, NO_RESPONSE),
        ("   ", NO_RESPONSE),
        ("Bence hepsi iyi markalar.", MARKER_MISSING),
        ("SIRALAMA:", EMPTY_VALUE),
        ("SIRALAMA: []", EMPTY_VALUE),
        ("SIRALAMA: [A, A, B]", DUPLICATE_LABEL),
        ("SIRALAMA: [A, Z, B]", UNKNOWN_LABEL),
        ("SIRALAMA: [A, B]", INCOMPLETE_RANKING),
        ("SIRALAMA: [Marka A, Marka B, Marka C]", UNKNOWN_LABEL),
    ],
)
def test_broken_responses_are_reported_not_dropped(text: str | None, reason: str) -> None:
    result = parse_response(text, protocol="alpha", candidates=CANDIDATES)

    assert result.status == "failed"
    assert result.failure_reason == reason
    assert result.brands == ()


def test_single_value_protocols_reject_lists() -> None:
    assert parse_response("ONERI: B", protocol="beta", candidates=CANDIDATES).brands == ("marka_b",)
    assert parse_response("SECIM: A", protocol="gamma", candidates=("kontrol", "v1")).brands == (
        "kontrol",
    )

    result = parse_response("ONERI: A, B", protocol="beta", candidates=CANDIDATES)

    assert result.failure_reason == EXPECTED_SINGLE_VALUE


def test_turkish_brand_names_survive_the_mapping() -> None:
    candidates = ("gülçiçek", "IŞIK Kozmetik", "şeker_bakım")

    result = parse_response("SIRALAMA: [C, A, B]", protocol="alpha", candidates=candidates)

    assert result.brands == ("şeker_bakım", "gülçiçek", "IŞIK Kozmetik")


def test_parser_does_not_crash_on_hostile_input() -> None:
    for text in ["SIRALAMA: [[[", "SIRALAMA: ,,,,", "\x00\x01", "SIRALAMA: " + "A," * 500]:
        result = parse_response(text, protocol="alpha", candidates=CANDIDATES)
        assert result.status in {"ok", "failed"}


def test_report_counts_every_response_and_splits_by_model() -> None:
    records = [
        {
            "status": "ok",
            "model_key": "claude_haiku",
            "protocol": "alpha",
            "candidates": list(CANDIDATES),
            "response_text": "SIRALAMA: [A, B, C]",
        },
        {
            "status": "ok",
            "model_key": "gpt_mini",
            "protocol": "alpha",
            "candidates": list(CANDIDATES),
            "response_text": "Sıralama yapamam.",
        },
        {
            "status": "error",
            "model_key": "gpt_mini",
            "protocol": "alpha",
            "candidates": list(CANDIDATES),
            "error_type": "RetryableProviderError",
        },
    ]

    results, report = parse_records(records)

    assert len(results) == 2, "error rows are not parsed but must not be counted as parse failures"
    assert report.total == 2
    assert report.succeeded == 1
    assert report.failures_by_reason == {MARKER_MISSING: 1}
    assert report.by_model["claude_haiku"]["succeeded"] == 1
    assert report.as_dict()["by_model"]["gpt_mini"]["success_rate"] == 0.0


def test_empty_report_has_no_division_by_zero() -> None:
    assert ParseReport().as_dict()["success_rate"] == 0.0
