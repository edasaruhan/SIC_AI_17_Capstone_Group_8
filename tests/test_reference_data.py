import json

import pandas as pd
import pytest

from reference_data import (
    DataQualityError,
    canonicalize_brand,
    decode_json_value,
    flatten_reference_frame,
    normalize_parquet_cell,
    parse_json_columns,
)


def encode_twice(value: object) -> str:
    return json.dumps(json.dumps(value))


def test_decode_json_value_accepts_single_and_double_encoding() -> None:
    assert decode_json_value('{"key": true}', column="core", record_id="one") == {"key": True}
    assert decode_json_value(
        encode_twice([{"query": "best vpn"}]),
        column="tool_calls",
        record_id="two",
    ) == [{"query": "best vpn"}]
    assert decode_json_value(None, column="search_aware", record_id="three") is None


def test_decode_json_value_reports_the_column_and_record() -> None:
    with pytest.raises(ValueError, match="core.*broken-record"):
        decode_json_value("not-json", column="core", record_id="broken-record")


def test_normalize_parquet_cell_produces_stable_string_lists() -> None:
    value = [{"b": 2, "a": 1}, "source", 3, None]

    assert normalize_parquet_cell(value) == [
        '{"a":1,"b":2}',
        "source",
        "3",
        "null",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Mullvad VPN", "Mullvad"),
        ("visual studio code", "VS Code"),
        ("ProtonVPN", "Proton VPN"),
        ("NordVPN", "NordVPN"),
        (None, None),
    ],
)
def test_canonicalize_brand(value: object, expected: object) -> None:
    assert canonicalize_brand(value) == expected


def test_parse_and_flatten_reference_fields() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["off-1", "on-1"],
            "condition": ["search_off", "search_on"],
            "core": [
                encode_twice(
                    {
                        "top_recommendation": "Mullvad VPN",
                        "brand_mentions": [{"name": "Mullvad", "position": 1}],
                        "confidence_in_extraction": "high",
                    }
                ),
                encode_twice(
                    {
                        "top_recommendation": "Visual Studio Code",
                        "brand_mentions": [],
                        "confidence_in_extraction": "medium",
                    }
                ),
            ],
            "domain": [encode_twice({"flag": True}), encode_twice({"flag": False})],
            "search_aware": [None, encode_twice({"cites_specific_sources": True})],
            "deterministic": [
                encode_twice({"response_length_words": 10}),
                encode_twice({"response_length_words": 20}),
            ],
            "tool_calls": [encode_twice([]), encode_twice([{"query": "best editor"}])],
            "search_results": [encode_twice([]), encode_twice([{"organic": []}])],
        }
    )

    parsed = parse_json_columns(frame)
    output = flatten_reference_frame(frame, parsed)

    assert not set(("core", "domain", "search_aware")).intersection(output.columns)
    assert output["top_recommendation_canonical"].tolist() == ["Mullvad", "VS Code"]
    assert output["core__brand_mentions"].iloc[0] == ['{"name":"Mullvad","position":1}']
    assert output["tool_calls__items"].iloc[1] == ['{"query":"best editor"}']


def test_parse_json_columns_rejects_the_wrong_top_level_type() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["bad-type"],
            "core": [encode_twice([])],
            "domain": [encode_twice({})],
            "search_aware": [None],
            "deterministic": [encode_twice({})],
            "tool_calls": [encode_twice([])],
            "search_results": [encode_twice([])],
        }
    )

    with pytest.raises(DataQualityError, match="Unexpected parsed type in core"):
        parse_json_columns(frame)
