from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bias_eval.config import load_suite
from bias_eval.exporter import EXPORT_COLUMNS, build_export_rows, export_dataset
from bias_eval.features import deterministic_features
from bias_eval.judge import CerebrasJudgeClient, judged_path, judgment_is_current, validate_judgment
from bias_eval.judge_schema import (
    JUDGE_PROMPT_VERSION,
    content_hash,
    prompt_hash,
    response_schema,
)
from bias_eval.normalization import alias_candidates, normalized_brand
from bias_eval.providers import ProviderError
from bias_eval.records import plan_cells
from bias_eval.report import build_report
from bias_eval.storage import atomic_write_json, result_path
from bias_eval.validation import validate_dataset


def schema_fixture(schema: dict[str, Any]) -> Any:
    declared = schema.get("type")
    if isinstance(declared, list):
        declared = next(item for item in declared if item != "null")
    if declared == "object":
        return {
            key: schema_fixture(value)
            for key, value in schema.get("properties", {}).items()
            if key in schema.get("required", [])
        }
    if declared == "array":
        return []
    if declared == "string":
        return schema.get("enum", [""])[0]
    if declared == "boolean":
        return False
    if declared == "integer":
        return schema.get("minimum", 0)
    raise AssertionError(f"unsupported test schema: {schema}")


def valid_judgment(category: str, condition: str) -> dict[str, Any]:
    value = schema_fixture(response_schema(category))
    value["search_aware"] = (
        None
        if condition == "search_off"
        else schema_fixture(response_schema(category)["properties"]["search_aware"])
    )
    return value


def raw_fixture(cell: Any) -> dict[str, Any]:
    return {
        "record_id": cell.record_id,
        "experiment_id": cell.experiment.experiment_id,
        "model_id": cell.model.model_id,
        "provider": cell.model.provider,
        "condition": cell.condition,
        "category": cell.experiment.category,
        "language": "tr",
        "query_id": cell.query.id,
        "query_text": cell.query.text,
        "run_index": cell.run_index,
        "temperature": 0.7,
        "final_response": "NordVPN iyi bir seçenektir.",
        "tool_calls": [],
        "search_results": [],
        "metadata": {"started_at": "2026-01-01T00:00:00Z", "completed_at": "2026-01-01T00:00:01Z"},
        "status": "completed",
    }


@pytest.mark.parametrize("category", ["vpn", "cosmetics"])
@pytest.mark.parametrize("condition", ["search_off", "search_on"])
def test_strict_judge_schema_covers_both_domains_and_conditions(
    category: str, condition: str
) -> None:
    value = valid_judgment(category, condition)
    assert validate_judgment(value, category, condition) is value
    value["core"]["unexpected"] = True
    with pytest.raises(ValueError, match="extra fields"):
        validate_judgment(value, category, condition)


def test_cerebras_request_uses_strict_schema_reasoning_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite()
    raw = raw_fixture(plan_cells(suite)[0])
    judgment = valid_judgment("vpn", "search_off")
    monkeypatch.setenv("CEREBRAS_API_KEY", "fixture-cerebras")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-oss-120b"
        assert payload["temperature"] == 0
        assert payload["reasoning_effort"] == "low"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert content_hash(raw) in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(judgment)}}]},
        )

    async def exercise() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result, _ = await CerebrasJudgeClient(suite.judge, http).judge(raw)
            return result

    assert asyncio.run(exercise())["search_aware"] is None


def test_cerebras_paces_requests_and_repairs_null_search_without_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite()
    raw = raw_fixture(next(cell for cell in plan_cells(suite) if cell.condition == "search_on"))
    judgment = valid_judgment("vpn", "search_on")
    judgment["search_aware"] = None
    sleeps: list[float] = []
    monkeypatch.setenv("CEREBRAS_API_KEY", "fixture-cerebras")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(judgment)}}]},
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def exercise() -> tuple[dict[str, Any], dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CerebrasJudgeClient(suite.judge, http, sleep=fake_sleep)
            first, _ = await client.judge(raw)
            second, _ = await client.judge(raw)
            return first, second

    first, second = asyncio.run(exercise())
    expected = {
        "explicitly_references_search_results": False,
        "cites_specific_sources": False,
        "source_names_cited": [],
        "uses_search_to_justify_top_pick": False,
    }
    assert first["search_aware"] == expected
    assert second["search_aware"] == expected
    assert len(sleeps) == 1
    assert 19.0 <= sleeps[0] <= 20.0


def test_cerebras_invalid_response_reports_the_schema_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite()
    raw = raw_fixture(plan_cells(suite)[0])
    judgment = valid_judgment("vpn", "search_off")
    judgment["core"]["unexpected"] = True
    monkeypatch.setenv("CEREBRAS_API_KEY", "fixture-cerebras")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(judgment)}}]},
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CerebrasJudgeClient(suite.judge, http)
            with pytest.raises(ProviderError, match="judgment.core has extra fields"):
                await client.judge(raw)

    asyncio.run(exercise())


def test_prompt_or_content_change_marks_judgment_stale(tmp_path: Path) -> None:
    suite = replace(load_suite(), judged_dir=tmp_path)
    raw = raw_fixture(plan_cells(suite)[0])
    path = judged_path(suite, raw)
    atomic_write_json(
        path,
        {
            "status": "completed",
            "content_hash": content_hash(raw),
            "judge_prompt_hash": prompt_hash("vpn"),
        },
    )
    assert judgment_is_current(path, raw)
    raw["final_response"] = "Changed"
    assert not judgment_is_current(path, raw)


def test_deterministic_features_use_search_queries_in_turkish_and_english() -> None:
    raw = {
        "final_response": "# Başlık\n1. Seçim\n- Neden",
        "query_text": "kullanıcı sorgusu",
        "tool_calls": [
            {"round": 0, "arguments": {"query": "2026 güncel en iyi NordVPN karşılaştırması"}},
            {"round": 1, "arguments": {"query": "latest VPN review"}},
        ],
        "search_results": [{"results": {"organic": [{}, {}]}}],
    }
    features = deterministic_features(raw)
    assert features["query_contains_year"]
    assert features["query_contains_comparison_terms"]
    assert features["query_contains_brand_names"]
    assert features["query_contains_currentness_terms"]
    assert features["num_search_rounds"] == 2
    assert features["num_search_results_returned"] == 2


def test_reference_compatible_export_has_19_columns_plus_language(tmp_path: Path) -> None:
    config = replace(
        load_suite(),
        raw_dir=tmp_path / "raw",
        judged_dir=tmp_path / "judged",
        processed_dir=tmp_path / "processed",
    )
    cell = next(cell for cell in plan_cells(config) if cell.condition == "search_off")
    raw = raw_fixture(cell)
    atomic_write_json(result_path(config.raw_dir, cell), raw)
    judgment = valid_judgment(cell.experiment.category, cell.condition)
    judgment["core"]["all_brands_mentioned"] = ["NordVPN"]
    atomic_write_json(
        judged_path(config, raw),
        {
            "record_id": raw["record_id"],
            "category": raw["category"],
            "content_hash": content_hash(raw),
            "judge_model": config.judge.model_id,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_prompt_hash": prompt_hash(raw["category"]),
            "judgment": judgment,
            "deterministic": deterministic_features(raw),
            "status": "completed",
        },
    )

    rows, _ = build_export_rows(config)
    manifest = export_dataset(config, require_complete=False)
    table = pq.read_table(config.processed_dir / "all" / "train.parquet")

    assert len(rows) == 1
    assert len(EXPORT_COLUMNS) == 20
    assert table.column_names == EXPORT_COLUMNS
    assert table.num_rows == 1
    assert json.loads(table.to_pylist()[0]["core"])["all_brands_mentioned"] == ["NordVPN"]
    assert table.to_pylist()[0]["search_aware"] is None
    assert len(manifest["missing_generation_record_ids"]) == 299


def test_validation_fails_closed_on_incomplete_data(tmp_path: Path) -> None:
    config = replace(
        load_suite(),
        raw_dir=tmp_path / "raw",
        judged_dir=tmp_path / "judged",
        processed_dir=tmp_path / "processed",
    )
    result = validate_dataset(config)
    assert not result["ok"]
    assert result["completed_generation_rows"] == 0
    assert any("expected 300" in issue for issue in result["issues"])


def test_brand_normalization_is_conservative_and_reports_near_matches() -> None:
    assert normalized_brand("  PROTON VPN™ ", category="vpn") == "protonvpn"
    records = [
        {"category": "cosmetics", "judgment": {"core": {"all_brands_mentioned": ["cerave"]}}},
        {"category": "cosmetics", "judgment": {"core": {"all_brands_mentioned": ["ceravé"]}}},
    ]
    candidates = alias_candidates(records)
    assert candidates
    assert candidates[0]["action"] == "manual_review"


def test_report_accepts_flattened_reference_data(tmp_path: Path) -> None:
    tr_path = tmp_path / "tr.parquet"
    en_path = tmp_path / "reference.parquet"
    output = tmp_path / "report.md"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "category": "vpn",
                    "condition": "search_off",
                    "query_id": "vpn_tr_01",
                    "core": json.dumps(
                        {"has_single_winner": True, "top_recommendation": "nordvpn"}
                    ),
                    "deterministic": json.dumps({"response_length_words": 20}),
                }
            ]
        ),
        tr_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "category": "vpn",
                    "condition": "search_off",
                    "query_id": "vpn_01",
                    "core__has_single_winner": False,
                    "core__top_recommendation": "Mullvad",
                    "deterministic__response_length_words": 10,
                }
            ]
        ),
        en_path,
    )

    summary = build_report(tr_path, en_path, output)

    assert summary["turkish_vpn"]["rows"] == 1
    assert summary["english_reference_vpn"]["mean_response_words"] == 10
    assert "yalnız betimseldir" in output.read_text(encoding="utf-8")
