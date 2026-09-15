"""The CLI demo is verified with synthetic data and HTTP fixtures only."""

import asyncio
import json
import socket
from collections import Counter

import httpx
import pytest

from brand_demo.__main__ import load_keys, main
from brand_demo.actions import build_action_plan, rebuild
from brand_demo.clients import FixtureClient, LiveClient
from brand_demo.core import advice, brand_aliases, final_text, mentions, plan, render, sources_from
from brand_demo.local import context
from brand_demo.workflow import Receipts, run
from evidence_eval.io import read_json, sha256, write_json


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in demo tests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def config(offline=True):
    return plan("Proton VPN", "vpn", ["Mullvad"], "fixture", offline)


def test_planning_is_neutral_generic_bounded_and_versioned():
    c = config()
    assert len(c["queries"]) == 3
    assert all("Proton" not in q for q in c["queries"])
    assert c["normal_uncached_calls"] == {"minimax": 7, "serper": 4}
    assert c["maximum_requested_output_tokens"] == 7 * 4096
    assert c["id"] == config()["id"]
    assert config(False)["id"] != c["id"]
    assert plan("Kahveci", "kahve", [], "test", True)["domain"] is None
    assert plan("CeraVe", "kozmetik", [], "test", True)["domain"] == "cosmetics"
    with pytest.raises(ValueError, match="genel"):
        plan("Proton", "Proton hizmetleri", [], "test", False)
    with pytest.raises(ValueError, match="karakter"):
        plan("bad\x1b[31m", "vpn", [], "test", True)
    with pytest.raises(ValueError, match="en fazla"):
        plan("A", "vpn", ["B"] * 6, "test", True)


def test_alias_boundaries_turkish_i_and_thinking_exclusion():
    assert mentions("ProtonVPN önerilir", brand_aliases("Proton VPN", "vpn"))
    assert mentions("IVPN", ["ıvpn"])
    assert not mentions("bAIRport", ["Air"])
    text = final_text("<think>Proton VPN olabilir</think>Yeterli bilgim yok.")
    assert not mentions(text, ["Proton VPN"])
    with pytest.raises(ValueError):
        final_text("<think>yarım çıktı")
    with pytest.raises(ValueError):
        final_text("<think>sadece düşünme</think>")


def test_sources_and_citation_validation_fail_closed():
    sources = sources_from(
        {
            "organic": [
                {
                    "title": "Denetim raporu",
                    "snippet": "Tarih ve kapsam belirtilmiştir.",
                    "link": "https://example.org/report",
                },
                {"link": "javascript:alert(1)"},
            ]
        },
        "S",
        "audit",
    )
    assert len(sources) == 1
    action = {
        "source_id": "S-1",
        "quote": "Tarih ve kapsam belirtilmiştir.",
        "suggestion": "Varsa orijinal raporu ve kapsamını doğrulayın.",
    }
    assert advice(json.dumps({"actions": [action]}), sources) == [action]
    for changed in (
        {**action, "source_id": "fake"},
        {**action, "quote": "Uydurma bir alıntı metni"},
        {**action, "extra": "not allowed"},
    ):
        with pytest.raises(ValueError):
            advice(json.dumps({"actions": [changed]}), sources)


def test_full_fixture_and_cached_rerun_have_exact_call_bound(tmp_path):
    calls = []
    fixture = FixtureClient("Proton VPN")

    async def client(service, payload):
        calls.append((service, payload))
        return await fixture(service, payload)

    c = config()
    report = asyncio.run(run(c, tmp_path, client, use_local=False))
    assert Counter(service for service, _ in calls) == {"minimax": 7, "serper": 4}
    assert len(report["answers"]) == 6
    assert {r["responses"] for r in report["observations"]} == {3}
    assert "SENTETİK" in render(report)
    assert all(s["purpose"] == "neutral_query" for a in report["answers"] for s in a["sources"])
    assert any(s["purpose"] == "brand_audit_only_not_visibility" for s in report["sources"])
    for service, payload in calls:
        if service == "minimax":
            assert "tools" not in payload
            assert payload["reasoning_split"] is True
    calls.clear()
    replay = asyncio.run(run(c, tmp_path, client, use_local=False))
    assert calls == []
    assert replay["collection_window_utc"] == report["collection_window_utc"]
    assert read_json(tmp_path / c["id"] / "state.json")["status"] == "completed"


def test_empty_search_skips_advice_does_not_invent_sources(tmp_path):
    async def client(service, payload):
        return {"organic": []} if service == "serper" else {"text": "Bilmiyorum.", "usage": {}}

    report = asyncio.run(run(config(), tmp_path, client, use_local=False))
    assert report["actions"] == report["sources"] == []
    assert report["cumulative_attempts"] == {"minimax": 6, "serper": 4}


def test_failed_or_unknown_step_requires_explicit_retry(tmp_path):
    count = 0

    async def client(service, payload):
        nonlocal count
        count += 1
        raise ValueError("fixture failure")

    receipts = Receipts(tmp_path, client, False)
    with pytest.raises(ValueError, match="fixture"):
        asyncio.run(receipts.call("step", "minimax", {}))
    with pytest.raises(ValueError, match="retry-failed"):
        asyncio.run(receipts.call("step", "minimax", {}))
    assert count == 1
    receipts.retry_failed = True
    for _ in range(2):
        with pytest.raises(ValueError, match="fixture"):
            asyncio.run(receipts.call("step", "minimax", {}))
    with pytest.raises(ValueError, match="3 deneme"):
        asyncio.run(receipts.call("step", "minimax", {}))
    assert count == 3


def test_bad_advice_is_retryable_without_repeating_collection(tmp_path):
    fixture = FixtureClient("Proton VPN")
    calls = []
    bad = True

    async def client(service, payload):
        calls.append(service)
        if (
            service == "minimax"
            and payload["messages"][0]["content"].startswith("Türkçe marka analizi yardımcısısın")
            and bad
        ):
            return {"text": '{"actions":[{"source_id":"fake"}]}'}
        return await fixture(service, payload)

    with pytest.raises(ValueError):
        asyncio.run(run(config(), tmp_path, client, use_local=False))
    assert read_json(tmp_path / config()["id"] / "steps" / "advice.json")["status"] == "error"
    calls.clear()
    bad = False
    result = asyncio.run(run(config(), tmp_path, client, retry_failed=True, use_local=False))
    assert result["status"] == "completed" and calls == ["minimax"]


def test_corrupt_receipt_refuses_new_paid_call(tmp_path):
    async def client(service, payload):
        return {"text": "fixture"}

    receipt = Receipts(tmp_path, client, False)
    asyncio.run(receipt.call("step", "minimax", {}))
    path = tmp_path / "steps" / "step.json"
    state = read_json(path)
    state["result"]["text"] = "changed"
    write_json(path, state)
    with pytest.raises(ValueError, match="bütünlüğü"):
        asyncio.run(receipt.call("step", "minimax", {}))


@pytest.mark.parametrize("status", [401, 402, 429, 500, 302])
def test_http_error_no_auto_retry_no_secret_body(monkeypatch, status):
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-secret-value")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, json={"error": "fixture-secret-value"}, headers={"Retry-After": "30"}
        )

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await LiveClient(http)("minimax", {})

    with pytest.raises(ValueError) as exc:
        asyncio.run(execute())
    assert len(calls) == 1 and "fixture-secret-value" not in str(exc.value)
    assert str(status) in str(exc.value)


def test_compatible_response_discards_reasoning_and_headers(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-key-value")

    def handler(request):
        assert str(request.url) == "https://api.minimax.io/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fixture-key-value"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "<think>Özel analiz</think>Nihai cevap",
                            "reasoning_details": [{"text": "Özel analiz"}],
                        },
                    }
                ],
                "usage": {"completion_tokens": 12},
                "base_resp": {"status_code": 0},
            },
        )

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await LiveClient(http)("minimax", {})

    result = asyncio.run(execute())
    assert result["text"] == "Nihai cevap"
    assert "Özel analiz" not in json.dumps(result)
    assert "fixture-key-value" not in json.dumps(result)


def test_literal_env_parser_never_executes_or_loads_other_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        'MINIMAX_API_KEY="literal-value"\nSERPER_API_KEY=literal-search\nUNRELATED_SECRET=ignored\n'
    )
    load_keys(path)
    import os

    assert os.environ["MINIMAX_API_KEY"] == "literal-value"
    assert "UNRELATED_SECRET" not in os.environ


def test_plan_and_offline_cli_do_not_load_env(tmp_path, monkeypatch, capsys):
    def forbidden(*args):
        raise AssertionError("No credentials in plan/offline")

    monkeypatch.setattr("brand_demo.__main__.load_keys", forbidden)
    args = ["--brand", "Proton VPN", "--sector", "vpn", "--output", str(tmp_path)]
    assert main(args) == 0
    assert "plan_only" in capsys.readouterr().out
    assert main([*args, "--offline"]) == 0
    assert "SENTETİK" in capsys.readouterr().out


def test_unconfirmed_live_cli_makes_no_call(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert (
        main(["--live", "--brand", "Proton VPN", "--sector", "vpn", "--output", str(tmp_path)]) == 0
    )
    assert list(tmp_path.iterdir()) == []


def test_unknown_sector_has_no_pilot_score(tmp_path):
    c = plan("Kahveci", "kahve", [], "fixture", False)
    assert context(c, [], tmp_path, tmp_path)["status"] == "unsupported_sector"


def zero_report():
    c = plan("warp", "vpn", [], "zero-test", False)
    return {
        "config": c,
        "sources": [
            {
                "source_id": "Q1-1",
                "title": "NordVPN güvenlik özellikleri",
                "snippet": "Fiyat, hız ve test kapsamı hakkında bilgiler.",
                "link": "https://example.org/comparison",
                "purpose": "neutral_query",
            },
            {
                "source_id": "B-1",
                "title": "Cloudflare WARP",
                "snippet": "WARP ürün bilgileri",
                "link": "https://example.org/brand",
                "purpose": "brand_audit_only_not_visibility",
            },
        ],
        "answers": [
            {"query": q, "condition": condition, "text": "NordVPN bir seçenektir.", "sources": []}
            for q in c["queries"]
            for condition in ("search_off", "search_on")
        ],
        "actions": [],
    }


def test_zero_mentions_produces_evidence_linked_deliverables_without_fabrication():
    report = zero_report()
    result = build_action_plan(report)
    assert result["facts"]["zero_mentions"] is True
    assert result["facts"]["neutral_source_mentions"] == 0
    assert result["facts"]["branded_source_mentions"] == 1
    assert "marka sorgusunda" in result["hypothesis"].lower()
    assert result["observed_competitors"][0]["brand"] == "NordVPN"
    assert result["observed_competitors"][0]["search_on"] == 3
    assert len(result["tasks"]) == 4
    lookup = {s["source_id"]: s for s in report["sources"]}
    for task in result["tasks"]:
        assert task["owner"] and task["deliverable"] and task["acceptance"] and task["steps"]
        assert task["status"] == "hypothesis_requires_review"
        for evidence in task["evidence"]:
            source = lookup[evidence["source_id"]]
            assert evidence["quote"] in source["title"] + "\n" + source["snippet"]
    assert result["measurement_plan"]["queries"] == report["config"]["queries"]


def test_empty_sources_still_produce_explicit_research_tasks():
    report = zero_report()
    report["sources"] = []
    result = build_action_plan(report)
    assert "veri yetersiz" in result["hypothesis"]
    assert result["tasks"] and result["source_themes"] == []
    assert all(task["evidence"] == [] for task in result["tasks"])
    assert all(
        task["basis"]
        in {"measured_observation", "measurement_gap", "verification_task_not_observed_defect"}
        for task in result["tasks"]
    )


def test_exposure_without_selection_has_different_hypothesis():
    report = zero_report()
    report["sources"][0]["snippet"] = "Cloudflare WARP ürün karşılaştırması"
    result = build_action_plan(report)
    assert result["facts"]["neutral_source_mentions"] == 1
    assert "yanıta taşınmamış" in result["hypothesis"]
    assert not any("Bulunabilirliği" in t["title"] for t in result["tasks"])


def test_action_plan_does_not_invent_unknown_sector_rivals():
    report = zero_report()
    report["config"] = plan("Kahveci", "kahve", [], "test", False)
    assert build_action_plan(report)["observed_competitors"] == []
    report["sources"].append(report["sources"][0])
    with pytest.raises(ValueError, match="benzersiz"):
        build_action_plan(report)


def test_offline_rebuild_keeps_old_paid_receipts_unchanged(tmp_path, monkeypatch, capsys):
    c = config()
    asyncio.run(run(c, tmp_path, FixtureClient(c["brand"]), use_local=False))
    folder = tmp_path / c["id"]
    originals = {p: sha256(p) for p in folder.rglob("*.json")}

    def forbidden(*args):
        raise AssertionError("Rebuild must not load credentials or create a live client")

    monkeypatch.setattr("brand_demo.__main__.load_keys", forbidden)
    monkeypatch.setattr("brand_demo.__main__.LiveClient", forbidden)
    assert main(["--rebuild-report", str(folder)]) == 0
    assert "API çağrısı: 0" in capsys.readouterr().out
    assert all(sha256(p) == checksum for p, checksum in originals.items())
    result = read_json(folder / "report.action-plan.v2.json")
    assert result["enrichment_provenance"]["api_calls"] == 0
    bad = read_json(folder / "report.json")
    bad["actions"] = []
    write_json(folder / "report.json", bad)
    with pytest.raises(ValueError, match="checksum"):
        rebuild(folder)
