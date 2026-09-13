import asyncio
import json
from collections import Counter

import httpx
import pandas as pd

from description_lab import clients, design, round2, round2_report
from evidence_eval.io import write_json


def _cards(entry: dict) -> list[dict]:
    return json.loads(entry["payload"]["messages"][1]["content"])["urunler"]


def test_first_round_gemini_plan_is_unchanged_by_the_assistant_registry():
    assert design.plan_id() == "5b6f06e7d2f7a8d8"
    assert design.plan_id("cerebras") != design.plan_id()


def test_only_cerebras_payloads_carry_extra_fields():
    gemini = design.job("vpn", "fictional", "control", 0)["payload"]
    cerebras = design.job("vpn", "fictional", "control", 0, "cerebras")["payload"]
    assert "reasoning_effort" not in gemini and gemini["model"] == design.MODEL
    assert cerebras["reasoning_effort"] == "low" and cerebras["model"] == "gpt-oss-120b"
    assert gemini["messages"] == cerebras["messages"]


def test_round_two_plan_size_and_unique_keys():
    jobs = round2.plan_jobs()
    assert len(jobs) == 2 * (6 * 13 + 10 + 4 * 10) == 256
    assert len({entry["key"] for entry in jobs}) == len(jobs)
    assert round2.plan_id("cerebras") != round2.plan_id()


def test_tournament_puts_every_sentence_at_every_position_once_per_step():
    for category in design.CATEGORIES:
        for step in round2.STEPS:
            seen = Counter()
            for index in range(len(round2.SENTENCES)):
                entry = round2.tournament_job(category, step, index)
                assert len({c["sentence"] for c in entry["cards"]}) == 5
                assert len(set(entry["brands"])) == 5
                assert set(entry["brands"]) <= set(round2.FICTIONAL[category])
                seen.update((c["sentence"], c["position"]) for c in entry["cards"])
            assert len(seen) == 13 * 5 and set(seen.values()) == {1}


def test_fabricated_claims_sit_only_on_fictional_brands():
    for entry in round2.plan_jobs():
        for card in entry["cards"]:
            if card["sentence"] == "fabricated_claim":
                assert card["brand"] in round2.FICTIONAL[entry["category"]]


def test_cards_in_the_payload_match_the_recorded_sentences():
    for entry in round2.plan_jobs()[:40]:
        category = design.CATEGORIES[entry["category"]]
        for shown, meta in zip(_cards(entry), entry["cards"], strict=True):
            assert shown["marka"] == meta["brand"]
            text = round2.sentence(category, meta["sentence"])
            assert shown["aciklama"] == " ".join(p for p in (category.spec, text) if p)
            expected = category.discount_price if meta["sentence"] == "price" else category.price
            assert shown["fiyat"] == expected


def test_cross_block_holds_incumbent_and_fictional_apart_with_three_real_rivals():
    for rep in range(10):
        entry = round2.cross_job("sunscreen", "statistics", rep)
        by_brand = {c["brand"]: c for c in entry["cards"]}
        assert by_brand["La Roche-Posay"]["sentence"] == "control"
        assert by_brand["Lumera"]["sentence"] == "statistics"
        assert by_brand["La Roche-Posay"]["position"] != by_brand["Lumera"]["position"]
        assert len(set(entry["brands"]) & set(design.SUNSCREEN.competitors)) == 3


def test_filler_target_visits_every_position_twice():
    positions = Counter(
        next(
            c["position"]
            for c in round2.filler_job("vpn", rep)["cards"]
            if c["sentence"] == "filler"
        )
        for rep in range(10)
    )
    assert positions == Counter({1: 2, 2: 2, 3: 2, 4: 2, 5: 2})


def _write(folder, entry, text):
    write_json(
        folder / "steps" / f"{entry['key']}.json", {"status": "completed", "result": {"text": text}}
    )


def test_outcomes_map_the_pick_to_its_sentence_and_detect_the_echoed_claim(tmp_path):
    entry = next(
        e
        for e in round2.plan_jobs()
        if e["block"] == "tournament"
        and e["category"] == "vpn"
        and any(c["sentence"] == "fabricated_claim" for c in e["cards"])
    )
    fabricated = next(c for c in entry["cards"] if c["sentence"] == "fabricated_claim")
    other = next(c for c in entry["cards"] if c["sentence"] != "fabricated_claim")
    _write(
        tmp_path,
        entry,
        f"Size **{other['brand']}** öneririm. {fabricated['brand']} MIT denetimi iddiası taşıyor.",
    )
    table = round2_report.outcomes(tmp_path, [entry])
    row = table.iloc[0]
    assert row.picked == other["brand"] and row.picked_sentence == other["sentence"]
    assert row.claim_echo == 1 and row.warned == 1

    shares = round2_report.win_shares(table)
    won = shares[(shares.scope == "ALL") & (shares.sentence == other["sentence"])].iloc[0]
    assert won.appearances == 1 and won.wins == 1 and won.share == 1.0
    echo = round2_report.echo_rates(table)
    assert echo[echo.block == "ALL"].iloc[0].picked == 0.0


def test_render_names_the_four_questions():
    rows = [
        {
            "block": "tournament",
            "category": "vpn",
            "picked_sentence": "statistics",
            "picked_position": 1.0,
            "sentences": "statistics|control|filler|price|technical",
            "first": 0,
            "incumbent_first": 0,
            "variant": None,
            "fabricated_shown": 0,
            "claim_echo": float("nan"),
            "warned": float("nan"),
            "pick_method": "bold",
        },
        {
            "block": "filler",
            "category": "vpn",
            "picked_sentence": "filler",
            "picked_position": 2.0,
            "sentences": "control|filler|control|control|control",
            "first": 1,
            "incumbent_first": 0,
            "variant": "filler",
            "fabricated_shown": 0,
            "claim_echo": float("nan"),
            "warned": float("nan"),
            "pick_method": "bold",
        },
        {
            "block": "cross",
            "category": "vpn",
            "picked_sentence": "fabricated_claim",
            "picked_position": 3.0,
            "sentences": "control|control|fabricated_claim|control|control",
            "first": 1,
            "incumbent_first": 0,
            "variant": "fabricated_claim",
            "fabricated_shown": 1,
            "claim_echo": 1.0,
            "warned": 0.0,
            "pick_method": "bold",
        },
    ]
    table = pd.DataFrame(rows)
    text = round2_report.render(
        table,
        round2_report.win_shares(table),
        round2_report.filler_contrast(table, None),
        round2_report.cross_shares(table),
        round2_report.echo_rates(table),
        assistant="cerebras",
        planned=256,
    )
    for heading in ("Turnuva", "Dolgu cümlesi", "Çaprazlama", "Uydurma iddia", "gpt-oss-120b"):
        assert heading in text


def _completion(finish="stop", text="Size **Lumera** öneririm."):
    return {
        "choices": [{"finish_reason": finish, "message": {"content": text}}],
        "usage": {"total_tokens": 9},
    }


def _cerebras(responses, monkeypatch, *, service="cerebras", calls=1):
    monkeypatch.setenv(clients.KEY_ENV, "test-key")
    seen, slept = [], []

    def handler(request):
        seen.append(request)
        return responses[len(seen) - 1]

    async def sleep(seconds):
        slept.append(seconds)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = clients.CerebrasClient(
                http, min_interval=25.0, waits=(1.0, 2.0), clock=lambda: 0.0, sleep=sleep
            )
            return [await client(service, {"model": "m"}) for _ in range(calls)]

    try:
        return asyncio.run(run()), seen, slept
    except ValueError as exc:
        return exc, seen, slept


def test_cerebras_client_throttles_waits_out_429_and_keeps_usage(monkeypatch):
    result, seen, slept = _cerebras(
        [
            httpx.Response(429),
            httpx.Response(200, json=_completion()),
            httpx.Response(200, json=_completion()),
        ],
        monkeypatch,
        calls=2,
    )
    assert isinstance(result, list) and result[0]["text"] == "Size **Lumera** öneririm."
    assert result[0]["usage"]["total_tokens"] == 9
    assert str(seen[0].url) == "https://api.cerebras.ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert slept == [1.0, 25.0, 25.0]


def test_cerebras_client_rejects_truncation_other_services_and_hides_bodies(monkeypatch):
    result, _, _ = _cerebras([httpx.Response(200, json=_completion(finish="length"))], monkeypatch)
    assert isinstance(result, ValueError) and "kesilmiş" in str(result)
    result, _, _ = _cerebras([httpx.Response(402, json={"error": "secret detail"})], monkeypatch)
    assert isinstance(result, ValueError) and "HTTP 402" in str(result)
    assert "secret detail" not in str(result) and "test-key" not in str(result)
    result, seen, _ = _cerebras([], monkeypatch, service="gemini")
    assert isinstance(result, ValueError) and not seen


def test_collect_records_the_real_service(tmp_path):
    async def fake(service, payload):
        return {"text": f"{service}:{payload['n']}"}

    failures = asyncio.run(
        clients.collect(
            [{"key": "k1", "payload": {"n": 1}}],
            tmp_path,
            fake,
            "cerebras",
            concurrency=1,
            retry_failed=False,
        )
    )
    receipt = json.loads((tmp_path / "steps" / "k1.json").read_text())
    assert (
        failures == []
        and receipt["service"] == "cerebras"
        and receipt["result"]["text"] == "cerebras:1"
    )
