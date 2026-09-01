import asyncio
from pathlib import Path

import pytest

from collect.config import ConfigError, load_collection_settings
from collect.plan import (
    Brand,
    DesignConfig,
    build_layer_a_specs,
    build_layer_b_specs,
    load_design,
    rotate_candidates,
    summarize_plan,
)
from collect.rate_limit import TokenBucket


class FakeClock:
    """Deterministic clock so pacing can be asserted without real waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += delay


def test_token_bucket_allows_a_burst_then_paces() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate_per_second=10.0, capacity=2.0, now=clock, sleep=clock.sleep)

    async def scenario() -> list[float]:
        return [await bucket.acquire() for _ in range(4)]

    waits = asyncio.run(scenario())

    assert waits[0] == 0.0 and waits[1] == 0.0, "the burst must not be throttled"
    assert waits[2] == pytest.approx(0.1) and waits[3] == pytest.approx(0.1)
    assert clock.now == pytest.approx(0.2)


def test_token_bucket_rejects_impossible_requests() -> None:
    bucket = TokenBucket(rate_per_second=1.0, capacity=2.0)

    with pytest.raises(ValueError, match="Cannot acquire"):
        asyncio.run(bucket.acquire(3.0))
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucket(rate_per_second=0.0)


def make_brands(count: int) -> tuple[Brand, ...]:
    return tuple(
        Brand(id=f"b{index}", name=f"Marka {index}", type="yerel", sector="kozmetik")
        for index in range(count)
    )


def test_rotation_covers_every_brand_evenly() -> None:
    brands = make_brands(4)
    appearances = {brand.id: 0 for brand in brands}
    for repetition in range(12):
        for brand in rotate_candidates(brands, size=3, repetition=repetition):
            appearances[brand.id] += 1

    assert len(set(appearances.values())) == 1, f"uneven brand coverage: {appearances}"


def test_rotation_refuses_a_list_longer_than_the_brand_universe() -> None:
    with pytest.raises(ConfigError, match="only 4 brands"):
        rotate_candidates(make_brands(4), size=8, repetition=0)


def test_layer_a_plan_has_the_expected_shape(design: DesignConfig) -> None:
    specs = list(
        build_layer_a_specs(
            design,
            run_id="test-run",
            protocol_name="alpha",
            model_keys=["m1", "m2"],
            seed=42,
        )
    )

    # 2 queries x 2 conditions x 2 models x 2 repetitions
    assert len(specs) == 16
    assert {spec.condition for spec in specs} == {"search_on", "search_off"}
    assert all(len(spec.candidates) == 3 for spec in specs)
    assert all(len(set(spec.candidates)) == 3 for spec in specs), "a brand may not repeat in a list"
    assert all(spec.temperature == 0.0 for spec in specs), "temperature is held fixed"
    assert all(spec.persona == design.persona for spec in specs), "persona is held fixed"


def test_call_ids_are_unique_and_stable(design: DesignConfig) -> None:
    def plan() -> list[str]:
        return [
            spec.call_id
            for spec in build_layer_a_specs(
                design, run_id="test-run", protocol_name="alpha", model_keys=["m1"], seed=42
            )
        ]

    first, second = plan(), plan()

    assert first == second, "the same design must always produce the same call ids"
    assert len(set(first)) == len(first), "call ids must not collide"


def test_a_different_run_id_produces_a_different_plan(design: DesignConfig) -> None:
    def ids(run_id: str) -> set[str]:
        return {
            spec.call_id
            for spec in build_layer_a_specs(
                design, run_id=run_id, protocol_name="alpha", model_keys=["m1"], seed=42
            )
        }

    assert ids("run-a").isdisjoint(ids("run-b"))


def test_presentation_order_is_reshuffled_across_repetitions(design: DesignConfig) -> None:
    orders = {
        spec.repetition: spec.candidates
        for spec in build_layer_a_specs(
            design, run_id="test-run", protocol_name="alpha", model_keys=["m1"], seed=42
        )
        if spec.query_id == "kozmetik_01" and spec.condition == "search_off"
    }

    assert len(orders) == 2
    assert orders[0] != orders[1], "position bias is only cancelled if the order changes"


def test_prompt_renders_the_candidate_block_in_presentation_order(design: DesignConfig) -> None:
    spec = next(
        iter(
            build_layer_a_specs(
                design, run_id="test-run", protocol_name="alpha", model_keys=["m1"], seed=42
            )
        )
    )
    names = {brand.id: brand.name for brand in design.brands}

    lines = [line for line in spec.prompt.splitlines() if line.startswith(("A)", "B)", "C)"))]

    assert lines == [
        f"{label}) {names[brand_id]}"
        for label, brand_id in zip("ABC", spec.candidates, strict=True)
    ]
    assert "SIRALAMA" in spec.prompt


def test_layer_b_pairs_every_variant_against_the_control(design: DesignConfig) -> None:
    specs = list(
        build_layer_b_specs(
            design, run_id="test-run", protocol_name="gamma", model_keys=["m1"], seed=42
        )
    )

    # 1 brand x 2 variants x 1 model x 2 repetitions
    assert len(specs) == 4
    assert all(spec.layer == "B" for spec in specs)
    assert all("control" in spec.variant_ids for spec in specs)
    assert all(len(spec.variant_ids) == 2 for spec in specs)
    assert {frozenset(spec.variant_ids) for spec in specs} == {
        frozenset({"control", "v01"}),
        frozenset({"control", "v02"}),
    }


def test_summary_counts_match_the_plan(design: DesignConfig) -> None:
    specs = list(
        build_layer_a_specs(
            design, run_id="test-run", protocol_name="alpha", model_keys=["m1", "m2"], seed=42
        )
    )

    summary = summarize_plan(specs)

    assert summary["calls"] == summary["unique_call_ids"] == 16
    assert summary["by_model"] == {"m1": 8, "m2": 8}
    assert summary["by_condition"] == {"search_off": 8, "search_on": 8}


def test_unknown_protocol_is_refused(design: DesignConfig) -> None:
    with pytest.raises(ConfigError, match="delta"):
        list(
            build_layer_a_specs(
                design, run_id="r", protocol_name="delta", model_keys=["m1"], seed=42
            )
        )


def test_brand_type_outside_the_four_categories_is_refused(design_dir: Path) -> None:
    (design_dir / "brands_tr.yaml").write_text(
        "version: 1\nsectors:\n  - id: kozmetik\n    brands:\n"
        '      - {id: x, name: "X", type: uydurma}\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="uydurma"):
        load_design(
            queries_path=design_dir / "queries_tr.yaml",
            brands_path=design_dir / "brands_tr.yaml",
            protocols_path=design_dir / "protocols.yaml",
        )


def test_shipped_collection_config_loads() -> None:
    settings = load_collection_settings("configs/collect.yaml")

    assert {model.provider for model in settings.models} == {"anthropic", "openai", "gemini"}
    assert settings.daily_spend_cap_usd > 0
    assert all(model.api_key_env.endswith("_API_KEY") for model in settings.models)
