"""Shared fixtures for the collection pipeline tests.

The design files here are deliberately tiny - four brands, two queries - so a
whole plan can be enumerated in a test and checked by eye.
"""

from pathlib import Path

import pytest

from collect.config import CollectionSettings, RetryPolicy
from collect.plan import DesignConfig, load_design
from collect.providers import ModelConfig

QUERIES_YAML = """
version: 1
sectors:
  - id: kozmetik
    label: Kozmetik ve kişisel bakım
    queries:
      - {id: kozmetik_01, text: "En iyi nemlendirici hangisi?", intent: genel_oneri}
      - {id: kozmetik_02, text: "Uygun fiyatlı güneş kremi önerir misin?", intent: fiyat}
"""

BRANDS_YAML = """
version: 1
sectors:
  - id: kozmetik
    brands:
      - {id: koz_m01, name: "Küresel Marka", type: kuresel}
      - {id: koz_m02, name: "Yerel Marka", type: yerel}
      - {id: koz_m03, name: "Işık Bakım", type: kucuk_yerel}
      - {id: koz_m04, name: "Nuvella", type: kurgusal}
"""

PROTOCOLS_YAML = """
version: 1
persona: "Sen Türkçe konuşan bir alışveriş danışmanısın."
temperature: 0.0
repetitions: 2
candidate_list_size: 3
protocols:
  alpha:
    layer: A
    marker: SIRALAMA
    template: |
      Soru: {query}
      Adaylar:
      {candidate_block}
      Son satırda {marker}: [{candidate_labels}] biçiminde sırala.
  gamma:
    layer: B
    marker: SECIM
    template: |
      {brand} markası için iki açıklama:
      A) {option_a}
      B) {option_b}
      Karşılaştırma: {variant_text}
      Son satırda {marker}: A veya B yaz.
"""

VARIANTS_YAML = """
version: 1
brands:
  - brand_id: koz_m02
    control: "Yerel Marka nemlendirici üretir."
    variants:
      - {id: v01, text: "Yerel Marka nemlendiricisi 12 saat nem sağlar."}
      - {id: v02, text: "Yerel Marka nemlendiricisi dermatolog testlidir."}
"""


@pytest.fixture
def design_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "queries_tr.yaml").write_text(QUERIES_YAML, encoding="utf-8")
    (directory / "brands_tr.yaml").write_text(BRANDS_YAML, encoding="utf-8")
    (directory / "protocols.yaml").write_text(PROTOCOLS_YAML, encoding="utf-8")
    (directory / "variants.yaml").write_text(VARIANTS_YAML, encoding="utf-8")
    return directory


@pytest.fixture
def design(design_dir: Path) -> DesignConfig:
    return load_design(
        queries_path=design_dir / "queries_tr.yaml",
        brands_path=design_dir / "brands_tr.yaml",
        protocols_path=design_dir / "protocols.yaml",
        variants_path=design_dir / "variants.yaml",
    )


@pytest.fixture
def model() -> ModelConfig:
    return ModelConfig(
        key="test_model",
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="TEST_API_KEY",
        base_url="https://api.test.local",
        requests_per_second=1000.0,
        burst=1000.0,
        input_cost_per_million=1.0,
        output_cost_per_million=5.0,
    )


@pytest.fixture
def settings(tmp_path: Path, model: ModelConfig) -> CollectionSettings:
    return CollectionSettings(
        raw_dir=tmp_path / "raw",
        models=(model,),
        concurrency=4,
        daily_spend_cap_usd=1000.0,
        retry=RetryPolicy(
            max_attempts=3,
            initial_seconds=0.001,
            max_seconds=0.002,
            jitter_seconds=0.0,
        ),
    )
