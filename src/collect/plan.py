"""Expand the frozen design files into the exact list of calls to make.

The plan is a pure function of the configuration and the seed, so the same
files always produce the same ``call_id`` set. That is what lets an interrupted
run resume, and what lets a reviewer regenerate the plan and compare.

Four rules from the sprint plan are enforced here rather than left to the
person running the collection:

* eight candidate brands per call, so one call yields eight labelled rows;
* presentation order reshuffled per repetition, because models favour position;
* one variant of a brand per candidate list, to keep lists uncontaminated;
* persona and temperature held fixed across the whole run.

Expected shape of the shared design files
-----------------------------------------
``configs/queries_tr.yaml``::

    version: 1
    sectors:
      - id: kozmetik
        label: Kozmetik ve kişisel bakım
        queries:
          - {id: kozmetik_01, text: "...", intent: genel_oneri}

``configs/brands_tr.yaml``::

    version: 1
    sectors:
      - id: kozmetik
        brands:
          - {id: kozmetik_m01, name: "...", type: kuresel}

``configs/protocols.yaml``::

    version: 1
    persona: "..."
    temperature: 0.0
    repetitions: 20
    candidate_list_size: 8
    protocols:
      alpha: {layer: A, marker: SIRALAMA, template: "..."}

``configs/variants.yaml`` (Layer B)::

    version: 1
    brands:
      - brand_id: kozmetik_m01
        control: "..."
        variants:
          - {id: v01, text: "..."}
"""

from __future__ import annotations

import random
import string
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, read_yaml
from .records import CONDITIONS, CallSpec

CANDIDATE_LABELS = string.ascii_uppercase
BRAND_TYPES = ("kuresel", "yerel", "kucuk_yerel", "kurgusal")


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    intent: str
    sector: str


@dataclass(frozen=True)
class Brand:
    id: str
    name: str
    type: str
    sector: str


@dataclass(frozen=True)
class Protocol:
    name: str
    layer: str
    marker: str
    template: str


@dataclass(frozen=True)
class DesignConfig:
    """The frozen experiment design, as the collector needs to see it."""

    persona: str
    temperature: float
    repetitions: int
    candidate_list_size: int
    queries: tuple[Query, ...]
    brands: tuple[Brand, ...]
    protocols: dict[str, Protocol]
    variants: dict[str, tuple[tuple[str, str], ...]]
    controls: dict[str, str]

    def brands_in(self, sector: str) -> tuple[Brand, ...]:
        return tuple(brand for brand in self.brands if brand.sector == sector)


def _require_list(content: dict[str, Any], key: str, source: str | Path) -> list[Any]:
    values = content.get(key)
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{source}: '{key}' must be a non-empty list")
    return values


def load_queries(path: str | Path) -> tuple[Query, ...]:
    content = read_yaml(path)
    queries: list[Query] = []
    for sector in _require_list(content, "sectors", path):
        sector_id = str(sector.get("id", "")).strip()
        if not sector_id:
            raise ConfigError(f"{path}: every sector needs an id")
        for entry in _require_list(sector, "queries", path):
            queries.append(
                Query(
                    id=str(entry["id"]),
                    text=str(entry["text"]),
                    intent=str(entry.get("intent", "belirtilmemis")),
                    sector=sector_id,
                )
            )
    identifiers = [query.id for query in queries]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigError(f"{path}: query ids must be unique")
    return tuple(queries)


def load_brands(path: str | Path) -> tuple[Brand, ...]:
    content = read_yaml(path)
    brands: list[Brand] = []
    for sector in _require_list(content, "sectors", path):
        sector_id = str(sector.get("id", "")).strip()
        if not sector_id:
            raise ConfigError(f"{path}: every sector needs an id")
        for entry in _require_list(sector, "brands", path):
            brand_type = str(entry.get("type", ""))
            if brand_type not in BRAND_TYPES:
                raise ConfigError(
                    f"{path}: brand {entry.get('id')} has type {brand_type!r}; "
                    f"expected one of {BRAND_TYPES}"
                )
            brands.append(
                Brand(
                    id=str(entry["id"]),
                    name=str(entry["name"]),
                    type=brand_type,
                    sector=sector_id,
                )
            )
    identifiers = [brand.id for brand in brands]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigError(f"{path}: brand ids must be unique")
    return tuple(brands)


def load_design(
    *,
    queries_path: str | Path,
    brands_path: str | Path,
    protocols_path: str | Path,
    variants_path: str | Path | None = None,
) -> DesignConfig:
    """Load the frozen design files the collection plan is built from."""

    protocol_content = read_yaml(protocols_path)
    raw_protocols = protocol_content.get("protocols")
    if not isinstance(raw_protocols, dict) or not raw_protocols:
        raise ConfigError(f"{protocols_path}: 'protocols' must be a non-empty mapping")

    protocols: dict[str, Protocol] = {}
    for name, values in raw_protocols.items():
        values = dict(values or {})
        template = str(values.get("template", ""))
        if "{query}" not in template and "{variant_text}" not in template:
            raise ConfigError(
                f"{protocols_path}: protocol {name} template must reference "
                "{query} (Layer A) or {variant_text} (Layer B)"
            )
        protocols[str(name)] = Protocol(
            name=str(name),
            layer=str(values.get("layer", "A")),
            marker=str(values.get("marker", "SIRALAMA")),
            template=template,
        )

    variants: dict[str, tuple[tuple[str, str], ...]] = {}
    controls: dict[str, str] = {}
    if variants_path is not None and Path(variants_path).exists():
        variant_content = read_yaml(variants_path)
        for entry in _require_list(variant_content, "brands", variants_path):
            brand_id = str(entry["brand_id"])
            controls[brand_id] = str(entry["control"])
            variants[brand_id] = tuple(
                (str(variant["id"]), str(variant["text"]))
                for variant in _require_list(entry, "variants", variants_path)
            )

    return DesignConfig(
        persona=str(protocol_content.get("persona", "")),
        temperature=float(protocol_content.get("temperature", 0.0)),
        repetitions=int(protocol_content.get("repetitions", 20)),
        candidate_list_size=int(protocol_content.get("candidate_list_size", 8)),
        queries=load_queries(queries_path),
        brands=load_brands(brands_path),
        protocols=protocols,
        variants=variants,
        controls=controls,
    )


def rotate_candidates(
    brands: Sequence[Brand],
    *,
    size: int,
    repetition: int,
) -> tuple[Brand, ...]:
    """Pick ``size`` brands for this repetition, rotating so coverage stays even.

    Sampling at random would leave some brands measured far less often than
    others, which would show up later as a brand effect that is really a
    sampling artefact.
    """

    if size > len(brands):
        raise ConfigError(
            f"Candidate list of {size} requested but only {len(brands)} brands are available"
        )
    offset = (repetition * size) % len(brands)
    doubled = list(brands) + list(brands)
    return tuple(doubled[offset : offset + size])


def _cell_rng(seed: int, *parts: object) -> random.Random:
    return random.Random("|".join([str(seed), *(str(part) for part in parts)]))


def _candidate_block(brands: Sequence[Brand]) -> str:
    return "\n".join(
        f"{CANDIDATE_LABELS[index]}) {brand.name}" for index, brand in enumerate(brands)
    )


def build_layer_a_specs(
    design: DesignConfig,
    *,
    run_id: str,
    protocol_name: str,
    model_keys: Sequence[str],
    seed: int,
    conditions: Sequence[str] = CONDITIONS,
) -> Iterator[CallSpec]:
    """One call per (query, condition, model, repetition) with a shuffled list."""

    protocol = design.protocols.get(protocol_name)
    if protocol is None:
        raise ConfigError(f"Unknown protocol {protocol_name!r}")

    for query in design.queries:
        sector_brands = design.brands_in(query.sector)
        if not sector_brands:
            raise ConfigError(f"No brands configured for sector {query.sector!r}")
        for condition in conditions:
            for model_key in model_keys:
                for repetition in range(design.repetitions):
                    selected = rotate_candidates(
                        sector_brands,
                        size=design.candidate_list_size,
                        repetition=repetition,
                    )
                    ordered = list(selected)
                    _cell_rng(seed, query.id, condition, model_key, repetition).shuffle(ordered)
                    prompt = protocol.template.format(
                        query=query.text,
                        intent=query.intent,
                        sector=query.sector,
                        marker=protocol.marker,
                        candidate_block=_candidate_block(ordered),
                        candidate_labels=", ".join(CANDIDATE_LABELS[: len(ordered)]),
                    )
                    yield CallSpec(
                        run_id=run_id,
                        layer="A",
                        protocol=protocol.name,
                        sector=query.sector,
                        query_id=query.id,
                        condition=condition,
                        model_key=model_key,
                        repetition=repetition,
                        persona=design.persona,
                        temperature=design.temperature,
                        prompt=prompt,
                        candidates=tuple(brand.id for brand in ordered),
                    )


def build_layer_b_specs(
    design: DesignConfig,
    *,
    run_id: str,
    protocol_name: str,
    model_keys: Sequence[str],
    seed: int,
    repetitions: int | None = None,
) -> Iterator[CallSpec]:
    """Pair each written variant against the brand's untouched control text.

    What Layer B measures is the difference from the control, not an absolute
    rate, so the control travels in every comparison. Which text is shown first
    is reshuffled per repetition for the same reason Layer A lists are.
    """

    protocol = design.protocols.get(protocol_name)
    if protocol is None:
        raise ConfigError(f"Unknown protocol {protocol_name!r}")
    if not design.variants:
        raise ConfigError("Layer B requires configs/variants.yaml")

    brands_by_id = {brand.id: brand for brand in design.brands}
    total_repetitions = design.repetitions if repetitions is None else repetitions

    for brand_id, brand_variants in design.variants.items():
        brand = brands_by_id.get(brand_id)
        if brand is None:
            raise ConfigError(f"variants.yaml references unknown brand {brand_id!r}")
        control_text = design.controls[brand_id]
        for variant_id, variant_text in brand_variants:
            for model_key in model_keys:
                for repetition in range(total_repetitions):
                    rng = _cell_rng(seed, brand_id, variant_id, model_key, repetition)
                    control_first = rng.random() < 0.5
                    first, second = (
                        (control_text, variant_text)
                        if control_first
                        else (variant_text, control_text)
                    )
                    prompt = protocol.template.format(
                        brand=brand.name,
                        sector=brand.sector,
                        marker=protocol.marker,
                        control_text=control_text,
                        variant_text=variant_text,
                        option_a=first,
                        option_b=second,
                    )
                    yield CallSpec(
                        run_id=run_id,
                        layer="B",
                        protocol=protocol.name,
                        sector=brand.sector,
                        query_id=brand_id,
                        condition="search_off",
                        model_key=model_key,
                        repetition=repetition,
                        persona=design.persona,
                        temperature=design.temperature,
                        prompt=prompt,
                        variant_ids=(
                            ("control", variant_id) if control_first else (variant_id, "control")
                        ),
                    )


def summarize_plan(specs: Iterable[CallSpec]) -> dict[str, Any]:
    """Counts a reviewer can check before any money is spent."""

    materialised = list(specs)
    call_ids = [spec.call_id for spec in materialised]
    by_model: dict[str, int] = {}
    by_condition: dict[str, int] = {}
    for spec in materialised:
        by_model[spec.model_key] = by_model.get(spec.model_key, 0) + 1
        by_condition[spec.condition] = by_condition.get(spec.condition, 0) + 1
    return {
        "calls": len(materialised),
        "unique_call_ids": len(set(call_ids)),
        "queries": len({spec.query_id for spec in materialised}),
        "by_model": dict(sorted(by_model.items())),
        "by_condition": dict(sorted(by_condition.items())),
    }
