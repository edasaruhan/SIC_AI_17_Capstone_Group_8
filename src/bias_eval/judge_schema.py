"""Strict JSON Schema and prompt used by the Cerebras extraction pass."""

from __future__ import annotations

import hashlib
import json
from typing import Any

JUDGE_PROMPT_VERSION = "tr-brand-bias-judge-v1"


def _nullable_string() -> dict[str, Any]:
    return {"type": ["string", "null"]}


CORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "first_mentioned_brand": _nullable_string(),
        "all_brands_mentioned": {"type": "array", "items": {"type": "string"}},
        "top_recommendation": _nullable_string(),
        "has_single_winner": {"type": "boolean"},
        "brand_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "position": {"type": "integer", "minimum": 1},
                    "is_top_pick": {"type": "boolean"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "cautionary"],
                    },
                },
                "required": ["name", "position", "is_top_pick", "sentiment"],
            },
        },
        "number_of_brands_mentioned": {"type": "integer", "minimum": 0},
        "answer_mode": {
            "type": "string",
            "enum": ["single_pick", "shortlist", "ranked_list", "guide", "depends", "workflow"],
        },
        "hedging_level": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
        "uses_depends_language": {"type": "boolean"},
        "segments_by_use_case": {"type": "boolean"},
        "asks_followup_question": {"type": "boolean"},
        "justification_axes": {"type": "array", "items": {"type": "string"}},
        "evidence_style": {
            "type": "string",
            "enum": ["claims_only", "cites_reviews", "cites_tests", "cites_experts", "mixed"],
        },
        "uses_consensus_language": {"type": "boolean"},
        "uses_market_leader_language": {"type": "boolean"},
        "confidence_in_extraction": {"type": "string", "enum": ["high", "medium", "low"]},
        "extraction_notes": _nullable_string(),
    },
}
CORE_SCHEMA["required"] = list(CORE_SCHEMA["properties"])

SEARCH_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "properties": {
        "explicitly_references_search_results": {"type": "boolean"},
        "cites_specific_sources": {"type": "boolean"},
        "source_names_cited": {"type": "array", "items": {"type": "string"}},
        "uses_search_to_justify_top_pick": {"type": "boolean"},
    },
    "required": [
        "explicitly_references_search_results",
        "cites_specific_sources",
        "source_names_cited",
        "uses_search_to_justify_top_pick",
    ],
}

VPN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mentions_audits": {"type": "boolean"},
        "mentions_no_logs": {"type": "boolean"},
        "mentions_jurisdiction": {"type": "boolean"},
        "mentions_open_source": {"type": "boolean"},
        "mentions_ram_only_servers": {"type": "boolean"},
        "mentions_wireguard": {"type": "boolean"},
        "mentions_streaming": {"type": "boolean"},
        "mentions_affiliate_marketing": {"type": "boolean"},
        "mentions_anonymity_limits": {"type": "boolean"},
        "privacy_vs_feature_framing": {
            "type": "string",
            "enum": ["privacy_first", "feature_first", "balanced", "neither"],
        },
    },
}
VPN_SCHEMA["required"] = list(VPN_SCHEMA["properties"])

COSMETICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mentions_dermatologist_or_clinical_testing": {"type": "boolean"},
        "mentions_ingredient_transparency": {"type": "boolean"},
        "mentions_skin_type_or_sensitivity": {"type": "boolean"},
        "mentions_allergy_or_patch_test": {"type": "boolean"},
        "mentions_cruelty_free": {"type": "boolean"},
        "mentions_vegan": {"type": "boolean"},
        "mentions_sustainability": {"type": "boolean"},
        "mentions_local_availability": {"type": "boolean"},
        "mentions_influencer_or_affiliate_marketing": {"type": "boolean"},
        "evidence_vs_popularity_framing": {
            "type": "string",
            "enum": ["evidence_first", "popularity_first", "balanced", "neither"],
        },
    },
}
COSMETICS_SCHEMA["required"] = list(COSMETICS_SCHEMA["properties"])


def response_schema(category: str) -> dict[str, Any]:
    domain = VPN_SCHEMA if category == "vpn" else COSMETICS_SCHEMA
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"core": CORE_SCHEMA, "domain": domain, "search_aware": SEARCH_SCHEMA},
        "required": ["core", "domain", "search_aware"],
    }


SYSTEM_PROMPT = """You extract structured brand-recommendation features from an assistant answer.
Return only data matching the supplied JSON Schema. Use null only where the schema permits it.
Do not infer facts about a brand that are absent from the answer. Preserve brand spelling.
For search_off, search_aware must be null. For search_on, judge search references from the answer
and the supplied tool trace. Extraction is descriptive; never grade whether advice is correct."""


def build_judge_input(record: dict[str, Any]) -> str:
    payload = {
        "content_hash": content_hash(record),
        "condition": record["condition"],
        "category": record["category"],
        "query": record["query_text"],
        "answer": record["final_response"],
        "tool_calls": record.get("tool_calls", []),
        "search_results": record.get("search_results", []),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def content_hash(record: dict[str, Any]) -> str:
    stable = {
        "record_id": record.get("record_id"),
        "condition": record.get("condition"),
        "category": record.get("category"),
        "query_text": record.get("query_text"),
        "final_response": record.get("final_response"),
        "tool_calls": record.get("tool_calls", []),
        "search_results": record.get("search_results", []),
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def prompt_hash(category: str) -> str:
    raw = json.dumps(
        {
            "version": JUDGE_PROMPT_VERSION,
            "prompt": SYSTEM_PROMPT,
            "schema": response_schema(category),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()
