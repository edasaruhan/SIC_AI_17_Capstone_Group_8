"""Load the English and Turkish corpora into one response-level schema.

The two sources store the same 19-column judge output differently: the English
reference parquet arrives flattened (``core__top_recommendation``), the Turkish
export keeps JSON strings. Both are reduced here to the same frame so every
downstream stage is written once and run twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .brands import load_registry

EN_PARQUET = Path("data/interim/reference.parquet")
TR_PARQUET = Path("data/interim/turkish_raw.parquet")

RESPONSE_COLUMNS = [
    "record_id",
    "language",
    "category",
    "query_id",
    "query_text",
    "model_id",
    "provider",
    "condition",
    "run_index",
    "final_response",
    "brands_mentioned",
    "top_recommendation",
    "first_mentioned_brand",
    "organic",
    "confidence",
    "answer_mode",
    "hedging_level",
    "evidence_style",
    "number_of_brands_mentioned",
    "response_length_words",
    "num_search_results_returned",
]


@dataclass(frozen=True)
class Track:
    """One training track: a language, a corpus and the categories it covers."""

    name: str
    language: str
    categories: tuple[str, ...]
    frame: pd.DataFrame


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _organic(search_results: Any) -> list[dict[str, Any]]:
    """Flatten every search round into one position-ordered organic result list."""
    rounds = _as_list(search_results)
    organic: list[dict[str, Any]] = []
    for item in rounds:
        payload = json.loads(item) if isinstance(item, str) else item
        if not isinstance(payload, dict):
            continue
        for result in payload.get("organic") or []:
            if isinstance(result, dict):
                organic.append(result)
    return organic


def _load_english() -> pd.DataFrame:
    frame = pd.read_parquet(EN_PARQUET)
    out = pd.DataFrame(
        {
            "record_id": frame["record_id"],
            "language": "en",
            "category": frame["category"],
            "query_id": frame["query_id"],
            "query_text": frame["query_text"],
            "model_id": frame["model_id"],
            "provider": frame.get("provider", pd.Series(["unknown"] * len(frame))),
            "condition": frame["condition"],
            "run_index": frame["run_index"],
            "final_response": frame["final_response"],
            "confidence": frame["core__confidence_in_extraction"],
            "answer_mode": frame["core__answer_mode"],
            "hedging_level": frame["core__hedging_level"],
            "evidence_style": frame["core__evidence_style"],
            "number_of_brands_mentioned": frame["core__number_of_brands_mentioned"],
            "response_length_words": frame["deterministic__response_length_words"],
            "num_search_results_returned": frame["deterministic__num_search_results_returned"],
        }
    )
    out["raw_mentions"] = [
        list(x) if x is not None else [] for x in frame["core__all_brands_mentioned"]
    ]
    out["raw_top"] = frame["top_recommendation_canonical"]
    out["raw_first"] = frame["core__first_mentioned_brand"]
    out["organic"] = [_organic(x) for x in frame["search_results__items"]]
    return out


def _load_turkish() -> pd.DataFrame:
    frame = pd.read_parquet(TR_PARQUET)
    cores = [json.loads(x) for x in frame["core"]]
    out = pd.DataFrame(
        {
            "record_id": frame["record_id"],
            "language": "tr",
            "category": frame["category"],
            "query_id": frame["query_id"],
            "query_text": frame["query_text"],
            "model_id": frame["model_id"],
            "provider": frame["provider"],
            "condition": frame["condition"],
            "run_index": frame["run_index"],
            "final_response": frame["final_response"],
            "confidence": [c["confidence_in_extraction"] for c in cores],
            "answer_mode": [c["answer_mode"] for c in cores],
            "hedging_level": [c["hedging_level"] for c in cores],
            "evidence_style": [c["evidence_style"] for c in cores],
            "number_of_brands_mentioned": [c["number_of_brands_mentioned"] for c in cores],
        }
    )
    deterministic = [json.loads(x) for x in frame["deterministic"]]
    out["response_length_words"] = [d["response_length_words"] for d in deterministic]
    out["num_search_results_returned"] = [d["num_search_results_returned"] for d in deterministic]
    out["raw_mentions"] = [list(c["all_brands_mentioned"]) for c in cores]
    out["raw_top"] = [c["top_recommendation"] for c in cores]
    out["raw_first"] = [c["first_mentioned_brand"] for c in cores]
    out["organic"] = [_organic(x) for x in frame["search_results"]]
    return out


def _name(value: Any) -> str | None:
    """A judge field is either a brand name or nothing; NaN counts as nothing."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value)


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    mentioned: list[list[str]] = []
    top: list[str | None] = []
    first: list[str | None] = []
    for _, row in frame.iterrows():
        registry = load_registry(str(row["category"]))
        mentioned.append(registry.resolve_all(row["raw_mentions"]))
        top.append(registry.resolve(_name(row["raw_top"])))
        first.append(registry.resolve(_name(row["raw_first"])))
    frame = frame.copy()
    frame["brands_mentioned"] = mentioned
    frame["top_recommendation"] = top
    frame["first_mentioned_brand"] = first
    return pd.DataFrame({name: frame[name] for name in RESPONSE_COLUMNS})


def load_track(name: str) -> Track:
    """Load one track by name: ``en`` (four domains) or ``tr`` (two domains)."""
    if name == "en":
        frame = _canonicalize(_load_english())
        return Track("en", "en", tuple(sorted(frame["category"].unique())), frame)
    if name == "tr":
        frame = _canonicalize(_load_turkish())
        return Track("tr", "tr", tuple(sorted(frame["category"].unique())), frame)
    raise ValueError(f"Unknown track {name!r}; expected 'en' or 'tr'")
