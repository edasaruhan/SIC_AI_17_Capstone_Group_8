"""Prepare versioned analysis outputs without touching legacy experiment files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import turkish_data
from modeling import datasets, splits
from modeling.brands import load_registry
from modeling.features import load_lexicon, snippet_features
from modeling.pairs import text_key
from reference_data import EXPECTED_CATEGORY_COUNTS, EXPECTED_MODELS

from . import VERSION
from .evidence import associations, retrieved_sources
from .io import digest, read_json, sha256, write_json

DEFAULT_ROOT = Path("data/processed/evidence_v1")
EN_REVISION = "400da04eced51d3afe52b6d20c0207fd613f8a4a"
EN_CONTENT_SHA256 = "ea4ddb106923d86d088866b8b57d264afe870847addd84148f1aa21da5ef0caf"


def validate_inputs(english: Path, turkish: Path) -> dict:
    tr_frame = pd.read_parquet(turkish)
    tr = turkish_data.check(tr_frame)
    if turkish_data.content_sha256(tr_frame) != turkish_data.CONTENT_SHA256:
        raise ValueError("Turkish response content differs from the pinned release")
    en = pd.read_parquet(english)
    if turkish_data.content_sha256(en) != EN_CONTENT_SHA256:
        raise ValueError("English prepared content differs from the frozen reference")
    if len(en) != 9586 or en.record_id.nunique() != len(en):
        raise ValueError("English reference must contain 9586 distinct responses")
    if en.category.value_counts().to_dict() != EXPECTED_CATEGORY_COUNTS:
        raise ValueError("English domain distribution changed")
    if set(en.model_id) != EXPECTED_MODELS:
        raise ValueError("English reference models changed")
    if en.final_response.isna().any():
        raise ValueError("Missing English response")
    return {"tr": tr, "en": {"rows": len(en)}}


def contract(english: Path, turkish: Path) -> dict:
    """Record actual local input hashes, not an invented verified download history."""
    files = sorted(Path("configs/modeling").rglob("*.yaml"))
    files += sorted(Path("configs/modeling").rglob("*.json"))
    files += sorted(Path("src/modeling").glob("*.py"))
    files += sorted(Path("src/evidence_eval").glob("*.py"))
    files += [Path("src/turkish_data.py"), Path("uv.lock")]
    return {
        "version": VERSION,
        "seed": 42,
        "candidate_policy": "full_frozen_registry",
        "sources": {
            "en": {
                "path": str(english.resolve()),
                "prepared_sha256": sha256(english),
                "expected_revision": EN_REVISION,
                "verification": "frozen_prepared_content_hash",
            },
            "tr": {
                "path": str(turkish.resolve()),
                "prepared_sha256": sha256(turkish),
                "expected_revision": turkish_data.REVISION,
                "expected_source_sha256": turkish_data.SOURCE_SHA256,
                "verification": (
                    "source_hash"
                    if sha256(turkish) == turkish_data.SOURCE_SHA256
                    else "frozen_prepared_content_hash"
                ),
            },
        },
        "implementation": {str(p): sha256(p) for p in files},
    }


def pair_table(frame: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Fixed candidate registry; labels never decide which candidates exist."""
    groups = {key: group for key, group in evidence.groupby(["record_id", "brand"])}
    rows = []
    for record in frame.to_dict("records"):
        registry = load_registry(record["category"])
        lexicon = load_lexicon(record["language"])
        for brand in registry.brands:
            hits = groups.get((record["record_id"], brand))
            texts, positions, types = [], [], []
            if hits is not None:
                texts = [str(r["title"]) + " " + str(r["snippet"]) for r in hits.to_dict("records")]
                positions = [
                    int(p) for p in hits.position if p is not None and pd.notna(p) and int(p) > 0
                ]
                types = list(hits.source_type)
            rows.append(
                {
                    **{
                        k: record[k]
                        for k in (
                            "record_id",
                            "language",
                            "category",
                            "query_id",
                            "model_id",
                            "condition",
                            "run_index",
                        )
                    },
                    "brand": brand,
                    "y_mention": int(brand in record["brands_mentioned"]),
                    "y_top": int(brand == record["top_recommendation"]),
                    "response_decided": int(bool(record["top_recommendation"])),
                    "confidence": record["confidence"],
                    "n_results_mentioning": len(texts),
                    "in_search_results": int(bool(texts)),
                    "best_position": min(positions, default=0),
                    **{
                        f"n_{kind}": types.count(kind)
                        for kind in (
                            "official",
                            "editorial",
                            "affiliate",
                            "forum",
                            "retailer",
                            "unknown",
                        )
                    },
                    "n_other": types.count("unknown") + types.count("retailer"),
                    **snippet_features(texts, [text_key(t) for t in texts], lexicon),
                }
            )
    return pd.DataFrame(rows)


def verify(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("Preparation incomplete; rerun prepare with the same contract")
    for name, expected in manifest["artifacts"].items():
        if not (root / name).is_file() or sha256(root / name) != expected:
            raise ValueError(f"Stale or altered artifact: {name}")
    for name, expected in manifest["contract"]["implementation"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Implementation changed: {name}; use a new experiment --root")
    return manifest


def responses(root: Path, track: str) -> pd.DataFrame:
    return pd.DataFrame(read_json(root / f"responses_{track}.json"))


def prepare(root: Path, english: Path, turkish: Path) -> dict:
    quality = validate_inputs(english, turkish)
    identity = contract(english, turkish)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["contract"] != identity:
            raise ValueError("Inputs/config/code changed; use a new --root, keep the old run")
        if old.get("status") == "completed":
            return verify(root)
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to adopt an existing directory without a manifest")
    manifest = {
        "status": "preparing",
        "contract": identity,
        "identity": digest(identity),
        "quality": quality,
    }
    write_json(manifest_path, manifest)
    previous = datasets.EN_PARQUET, datasets.TR_PARQUET
    artifacts = []
    try:
        datasets.EN_PARQUET, datasets.TR_PARQUET = english, turkish
        for track in ("en", "tr"):
            frame = datasets.load_track(track).frame
            # Retain the existing frozen query splits; no random repartitioning.
            folds = splits.load_frozen(track)
            if set(frame.query_id) != set(folds):
                raise ValueError("Frozen query split differs from the input corpus")
            write_json(
                root / f"responses_{track}.json",
                json.loads(str(frame.to_json(orient="records", force_ascii=False))),
            )
            write_json(root / f"folds_{track}.json", folds)
            sources = retrieved_sources(frame)
            evidence = associations(sources)
            pairs = pair_table(frame, evidence)
            for name, table in (("sources", sources), ("evidence", evidence), ("pairs", pairs)):
                table.to_parquet(root / f"{name}_{track}.parquet", index=False)
                artifacts.append(f"{name}_{track}.parquet")
            artifacts += [f"responses_{track}.json", f"folds_{track}.json"]
    finally:
        datasets.EN_PARQUET, datasets.TR_PARQUET = previous
    manifest.update(status="completed", artifacts={name: sha256(root / name) for name in artifacts})
    write_json(manifest_path, manifest)
    return manifest
