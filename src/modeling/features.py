"""Model inputs: brand priors, and language features of the retrieved snippets.

Two rules govern this module.

**Nothing derived from the answer may become a feature.** The task is to predict
what the assistant will say from what it saw beforehand, so response length,
hedging level, answer mode, evidence style and the judge's brand counts are all
labels or label-adjacent, never inputs. The feature set is therefore: the brand's
prior standing, the retrieved content supporting it, and the run's condition,
model and category.

**Priors are fitted inside the training folds only.** A brand prior estimated on
all rows would carry the held-out queries' own outcomes back into training. Every
prior here is computed from a caller-supplied training mask and smoothed toward
the category base rate so unseen brands do not get an undefined value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import yaml

LEXICON_DIR = Path("configs/modeling/lexicons")
PRIOR_SMOOTHING = 10.0


def select(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Boolean row selection that keeps its DataFrame type.

    Plain ``frame[mask]`` is typed as a DataFrame/Series union, which then leaks
    through every later call. Narrowing once here keeps the rest of the pipeline
    readable instead of casting at each use.
    """
    return cast(pd.DataFrame, frame[mask])

# Features the model is allowed to see. Anything absent here is either a label,
# an identifier, or derived from the answer being predicted.
NUMERIC_FEATURES = [
    "prior_top_off",
    "prior_mention_off",
    "prior_top_all",
    "prior_mention_all",
    "in_search_results",
    "n_results_mentioning",
    "best_position",
    "best_position_missing",
    "n_official",
    "n_editorial",
    "n_affiliate",
    "n_forum",
    "n_other",
    "lex_authority",
    "lex_social_proof",
    "lex_specificity",
    "lex_hedging",
    "lex_superlative",
    "snippet_words",
    "snippet_has_year",
    "snippet_digit_share",
]
CATEGORICAL_FEATURES = ["condition", "model_id", "category"]

# Feature blocks, so a model family can be built up one block at a time.
PRIOR_FEATURES = ["prior_top_off", "prior_mention_off", "prior_top_all", "prior_mention_all"]
STRUCTURAL_FEATURES = [
    "in_search_results",
    "n_results_mentioning",
    "best_position",
    "best_position_missing",
    "n_official",
    "n_editorial",
    "n_affiliate",
    "n_forum",
    "n_other",
]
LANGUAGE_FEATURES = [
    "lex_authority",
    "lex_social_proof",
    "lex_specificity",
    "lex_hedging",
    "lex_superlative",
    "snippet_words",
    "snippet_has_year",
    "snippet_digit_share",
]

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class Lexicon:
    """Versioned term groups for one language."""

    language: str
    version: int
    groups: dict[str, tuple[str, ...]]


@cache
def load_lexicon(language: str) -> Lexicon:
    path = LEXICON_DIR / f"{language}.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = {
        str(name): tuple(str(term) for term in terms)
        for name, terms in (document.get("groups") or {}).items()
    }
    return Lexicon(str(document["language"]), int(document["version"]), groups)


def snippet_features(texts: list[str], normalised: list[str], lexicon: Lexicon) -> dict[str, float]:
    """Summarise the retrieved results that mention one brand.

    Counts are per-snippet rates rather than raw totals, so a brand appearing in
    ten results is not automatically scored as more authoritative than one
    appearing in two.
    """
    if not texts:
        return {
            **{f"lex_{name}": 0.0 for name in lexicon.groups},
            "snippet_words": 0.0,
            "snippet_has_year": 0.0,
            "snippet_digit_share": 0.0,
        }
    joined = " ".join(normalised)
    words = joined.split()
    counts = {}
    for name, terms in lexicon.groups.items():
        counts[f"lex_{name}"] = float(sum(joined.count(f" {term} ") for term in terms)) / len(texts)
    digits = sum(char.isdigit() for char in joined)
    return {
        **counts,
        "snippet_words": float(len(words)) / len(texts),
        "snippet_has_year": float(any(_YEAR.search(text) for text in texts)),
        "snippet_digit_share": float(digits) / max(len(joined), 1),
    }


def _smoothed_rate(hits: float, total: float, base: float) -> float:
    """Additive smoothing toward the category base rate.

    A brand seen twice in training should not get a prior of 1.0 because both
    of those responses happened to pick it.
    """
    return (hits + PRIOR_SMOOTHING * base) / (total + PRIOR_SMOOTHING)


def add_priors(pairs: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Attach brand priors fitted on the training rows only.

    ``*_off`` priors use the retrieval-off condition, which is the corpus's
    observed recommendation frequency without retrieval, not a direct measure
    of internal beliefs or recognition. ``*_all`` priors pool both conditions.
    """
    train = select(pairs, train_mask)
    out = pairs.copy()

    off = select(train, train["condition"] == "search_off")
    for suffix, subset in (("off", off), ("all", train)):
        decided = select(subset, subset["response_decided"] == 1)
        for target, source in (("top", decided), ("mention", subset)):
            column = f"prior_{target}_{suffix}"
            label = "y_top" if target == "top" else "y_mention"
            if len(source) == 0:
                out[column] = 0.0
                continue
            base_by_category = {
                str(category): float(value)
                for category, value in source.groupby("category")[label].mean().items()
            }
            grouped = (
                source.groupby(["category", "brand"])[label]
                .agg(hits="sum", total="size")
                .reset_index()
            )
            rates = {
                (str(category), str(brand)): _smoothed_rate(
                    float(hits), float(total), base_by_category.get(str(category), 0.0)
                )
                for category, brand, hits, total in zip(
                    grouped["category"],
                    grouped["brand"],
                    grouped["hits"],
                    grouped["total"],
                    strict=True,
                )
            }
            values = [
                rates.get((str(category), str(brand)), base_by_category.get(str(category), 0.0))
                for category, brand in zip(out["category"], out["brand"], strict=True)
            ]
            out[column] = np.asarray(values, dtype=float)
    return out


def finalise(pairs: pd.DataFrame) -> pd.DataFrame:
    """Derive the remaining numeric columns and check no answer-derived leak."""
    out = pairs.copy()
    out["best_position_missing"] = (out["best_position"] == 0).astype(int)
    # A missing position must not read as "rank 0, the very best".
    out["best_position"] = out["best_position"].replace(0, 99).astype(float)
    forbidden = {
        "response_length_words",
        "number_of_brands_mentioned",
        "hedging_level",
        "answer_mode",
        "evidence_style",
        "confidence",
        "first_mentioned_brand",
    }
    present = forbidden & set(NUMERIC_FEATURES) | forbidden & set(CATEGORICAL_FEATURES)
    if present:
        raise ValueError(f"Answer-derived columns leaked into the feature set: {sorted(present)}")
    return out
