"""The transferable signal, trained once on every recorded sector and saved.

This is the piece the proposal was about: learn what decides visibility from the
sample data, then apply it to a sector the model has never seen. It is M2-Invariant
from ``reports/generalization/``: no brand priors, no sector, no generator model --
only retrieval signals expressed relative to the other candidates in the same answer
(share of results, rank against the leader, source mix, content percentiles).

Why this model and not another: in leave-one-domain-out testing it beat the
sector-specific model on all four unseen English sectors (e.g. VPN PR-AUC 0.566
against 0.256) and, trained on English alone, carried over to Turkish VPN. That
held-out evidence is the validation. The fit reported here is in-sample and is not.

The model is trained on the pooled English and Turkish panels without ``model_id``,
the configuration the cross-lingual transfer used, because the live assistant is not
one of the corpus generators.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier

from evidence_eval.baselines import ESTIMATOR
from evidence_eval.io import read_json, sha256, write_json
from evidence_eval.workspace import verify
from modeling import evaluate
from visibility import generalization as g

EVIDENCE_ROOT = Path("data/processed/evidence_v2")
MODEL_DIR = Path("data/processed/advisor_model")
TRACKS = ("en", "tr")
FEATURES = list(g.INVARIANT_FEATURES)
SCORE_COLUMNS = {"y_mention": "score_mention", "y_top": "score_top"}


def training_frames(root: Path = EVIDENCE_ROOT) -> dict[str, pd.DataFrame]:
    """The same panels the generalization report validated, pooled across languages."""
    frames = {}
    for target in g.TARGETS:
        parts = []
        for track in TRACKS:
            pairs = pd.read_parquet(root / f"pairs_{track}.parquet")
            per_response = pd.read_parquet(root / f"sources_{track}.parquet", columns=["record_id"])
            parts.append(
                g.add_relative(g.panel(pairs, target), per_response["record_id"].value_counts())
            )
        frames[target] = pd.concat(parts, ignore_index=True)
    return frames


def matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in the saved column order; training and scoring share it."""
    return pd.DataFrame({feature: frame[feature].astype(float) for feature in FEATURES})


def train(root: Path = EVIDENCE_ROOT, out: Path = MODEL_DIR) -> dict:
    manifest = verify(root)  # never train on altered or stale evidence
    out.mkdir(parents=True, exist_ok=True)
    training, files = {}, {}
    for target, frame in training_frames(root).items():
        labels = frame[target].to_numpy()
        model = LGBMClassifier(**ESTIMATOR)
        model.fit(matrix(frame), labels)
        path = out / f"invariant_{target}.txt"
        model.booster_.save_model(str(path))
        files[path.name] = sha256(path)
        fitted = np.asarray(model.predict_proba(matrix(frame)))[:, 1]
        training[target] = {
            "rows": int(len(frame)),
            "sectors": sorted(str(c) for c in frame["category"].unique()),
            "languages": sorted(str(c) for c in frame["language"].unique()),
            "positive_rate": float(labels.mean()),
            "in_sample_pr_auc": float(evaluate.pr_auc(labels, fitted)),
        }
    record = {
        "version": 1,
        "model": "M2-Invariant",
        "features": FEATURES,
        "estimator": ESTIMATOR,
        "with_generator_model": False,
        "evidence_identity": manifest["identity"],
        "training": training,
        "files": files,
        "validation": (
            "reports/generalization: leave-one-domain-out and English-to-Turkish transfer. "
            "in_sample_pr_auc is a fit diagnostic, not a validation."
        ),
    }
    write_json(out / "manifest.json", record)
    return record


def load(out: Path = MODEL_DIR) -> dict[str, Booster]:
    """Load the saved boosters, refusing files that no longer match their manifest."""
    path = out / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{out} altında model yok; önce `make advisor-train` çalıştırın.")
    record = read_json(path)
    if record["features"] != FEATURES:
        raise ValueError("Kayıtlı modelin özellik listesi koddakinden farklı; yeniden eğitin.")
    boosters = {}
    for target in g.TARGETS:
        file = out / f"invariant_{target}.txt"
        if sha256(file) != record["files"][file.name]:
            raise ValueError(f"{file} manifestteki hash ile uyuşmuyor; yeniden eğitin.")
        boosters[target] = Booster(model_file=str(file))
    return boosters


def score(boosters: dict[str, Booster], frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the predicted probability of being named and of being the top pick."""
    if frame.empty:
        return frame.assign(**{column: [] for column in SCORE_COLUMNS.values()})
    x = matrix(frame)
    out = frame.copy()
    for target, column in SCORE_COLUMNS.items():
        out[column] = np.asarray(boosters[target].predict(x), dtype=float)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)
    record = train(args.root, args.out)
    for target, info in record["training"].items():
        print(
            f"{target}: {info['rows']} satır, sektörler {', '.join(info['sectors'])}, "
            f"diller {', '.join(info['languages'])}"
        )
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
