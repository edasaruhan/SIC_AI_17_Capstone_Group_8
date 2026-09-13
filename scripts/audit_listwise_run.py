"""Audit a completed listwise run; optionally reload one response per checkpoint.

No model downloads or provider calls. Run with PYTHONPATH=src and the m3 extra
for --reload. The report is written separately from the frozen training artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_eval.io import clean, digest, read_json, sha256, write_json
from evidence_eval.listwise import inputs, runtime, validate_groups
from evidence_eval.metrics import summarise
from evidence_eval.workspace import verify
from modeling.features import select


def check_panel(scored: pd.DataFrame, expected: pd.DataFrame, folds: dict) -> None:
    validate_groups(scored)
    keys = ["record_id", "brand"]
    columns = keys + ["query_id", "y_top"]
    actual = pd.DataFrame(scored[columns]).sort_values(keys).reset_index(drop=True)
    wanted = pd.DataFrame(expected[columns]).sort_values(keys).reset_index(drop=True)
    if not actual.equals(wanted):
        raise ValueError("Evaluation candidates, queries or labels differ from the frozen panel")
    if not scored.fold.eq(scored.query_id.map(folds)).all():
        raise ValueError("Prediction fold differs from the held-out query fold")


def check_scores(scored: pd.DataFrame, expected: pd.DataFrame, folds: dict) -> None:
    check_panel(scored, expected, folds)
    values = scored.score_M3.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid listwise probabilities")
    if not np.allclose(scored.groupby("record_id").score_M3.sum(), 1.0, atol=1e-5):
        raise ValueError("Listwise probabilities must sum to one per response")


def reload_check(folder: Path, config: dict, group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    torch, tokenizer_class, model_class = runtime()
    tokenizer = tokenizer_class.from_pretrained(folder / "tokenizer", local_files_only=True)
    model = model_class.from_pretrained(
        config["encoder"], revision=config["revision"], local_files_only=True, num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["config_hash"] != digest(config):
        raise ValueError("Checkpoint configuration mismatch")
    if checkpoint["next_epoch"] != (1 if config["smoke"] else config["epochs"]):
        raise ValueError("Checkpoint has not completed all epochs")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to("cuda").eval()
    logits = []
    with torch.no_grad():
        for offset in range(0, len(group), config["candidate_batch"]):
            batch = tokenizer(
                group.text.iloc[offset : offset + config["candidate_batch"]].tolist(),
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            ).to("cuda")
            logits.append(model(**batch).logits.reshape(-1))
        actual = torch.softmax(torch.cat(logits), dim=0).cpu().numpy()
    expected = scored.set_index("brand").loc[group.brand, "score_M3"].to_numpy()
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-6):
        raise ValueError("Reloaded checkpoint does not reproduce saved predictions")
    result = {
        "record_id": str(group.record_id.iloc[0]),
        "candidates": len(group),
        "max_absolute_error": float(np.abs(actual - expected).max()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    del model, logits
    torch.cuda.empty_cache()
    return result


def audit(run: Path, reload: bool = False) -> dict:
    root = run.parent.parent
    manifest = verify(root)
    config = read_json(run / "config.json")
    if config["experiment"] != manifest["identity"] or run.name != digest(config)[:16]:
        raise ValueError("Run identity does not match its experiment/configuration")
    if config["smoke"]:
        raise ValueError("Audit expects a full run, not smoke-test metrics")
    folds = read_json(root / f"folds_{config['track']}.json")
    baseline_dir = root / "baselines"
    baseline_state = read_json(baseline_dir / f"{config['track']}_y_top.json")
    if (
        baseline_state["status"] != "completed"
        or baseline_state["identity"]["experiment"] != manifest["identity"]
        or baseline_state["identity"]["target"] != "y_top"
        or baseline_state["identity"]["track"] != config["track"]
    ):
        raise ValueError("Baseline comparison requires completed baseline results")
    for name, checksum in baseline_state["artifacts"].items():
        if sha256(baseline_dir / name) != checksum:
            raise ValueError("Baseline artifact checksum mismatch")
    baseline = pd.read_parquet(baseline_dir / f"scores_{config['track']}_y_top.parquet")
    result = {
        "config": config,
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "audit_script_sha256": sha256(Path(__file__)),
        "package_versions": (
            {name: version(name) for name in ("numpy", "pandas", "torch", "transformers")}
            if reload
            else {name: version(name) for name in ("numpy", "pandas")}
        ),
        "checkpoint_reload_scope": "one_response_per_fold" if reload else "not_requested",
        "comparison_scope": (
            "Identical held-out responses/candidates; training scopes differ: "
            "M0-M2 use both conditions/all track categories; M3 uses one category/search_on. "
            "This is not a controlled architecture-only comparison or a causal estimate."
        ),
        "variants": {},
    }
    baseline_panel = None
    for variant in ("named", "masked"):
        frame, coverage = inputs(root, config["track"], config["category"], variant == "masked")
        parts, checks = [], []
        for fold in sorted({folds[q] for q in frame.query_id}):
            folder = run / f"{variant}_fold{fold}"
            state = read_json(folder / "state.json")
            if state["status"] != "completed" or state["config"] != config:
                raise ValueError("Incomplete or incompatible fold")
            for name, checksum in state["artifacts"].items():
                if sha256(folder / name) != checksum:
                    raise ValueError(f"Checkpoint/score checksum mismatch: {folder / name}")
            part = pd.read_parquet(folder / "scores.parquet")
            if not part.fold.eq(fold).all():
                raise ValueError("Scores saved in the wrong fold directory")
            if reload:
                record_id = sorted(part.record_id.unique())[0]
                checks.append(
                    reload_check(
                        folder,
                        config,
                        select(frame, frame.record_id == record_id),
                        select(part, part.record_id == record_id),
                    )
                )
            parts.append(part)
        scored = pd.concat(parts, ignore_index=True)
        check_scores(scored, frame, folds)
        baseline_panel = select(baseline, baseline.record_id.isin(scored.record_id)).reset_index(
            drop=True
        )
        check_panel(baseline_panel, frame, folds)
        result["variants"][variant] = {
            "coverage": coverage,
            "metrics": summarise(scored, "score_M3", "y_top"),
            "reloaded_checkpoints": checks,
        }
    assert baseline_panel is not None
    result["baselines_on_same_panel"] = {
        name: summarise(baseline_panel, "score_" + name, "y_top")
        for name in ("naive_frequency", "naive_position", "M0", "M1", "M2")
    }
    return clean(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    report = audit(args.run, args.reload)
    write_json(args.run / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
