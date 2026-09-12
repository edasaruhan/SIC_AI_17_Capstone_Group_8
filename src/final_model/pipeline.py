"""Train and reload the selected experimental release, without network access."""

from __future__ import annotations

import fcntl
import gc
import importlib.metadata
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier
from loguru import logger

from evidence_eval.baselines import BLOCKS, ESTIMATOR, design, priors
from evidence_eval.evidence import mask_text, source_type, spans
from evidence_eval.io import digest, read_json, sha256, write_json
from evidence_eval.listwise import inputs, runtime
from evidence_eval.workspace import verify
from modeling.brands import load_registry
from modeling.features import CATEGORICAL_FEATURES, PRIOR_FEATURES, finalise, select

SELECTION = {
    "tr_vpn": {"run": "9dd1e55ee84de53b", "variant": "masked", "family": "M3"},
    "tr_cosmetics": {"run": "1c5aadf2b8963237", "variant": "M1", "family": "M1"},
    "en_vpn": {"run": "ff0c305a3479fcad", "variant": "named", "family": "M3"},
}
KEYS = ["category", "model_id", "brand"]
LIMITATIONS = [
    "Experimental full-data refit; no independent final-model test set.",
    "Selection used historical held-out PR-AUC; selection and evaluation are not independent.",
    "TR top-1 results did not beat the frequency baseline; keep that comparator.",
    "Labels come from the existing dataset/judge, not AI review CSV or human validation.",
    "Scores predict historical single-winner preferences, not causal visibility uplift.",
    "Only recorded snippets are inputs, not full websites or hidden model reasoning.",
    "Only frozen brands, generator IDs and search_on inference are supported.",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def baseline_rows(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "pairs_tr.parquet")
    frame = select(frame, frame.confidence != "low").copy().reset_index(drop=True)
    folds = read_json(root / "folds_tr.json")
    if set(frame.query_id) - set(folds):
        raise ValueError("Unknown query fold")
    frame["fold"] = frame.query_id.map(folds)
    return frame


def oof_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Every training prior excludes its own query fold, including mention labels."""
    if frame.fold.nunique() < 2 or frame.fold.isna().any():
        raise ValueError("At least two known folds are required")
    return finalise(
        pd.concat(
            [
                priors(select(frame, frame.fold != f), select(frame, frame.fold == f))
                for f in sorted(frame.fold.unique())
            ]
        ).sort_index()
    )


def plan(root: Path) -> dict:
    frozen = verify(root)
    models = {}
    for name, choice in SELECTION.items():
        folder = root / "listwise" / choice["run"]
        audit = read_json(folder / "audit.json")
        config = read_json(folder / "config.json")
        if audit["config"] != config or config["experiment"] != frozen["identity"]:
            raise ValueError("Selection provenance mismatch")
        if config["smoke"]:
            raise ValueError("Cannot select smoke training")
        metric = (
            audit["variants"][choice["variant"]]["metrics"]
            if choice["family"] == "M3"
            else audit["baselines_on_same_panel"]["M1"]
        )
        if choice["family"] == "M3":
            frame, coverage = inputs(
                root, config["track"], config["category"], choice["variant"] == "masked"
            )
        else:
            frame = baseline_rows(root)
            frame = select(frame, frame.response_decided == 1)
            coverage = {
                "training_scope": "all TR categories and both conditions, as in historical M1"
            }
        models[name] = {
            **choice,
            "config": config,
            "historical_cv_metrics_not_final_test": metric,
            "historical_frequency_metrics": audit["baselines_on_same_panel"]["naive_frequency"],
            "audit_sha256": sha256(folder / "audit.json"),
            "coverage": {
                **coverage,
                "training_responses": frame.record_id.nunique(),
                "training_pairs": len(frame),
            },
            "brands": load_registry(config["category"]).brands,
            "generator_ids": sorted(frame.model_id.unique()),
        }
    identity = {
        "version": 1,
        "experiment": frozen["identity"],
        "selection_criterion": "historical held-out PR-AUC, exploratory selection",
        "implementation": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "lightgbm", "numpy", "pandas")
        },
        "models": models,
        "limitations": LIMITATIONS,
    }
    return {"identity": identity, "release_id": digest(identity)[:16]}


def check_completed(folder: Path, identity: str) -> dict | None:
    path = folder / "state.json"
    if not path.exists():
        return None
    state = read_json(path)
    if state["identity"] != identity:
        raise ValueError("Refusing incompatible final-model state")
    if state["status"] != "completed":
        return None
    if not state.get("artifacts"):
        raise ValueError("Completed model has no artifacts")
    for name, checksum in state["artifacts"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Final artifact checksum mismatch: {name}")
    return state


def complete(folder: Path, identity: str, details: dict) -> dict:
    artifacts = {
        str(p.relative_to(folder)): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name not in {"state.json", "checkpoint.pt"}
    }
    state = {
        "status": "completed",
        "identity": identity,
        "finished_at": now(),
        "artifacts": artifacts,
        **details,
    }
    write_json(folder / "state.json", state)
    return state


def attach_priors(panel: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    panel = panel.drop(columns=PRIOR_FEATURES, errors="ignore")
    result = panel.merge(lookup, on=KEYS, how="left", validate="many_to_one", sort=False)
    if result.reindex(columns=PRIOR_FEATURES).isna().to_numpy().any():
        raise ValueError(
            "Unknown brand/generator/domain prior; collect and evaluate before extending"
        )
    return finalise(result)


def baseline_design(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = pd.DataFrame({c: frame[c].astype(float) for c in BLOCKS["M1"]})
    for name, categories in metadata["categories"].items():
        if set(frame[name]) - set(categories):
            raise ValueError(f"Unsupported categorical value: {name}")
        out[name] = pd.Categorical(frame[name], categories=categories)
    return out


def train_baseline(root: Path, folder: Path, identity: str) -> dict:
    raw = baseline_rows(root)
    frame = oof_priors(raw)
    frame = select(frame, frame.response_decided == 1)
    x_train, _ = design(frame, frame, BLOCKS["M1"])
    model = LGBMClassifier(**ESTIMATOR).fit(x_train, frame.y_top.to_numpy())
    model.booster_.save_model(str(folder / "model.txt"))
    destination = raw.reindex(columns=KEYS).drop_duplicates().reset_index(drop=True)
    lookup = priors(raw, destination).reindex(columns=KEYS + PRIOR_FEATURES)
    lookup.to_parquet(folder / "priors.parquet", index=False)
    metadata = {
        "categories": {c: list(x_train[c].cat.categories) for c in CATEGORICAL_FEATURES},
        "features": list(x_train.columns),
        "estimator": ESTIMATOR,
        "prior_training": "out_of_query_fold",
        "prior_inference": "all_historical_non_low_confidence_TR",
    }
    write_json(folder / "preprocessing.json", metadata)
    sample = x_train.iloc[:148]
    expected = np.asarray(model.predict_proba(sample))[:, 1]
    actual = np.asarray(Booster(model_file=str(folder / "model.txt")).predict(sample))
    if not np.allclose(expected, actual, atol=1e-12, rtol=0):
        raise ValueError("LightGBM reload failed")
    return complete(
        folder,
        identity,
        {
            "reload_max_absolute_error": float(np.max(np.abs(actual - expected))),
            "training_responses": frame.record_id.nunique(),
            "training_pairs": len(frame),
        },
    )


def logits_for(
    torch: Any, tokenizer: Any, model: Any, texts: list[str], config: dict, device: str
) -> Any:
    outputs = []
    for offset in range(0, len(texts), config["candidate_batch"]):
        batch = tokenizer(
            texts[offset : offset + config["candidate_batch"]],
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        ).to(device)
        outputs.append(model(**batch).logits.reshape(-1))
    return torch.cat(outputs)


def train_transformer(root: Path, folder: Path, identity: str, spec: dict) -> dict:
    torch, tokenizer_class, model_class = runtime()
    config = spec["config"]
    frame, coverage = inputs(root, config["track"], config["category"], spec["variant"] == "masked")
    cache = str(root / "hf_cache" / "hub")
    tokenizer = tokenizer_class.from_pretrained(
        config["encoder"], revision=config["revision"], cache_dir=cache, local_files_only=True
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[TARGET_BRAND]",
                *[f"[OTHER_BRAND_{n}]" for n in range(1, len(spec["brands"]) + 1)],
            ]
        }
    )
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = model_class.from_pretrained(
        config["encoder"],
        revision=config["revision"],
        cache_dir=cache,
        local_files_only=True,
        num_labels=1,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    checkpoint_path = folder / "checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["next_epoch"]
        del saved
    groups = [part for _, part in frame.groupby("record_id", sort=True)]
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        torch.manual_seed(config["seed"] + epoch)
        order = list(range(len(groups)))
        random.Random(config["seed"] + epoch).shuffle(order)
        losses = []
        for step, index in enumerate(order, 1):
            group = groups[index]
            optimizer.zero_grad(set_to_none=True)
            logits = logits_for(torch, tokenizer, model, group.text.tolist(), config, "cuda")
            winner = int(np.flatnonzero(group.y_top.to_numpy())[0])
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([winner], device="cuda")
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 25 == 0 or step == len(order):
                state = {
                    "status": "running",
                    "identity": identity,
                    "updated_at": now(),
                    "epoch": epoch + 1,
                    "epochs": config["epochs"],
                    "step": step,
                    "steps": len(order),
                    "mean_training_loss_not_test_metric": float(np.mean(losses)),
                }
                write_json(folder / "state.json", state)
                logger.info(
                    "{} epoch={}/{} response={}/{} train_loss={:.4f}",
                    folder.name,
                    epoch + 1,
                    config["epochs"],
                    step,
                    len(order),
                    np.mean(losses),
                )
            del loss, logits
        temporary = folder / "checkpoint.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "identity": identity,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
    model.eval()
    probe = groups[0].text.tolist()
    with torch.no_grad():
        expected = logits_for(torch, tokenizer, model, probe, config, "cuda").cpu().numpy()
    model.save_pretrained(folder / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(folder / "encoder")
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded = (
        model_class.from_pretrained(folder / "encoder", local_files_only=True).to("cuda").eval()
    )
    loaded_tokenizer = tokenizer_class.from_pretrained(folder / "encoder", local_files_only=True)
    with torch.no_grad():
        actual = logits_for(torch, loaded_tokenizer, loaded, probe, config, "cuda").cpu().numpy()
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise ValueError("Transformer/tokenizer reload prediction mismatch")
    write_json(
        folder / "reload_check.json",
        {
            "record_id": str(groups[0].record_id.iloc[0]),
            "expected_logits": expected.tolist(),
            "reloaded_logits": actual.tolist(),
            "scope": "one training response, technical check only",
        },
    )
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    return complete(
        folder,
        identity,
        {
            "coverage": coverage,
            "reload_max_absolute_error": float(np.max(np.abs(actual - expected))),
            "training_responses": len(groups),
            "training_pairs": len(frame),
        },
    )


def train(root: Path, output: Path) -> dict:
    release = plan(root)
    folder = output / release["release_id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This release is already being trained") from exc
        log_id = logger.add(folder / "training.log", rotation="10 MB")
        try:
            write_json(folder / "manifest.json", {**release, "status": "training"})
            for name in ("tr_cosmetics", "tr_vpn", "en_vpn"):
                target = folder / name
                target.mkdir(exist_ok=True)
                if check_completed(target, release["release_id"]):
                    logger.info("{} already completed and verified; skipping", name)
                    continue
                logger.info("Training {} on all eligible data; no API calls", name)
                write_json(
                    target / "state.json",
                    {"status": "running", "identity": release["release_id"], "started_at": now()},
                )
                try:
                    if name == "tr_cosmetics":
                        train_baseline(root, target, release["release_id"])
                    else:
                        train_transformer(
                            root, target, release["release_id"], release["identity"]["models"][name]
                        )
                except Exception as exc:
                    write_json(
                        target / "state.json",
                        {
                            "status": "error",
                            "identity": release["release_id"],
                            "error_type": type(exc).__name__,
                            "updated_at": now(),
                        },
                    )
                    logger.exception("Final training failed for {}", name)
                    raise
                logger.info("{} saved and reloaded successfully", name)
            manifest = {**release, "status": "completed", "finished_at": now()}
            write_json(folder / "manifest.json", manifest)
            return manifest
        finally:
            logger.remove(log_id)


def status(output: Path) -> dict:
    result = {}
    for manifest_path in sorted(output.glob("*/manifest.json")):
        release = read_json(manifest_path)
        states = {}
        for name in release["identity"]["models"]:
            path = manifest_path.parent / name / "state.json"
            states[name] = read_json(path) if path.exists() else {"status": "not_started"}
            if states[name]["status"] == "completed":
                check_completed(path.parent, release["release_id"])
            states[name].pop("artifacts", None)
        result[release["release_id"]] = {"manifest_status": release["status"], "models": states}
    return result


def prepare_request(request: dict, spec: dict) -> pd.DataFrame:
    """Answer-free inference preprocessing, identical to training for supported inputs."""
    config = spec["config"]
    for key, expected in (
        ("language", config["track"]),
        ("category", config["category"]),
        ("condition", "search_on"),
    ):
        if request.get(key) != expected:
            raise ValueError(f"Unsupported {key}")
    if request.get("model_id") not in spec["generator_ids"]:
        raise ValueError("Unsupported generator model_id")
    if not isinstance(request.get("query_text"), str) or not request["query_text"].strip():
        raise ValueError("Non-empty query_text required")
    sources = request.get("sources")
    if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
        raise ValueError("sources must be a list of recorded search-result objects")
    registry = load_registry(config["category"])
    if registry.brands != spec["brands"]:
        raise ValueError("Frozen brand registry changed")
    rules = read_json(Path("configs/modeling/source_domains.json"))
    rows = []
    for brand in registry.brands:
        hits = [
            s
            for s in sources
            if brand
            in {
                b
                for _, _, b in spans(
                    str(s.get("title") or "") + "\n" + str(s.get("snippet") or ""), registry
                )
            }
        ]
        snippets = "\n".join(f"{s.get('title') or ''}\n{s.get('snippet') or ''}" for s in hits)
        text = f"Model: {request['model_id']}\nQuestion: {request['query_text']}\nCandidate: {brand}\n{snippets}"
        kinds = [source_type(str(s.get("link") or ""), brand, rules) for s in hits]
        positions = [
            int(s["position"])
            for s in hits
            if s.get("position") is not None and pd.notna(s["position"]) and int(s["position"]) > 0
        ]
        rows.append(
            {
                "brand": brand,
                "category": config["category"],
                "model_id": request["model_id"],
                "condition": "search_on",
                "text": mask_text(text, brand, registry) if spec["variant"] == "masked" else text,
                "in_search_results": int(bool(hits)),
                "n_results_mentioning": len(hits),
                "best_position": min(positions, default=0),
                **{
                    f"n_{kind}": kinds.count(kind)
                    for kind in (
                        "official",
                        "editorial",
                        "affiliate",
                        "forum",
                        "retailer",
                        "unknown",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def predict(root: Path, folder: Path, request: dict, device: str = "cpu") -> dict:
    frozen = verify(root)
    manifest = read_json(folder / "manifest.json")
    if frozen["identity"] != manifest["identity"]["experiment"]:
        raise ValueError("Inference workspace mismatch")
    for path, checksum in manifest["identity"]["implementation"].items():
        if sha256(Path(path)) != checksum:
            raise ValueError("Final-model implementation changed")
    name = f"{request.get('language')}_{request.get('category')}"
    if name not in manifest["identity"]["models"]:
        raise ValueError("Unsupported language/domain route")
    spec = manifest["identity"]["models"][name]
    target = folder / name
    if not check_completed(target, manifest["release_id"]):
        raise ValueError("Selected final model is not completed")
    frame = prepare_request(request, spec)
    if spec["family"] == "M1":
        features = attach_priors(frame, pd.read_parquet(target / "priors.parquet"))
        matrix = baseline_design(features, read_json(target / "preprocessing.json"))
        scores = np.asarray(Booster(model_file=str(target / "model.txt")).predict(matrix))
        frame["frequency_comparator"] = features.prior_top_off.to_numpy()
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(target / "encoder", local_files_only=True)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                target / "encoder", local_files_only=True
            )
            .to(device)
            .eval()
        )
        with torch.no_grad():
            scores = (
                torch.softmax(
                    logits_for(
                        torch, tokenizer, model, frame.text.tolist(), spec["config"], device
                    ),
                    dim=0,
                )
                .cpu()
                .numpy()
            )
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite inference scores")
    frame["score"] = scores
    columns = ["brand", "score"] + (
        ["frequency_comparator"] if "frequency_comparator" in frame else []
    )
    return {
        "release_id": manifest["release_id"],
        "model": name,
        "ranking": frame.sort_values(["score", "brand"], ascending=[False, True])
        .reindex(columns=columns)
        .to_dict("records"),
        "score_semantics": (
            "uncalibrated binary classifier score"
            if spec["family"] == "M1"
            else "conditional softmax over the frozen candidate panel"
        ),
        "limitations": LIMITATIONS,
    }
