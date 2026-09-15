"""Explicitly invoked listwise training; no downloads or CPU fallback by default."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modeling.brands import load_registry
from modeling.cross_encoder import DEFAULT_MODELS
from modeling.features import select

from .evidence import mask_text
from .io import digest, read_json, sha256, write_json
from .metrics import summarise
from .workspace import responses, verify


def validate_groups(frame: pd.DataFrame) -> None:
    if frame.empty or frame.duplicated(["record_id", "brand"]).any():
        raise ValueError("Empty or duplicate listwise candidates")
    if not frame.groupby("record_id").y_top.sum().eq(1).all():
        raise ValueError("Every listwise response must have exactly one winner")
    if not frame.groupby("record_id").query_id.nunique().eq(1).all():
        raise ValueError("One response must map to one query")


def reference_loss(logits: list[float], winner: int) -> float:
    """Stable NumPy oracle for the response-level softmax objective."""
    values = np.asarray(logits, dtype=float)
    if len(values) < 2 or not 0 <= winner < len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid listwise logits/winner")
    values -= values.max()
    return float(np.log(np.exp(values).sum()) - values[winner])


def inputs(root: Path, track: str, category: str, masked: bool) -> tuple[pd.DataFrame, dict]:
    pairs = pd.read_parquet(root / f"pairs_{track}.parquet")
    eligible = select(pairs, (pairs.category == category) & (pairs.condition == "search_on"))
    selected = select(
        eligible, (eligible.response_decided == 1) & (eligible.confidence != "low")
    ).copy()
    validate_groups(selected)
    frame = responses(root, track).set_index("record_id")
    evidence = pd.read_parquet(root / f"evidence_{track}.parquet")
    groups = {key: part for key, part in evidence.groupby(["record_id", "brand"])}
    texts = []
    registry = load_registry(category)
    for row in selected.to_dict("records"):
        record = frame.loc[row["record_id"]]
        sources = groups.get((row["record_id"], row["brand"]))
        snippet = (
            ""
            if sources is None
            else "\n".join(f"{s['title']}\n{s['snippet']}" for s in sources.to_dict("records"))
        )
        text = f"Model: {row['model_id']}\nQuestion: {record['query_text']}\nCandidate: {row['brand']}\n{snippet}"
        texts.append(mask_text(text, row["brand"], registry) if masked else text)
    selected["text"] = texts
    coverage = {
        "search_on_responses": eligible.record_id.nunique(),
        "included_responses": selected.record_id.nunique(),
        "excluded_responses": eligible.record_id.nunique() - selected.record_id.nunique(),
        "candidate_policy": "full_registry",
        "target": "single_winner",
        "interpretation": "surrogate sensitivity, not causal brand/content shares",
    }
    return selected.reset_index(drop=True), coverage


def runtime() -> tuple[Any, Any, Any]:
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise ValueError(
            "Install the optional m3 extra first; default checks do not need torch"
        ) from exc
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Long training will not fall back to CPU")
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def train(root: Path, args: argparse.Namespace) -> dict:
    manifest = verify(root)
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision or ""):
        raise ValueError("Supply --model-revision with the exact 40-character encoder commit")
    if args.epochs < 1 or args.max_length < 16 or args.candidate_batch < 1:
        raise ValueError("Invalid training budget")
    torch, tokenizer_class, model_class = runtime()
    encoder = DEFAULT_MODELS[args.track]
    if args.allow_download:
        print("Explicit encoder download enabled; no generation API is used", flush=True)
    tokenizer = tokenizer_class.from_pretrained(
        encoder, revision=args.model_revision, local_files_only=not args.allow_download
    )
    frames = {
        variant: inputs(root, args.track, args.category, variant == "masked")
        for variant in ("named", "masked")
    }
    registry = load_registry(args.category)
    markers = [
        "[TARGET_BRAND]",
        *[f"[OTHER_BRAND_{n}]" for n in range(1, len(registry.brands) + 1)],
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": markers})
    named = frames["named"][0]
    masked = frames["masked"][0]
    if not named[["record_id", "brand", "y_top"]].equals(masked[["record_id", "brand", "y_top"]]):
        raise ValueError("Named/masked candidate panels differ")
    config = {
        "experiment": manifest["identity"],
        "track": args.track,
        "category": args.category,
        "encoder": encoder,
        "revision": args.model_revision,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "candidate_batch": args.candidate_batch,
        "smoke": args.smoke,
        "learning_rate": 2e-5,
        "objective": "response_listwise_softmax",
    }
    output = root / "listwise" / digest(config)[:16]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    folds = read_json(root / f"folds_{args.track}.json")
    if set(named.query_id) - set(folds):
        raise ValueError("Unknown listwise query fold")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results = {}
    for variant, (frame, coverage) in frames.items():
        frame = frame.copy()
        frame["fold"] = frame.query_id.map(folds)
        completed = []
        selected_folds = (
            sorted(frame.fold.unique())[:1] if args.smoke else sorted(frame.fold.unique())
        )
        for fold in selected_folds:
            folder = output / f"{variant}_fold{fold}"
            folder.mkdir(exist_ok=True)
            score_path, state_path = folder / "scores.parquet", folder / "state.json"
            if state_path.exists() and read_json(state_path).get("status") == "completed":
                state = read_json(state_path)
                for filename, checksum in state["artifacts"].items():
                    if sha256(folder / filename) != checksum:
                        raise ValueError("Listwise checkpoint or score checksum mismatch")
                completed.append(pd.read_parquet(score_path))
                continue
            write_json(
                state_path,
                {"status": "running", "config": config, "variant": variant, "fold": int(fold)},
            )
            try:
                torch.manual_seed(args.seed)
                model = model_class.from_pretrained(
                    encoder,
                    revision=args.model_revision,
                    local_files_only=not args.allow_download,
                    num_labels=1,
                )
                model.resize_token_embeddings(len(tokenizer))
                model.gradient_checkpointing_enable()
                model.to("cuda")
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                checkpoint_path = folder / "checkpoint.pt"
                start_epoch = 0
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
                    if checkpoint["config_hash"] != digest(config):
                        raise ValueError("Checkpoint configuration changed")
                    model.load_state_dict(checkpoint["model"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    start_epoch = checkpoint["next_epoch"]
                train_frame = select(frame, frame.fold != fold)
                test_frame = select(frame, frame.fold == fold)
                train_groups = [g for _, g in train_frame.groupby("record_id", sort=True)]
                if not train_groups or test_frame.empty:
                    raise ValueError("Empty listwise training/test fold")
                if args.smoke:
                    train_groups = train_groups[:2]
                    test_frame = select(
                        test_frame, test_frame.record_id.isin(test_frame.record_id.unique()[:2])
                    )

                def logits_for(group: pd.DataFrame, active_model: Any = model) -> Any:
                    outputs = []
                    for offset in range(0, len(group), args.candidate_batch):
                        batch = tokenizer(
                            group.text.iloc[offset : offset + args.candidate_batch].tolist(),
                            padding=True,
                            truncation=True,
                            max_length=args.max_length,
                            return_tensors="pt",
                        ).to("cuda")
                        outputs.append(active_model(**batch).logits.reshape(-1))
                    return torch.cat(outputs)

                epochs = 1 if args.smoke else args.epochs
                for epoch in range(start_epoch, epochs):
                    model.train()
                    torch.manual_seed(args.seed + epoch)
                    order = list(range(len(train_groups)))
                    random.Random(args.seed + epoch).shuffle(order)
                    for i in order:
                        group = train_groups[i]
                        winner = int(np.flatnonzero(group.y_top.to_numpy())[0])
                        optimizer.zero_grad(set_to_none=True)
                        logits = logits_for(group)
                        loss = torch.nn.functional.cross_entropy(
                            logits.unsqueeze(0), torch.tensor([winner], device="cuda")
                        )
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite listwise loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    temporary = folder / "checkpoint.tmp"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "next_epoch": epoch + 1,
                            "config_hash": digest(config),
                        },
                        temporary,
                    )
                    temporary.replace(checkpoint_path)
                    print(f"{variant} fold={fold} epoch={epoch + 1}/{epochs}", flush=True)
                model.eval()
                scores = []
                with torch.no_grad():
                    for _, group in test_frame.groupby("record_id", sort=True):
                        row = group.drop(columns="text").copy()
                        row["score_M3"] = torch.softmax(logits_for(group), dim=0).cpu().numpy()
                        scores.append(row)
                scored = pd.concat(scores, ignore_index=True)
                scored.to_parquet(score_path, index=False)
                tokenizer.save_pretrained(folder / "tokenizer")
                write_json(
                    state_path,
                    {
                        "status": "completed",
                        "config": config,
                        "artifacts": {
                            "scores.parquet": sha256(score_path),
                            "checkpoint.pt": sha256(checkpoint_path),
                        },
                    },
                )
                completed.append(scored)
                del model, optimizer
                torch.cuda.empty_cache()
            except Exception as exc:
                write_json(
                    state_path,
                    {"status": "error", "error_type": type(exc).__name__, "config": config},
                )
                raise
        scored_all = pd.concat(completed, ignore_index=True)
        results[variant] = {
            "coverage": coverage,
            "smoke_only": args.smoke,
            "metrics": None if args.smoke else summarise(scored_all, "score_M3", "y_top"),
        }
    write_json(output / "results.json", results)
    return {
        "output": str(output),
        "status": "smoke_completed" if args.smoke else "completed",
        "results": results,
    }
