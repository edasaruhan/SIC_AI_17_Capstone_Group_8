"""M3: a cross-encoder over retrieved text, and the masking ablation.

The ablation is the reason this model exists. M3 is trained twice on identical
rows and identical folds:

* **named** -- the candidate and the retrieved snippets keep their brand names.
* **masked** -- every registry surface form in both is replaced by ``[BRAND]``.

The gap between the two is how much of the decision rides on recognising a name
rather than on what the content says. A feature-only model cannot perform that
intervention, because masking is an edit to the text itself.

Only retrieval-on responses are used: with retrieval off there is no snippet to
read, so masking would remove the entire input rather than isolating one part of
it. The ablation therefore speaks about retrieval-on answers -- which is also
where content is actionable, since that is the content an assistant can be shown.

Model sizes are chosen for a 4 GB laptop GPU. A 250k-vocabulary multilingual
encoder does not fit once Adam states are allocated, so each track uses a
base-size monolingual BERT; that also keeps the two tracks architecturally
comparable rather than mixing model families.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_MODELS = {
    "en": "bert-base-uncased",
    "tr": "dbmdz/bert-base-turkish-cased",
}
MASK_TOKEN = "[BRAND]"


@dataclass
class EncoderConfig:
    """Training budget for one cross-encoder run."""

    model_name: str
    # Half the pairs are a bare "Brand : " with no retrieved snippet (29 tokens at
    # the median) while the tail runs past 1600. Raising the cap from 128 to 192
    # rescues only six percent more rows from truncation and costs half again as
    # much compute, so 128 is where the trade sits.
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 2e-5
    negatives_per_response: int = 7
    seed: int = 42
    fp16: bool = True
    extra: dict = field(default_factory=dict)


def build_inputs(
    pair_rows: pd.DataFrame,
    snippets: pd.DataFrame,
    *,
    masked: bool,
) -> pd.DataFrame:
    """Attach the encoder's text fields to scored pairs.

    ``snippets`` comes from ``pairs.brand_snippets(frame, mask=masked)`` so the
    named and masked variants share every row, label and fold -- only the text
    differs.
    """
    merged = pair_rows.merge(snippets, on=["record_id", "brand"], how="left", validate="1:1")
    unmatched = merged["query_text"].isna()
    if unmatched.any():
        # A scored pair with no snippet row means the two sides disagree about the
        # candidate universe, which would silently feed NaN text to the tokenizer.
        missing = merged.loc[unmatched, ["record_id", "brand"]].head(5).to_dict("records")
        raise ValueError(
            f"{int(unmatched.sum())} scored pairs have no snippet row, e.g. {missing}. "
            "Build snippets from the same frame the pairs were built from."
        )
    merged["snippet_text"] = merged["snippet_text"].fillna("").astype(str)
    label = MASK_TOKEN if masked else merged["brand"].astype(str)
    merged["text_a"] = merged["query_text"].astype(str)
    merged["text_b"] = label + " : " + merged["snippet_text"]
    return merged


def sample_negatives(
    frame: pd.DataFrame,
    target: str,
    *,
    negatives_per_response: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep every positive and a bounded random sample of negatives.

    Scoring stays over the full candidate set; only training is subsampled, so
    the metrics remain comparable with M0-M2. Sampling seven negatives per
    positive restores roughly the one-in-eight balance the concept note assumed.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in frame.groupby("record_id", sort=False):
        positives = group[group[target] == 1]
        negatives = group[group[target] == 0]
        if len(negatives) > negatives_per_response:
            chosen = rng.choice(len(negatives), size=negatives_per_response, replace=False)
            negatives = negatives.iloc[chosen]
        keep.append(pd.concat([positives, negatives]))
    return pd.concat(keep, ignore_index=True)


def length_grouped_batches(
    lengths: list[int],
    batch_size: int,
    *,
    rng: np.random.Generator | None = None,
    megabatch_factor: int = 50,
) -> list[list[int]]:
    """Batch indices so that rows of similar length travel together.

    Every row in a batch is padded to the batch's longest row. Half of these
    pairs are a bare "Brand : " and a minority run to hundreds of tokens, so with
    randomly mixed batches a single long row stretches all sixteen -- at batch 16
    the chance a batch contains one is about 99%, and nearly all the compute goes
    into padding.

    Shuffling within a large megabatch and sorting only inside it keeps the
    training order effectively random while collapsing the padding.
    """
    order = np.arange(len(lengths))
    if rng is not None:
        rng.shuffle(order)
    span = batch_size * megabatch_factor
    batches: list[list[int]] = []
    for start in range(0, len(order), span):
        window = order[start : start + span]
        window = window[np.argsort([lengths[index] for index in window], kind="stable")]
        batches.extend(
            window[offset : offset + batch_size].tolist()
            for offset in range(0, len(window), batch_size)
        )
    if rng is not None:
        rng.shuffle(batches)
    return batches


def run_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    config: EncoderConfig,
) -> np.ndarray:
    """Fine-tune on one fold's training rows and score its held-out rows.

    Both halves tokenize once up front and batch by length; scores are returned
    in the caller's row order, not the order the batches ran in.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = config.fp16 and device == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_tokens([MASK_TOKEN], special_tokens=True)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    def encode(frame: pd.DataFrame) -> list[list[int]]:
        return tokenizer(
            frame["text_a"].tolist(),
            frame["text_b"].tolist(),
            truncation=True,
            max_length=config.max_length,
        )["input_ids"]

    def as_batch(ids: list[list[int]]) -> dict:
        width = max(len(item) for item in ids)
        pad = tokenizer.pad_token_id or 0
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        attention = torch.zeros((len(ids), width), dtype=torch.long)
        for row, item in enumerate(ids):
            input_ids[row, : len(item)] = torch.tensor(item, dtype=torch.long)
            attention[row, : len(item)] = 1
        return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device)}

    train_ids = encode(train)
    train_labels = train[target].to_numpy()
    train_lengths = [len(item) for item in train_ids]
    rng = np.random.default_rng(config.seed)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for _ in range(config.epochs):
        for indices in length_grouped_batches(train_lengths, config.batch_size, rng=rng):
            batch = as_batch([train_ids[index] for index in indices])
            batch["labels"] = torch.tensor(train_labels[indices], dtype=torch.long, device=device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

    model.eval()
    test_ids = encode(test)
    test_lengths = [len(item) for item in test_ids]
    scores = np.zeros(len(test_ids), dtype=float)
    with torch.no_grad():
        for indices in length_grouped_batches(test_lengths, config.eval_batch_size):
            batch = as_batch([test_ids[index] for index in indices])
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**batch).logits
            # Scatter back to the caller's order; the batches ran sorted by length.
            scores[indices] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def run_variant(
    frame: pd.DataFrame,
    fold_by_query: dict[str, int],
    target: str,
    config: EncoderConfig,
) -> pd.DataFrame:
    """Out-of-fold scores for one masking variant, on the M0-M2 folds."""
    frame = frame.copy()
    frame["fold"] = frame["query_id"].map(fold_by_query)
    frame["score_M3"] = np.nan
    for fold in sorted(frame["fold"].unique()):
        train_rows = frame[frame["fold"] != fold]
        test_rows = frame[frame["fold"] == fold]
        sampled = sample_negatives(
            train_rows,
            target,
            negatives_per_response=config.negatives_per_response,
            seed=config.seed,
        )
        if sampled[target].nunique() < 2:
            frame.loc[test_rows.index, "score_M3"] = float(train_rows[target].mean())
            continue
        frame.loc[test_rows.index, "score_M3"] = run_fold(sampled, test_rows, target, config)
        print(f"    fold {fold}: trained on {len(sampled)}, scored {len(test_rows)}", flush=True)
    return frame
