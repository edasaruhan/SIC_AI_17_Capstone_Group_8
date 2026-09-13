"""One way to run the advisor, shared by the command line and the interface.

Both front ends must agree on everything that costs money or decides a result: the
run identity (which receipt folder a run reads and writes), the call estimate the user
confirms, the budget, and what gets saved. Keeping that in one module means the
interface cannot quietly differ from the CLI the tests and the documentation describe.

``stream`` yields the state after every graph node, so an interface can show progress
node by node. LangGraph's "updates" stream reports only what each node added; the
running state is rebuilt here with the same reducers the graph uses, read from the
state schema itself rather than repeated by hand.
"""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from brand_demo.workflow import Receipts
from evidence_eval.io import digest, write_json, write_text

from . import candidates, features, nodes
from . import state as schema
from .clients import AdvisorClient

OUTPUT = Path("data/processed/advisor")
# What a person watching the run should read for each node.
STEP_LABELS = {
    "plan": "Sorular hazırlanıyor",
    "search": "Google'da aranıyor",
    "discover": "Rakip markalar bulunuyor",
    "interrogate": "Yapay zekâya soruluyor",
    "analyse": "Ölçülüyor, öğrenilmiş sinyal uygulanıyor, teşhis konuyor",
    "advise_absent": "Öneriler hazırlanıyor",
    "advise_low_rank": "Öneriler hazırlanıyor",
    "advise_ceiling": "Öneriler hazırlanıyor",
    "advise_leader": "Öneriler hazırlanıyor",
    "advise_thin": "Öneriler hazırlanıyor",
    "report": "Rapor yazılıyor",
}
RECORD_KEYS = (
    "brand",
    "sector",
    "language",
    "curated",
    "queries",
    "candidates",
    "diagnosis",
    "measures",
    "scores",
    "signals",
    "notes",
)


@dataclass(frozen=True)
class Options:
    """Everything that defines one advisor run."""

    brand: str
    sector: str
    language: str = "tr"
    brand_aliases: tuple[str, ...] = ()
    add_rivals: tuple[str, ...] = ()
    drop_rivals: tuple[str, ...] = ()
    queries: int = 3
    reps: int = 2
    max_calls: int = 20
    concurrency: int = 2
    retry_failed: bool = False
    output: Path = OUTPUT


def run_id(options: Options) -> str:
    """The receipt folder. Rival corrections are deliberately not part of it.

    Corrections change which names are matched, not any paid payload, so a corrected
    rerun reads every call from the cache. Every setting that shapes a payload is part
    of it, or a changed setting would collide with receipts written under the old one.
    """
    return digest(
        {
            "brand": options.brand,
            "aliases": sorted(options.brand_aliases),
            "sector": options.sector,
            "language": options.language,
            "queries": options.queries,
            "reps": options.reps,
            "model": nodes.MODEL,
            "temperature": nodes.TEMPERATURE,
            "max_tokens": nodes.MAX_TOKENS,
            "prompts": digest(
                [nodes.QUERY_PROMPT, nodes.SYSTEM, nodes.SYSTEM_OFFLINE, candidates.EXTRACT_PROMPT]
            ),
        }
    )[:16]


def run_folder(options: Options) -> Path:
    return options.output / run_id(options)


def estimated_calls(options: Options) -> int:
    """Query generation (only without recorded queries) + searches + extraction + answers."""
    recorded = nodes.recorded_queries(options.sector, options.language, options.queries)
    generation = 0 if recorded else 1
    return generation + options.queries * (1 + 2 * options.reps) + 1


_EXTEND_KEYS = frozenset(
    key
    for key, hint in typing.get_type_hints(schema.AdvisorState, include_extras=True).items()
    if schema.extend in getattr(hint, "__metadata__", ())
)


def merge(current: dict, update: dict) -> dict:
    """Apply one node's update the way the graph does: reducer keys append, others replace."""
    merged = dict(current)
    for key, value in update.items():
        merged[key] = [*merged.get(key, []), *value] if key in _EXTEND_KEYS else value
    return merged


async def stream(options: Options, boosters: dict) -> AsyncIterator[tuple[str, dict]]:
    """Run the graph, yielding (node, state so far) after every node."""
    from .graph import build

    folder = run_folder(options)
    folder.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as http:
        runtime = nodes.Runtime(
            receipts=Receipts(folder, AdvisorClient(http), options.retry_failed),
            boosters=boosters,
            rules=features.load_rules(),
            brand_aliases=list(options.brand_aliases),
            add_rivals=list(options.add_rivals),
            drop_rivals=list(options.drop_rivals),
            reps=options.reps,
            n_queries=options.queries,
            concurrency=options.concurrency,
        )
        graph = build(runtime, options.max_calls)
        state: dict = {
            "brand": options.brand,
            "sector": options.sector,
            "language": options.language,
            "max_calls": options.max_calls,
        }
        async for chunk in graph.astream(dict(state), stream_mode="updates"):
            for node, update in chunk.items():
                state = merge(state, update or {})
                yield node, state


def save(options: Options, final: dict) -> Path:
    """Write the report and a machine-readable record beside the receipts."""
    folder = run_folder(options)
    folder.mkdir(parents=True, exist_ok=True)
    write_text(folder / "report.md", final.get("report", ""))
    write_json(
        folder / "run.json",
        {key: final.get(key) for key in RECORD_KEYS}
        | {
            "model": nodes.MODEL,
            "corrections": {"add": list(options.add_rivals), "drop": list(options.drop_rivals)},
        },
    )
    return folder


async def run(options: Options, boosters: dict) -> dict:
    """Run to completion and save; what the CLI calls."""
    final: dict = {}
    async for _, state in stream(options, boosters):
        final = state
    save(options, final)
    return final
