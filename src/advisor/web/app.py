# pyright: reportMissingImports=false
"""HTTP API over the advisor, and the page that uses it.

The API adds no logic of its own: profiles and questions come from ``profile``, runs from
``service`` (the same run identity, estimate, budget and receipts as the CLI), the audit
from ``audit``. A run executes as a background task in this process; the page polls its
status. Paid work only starts from ``POST /api/runs``, which the page sends after the user
has seen the call estimate.

    PYTHONPATH=src uv run --with fastapi --with uvicorn --with langgraph \\
        uvicorn advisor.web.app:app --port 8600
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from advisor import assistants, audit, model, profile, service
from advisor.advise import DIAGNOSES
from advisor.render import DIAGNOSIS_NOTE
from advisor.web import pdf
from brand_demo.workflow import Receipts
from evidence_eval.io import digest

STATIC = Path(__file__).parent / "static"
PROFILES = Path("data/processed/advisor_profiles")
RESULT_KEYS = (
    "brand",
    "sector",
    "language",
    "curated",
    "queries",
    "named_queries",
    "candidates",
    "rivals",
    "diagnosis",
    "measures",
    "assistant_measures",
    "named_measures",
    "scores",
    "signals",
    "evidence",
    "recommendations",
    "description_audit",
    "report",
    "notes",
)

app = FastAPI(title="kısaliste", docs_url=None, redoc_url=None)
RUNS: dict[str, dict[str, Any]] = {}
_boosters: dict | None = None


class ProfileIn(BaseModel):
    value: str = Field(min_length=2, max_length=300)
    refresh: bool = False
    discovery: int = Field(3, ge=1, le=6)
    named: int = Field(2, ge=0, le=4)


class QueriesIn(BaseModel):
    profile: dict
    discovery: int = Field(3, ge=1, le=6)
    named: int = Field(2, ge=0, le=4)


class PlanIn(BaseModel):
    brand: str = Field(min_length=1, max_length=120)
    sector: str = Field(min_length=1, max_length=120)
    language: str = "tr"
    aliases: list[str] = []
    discovery: list[str] = Field(min_length=1, max_length=6)
    named: list[str] = Field([], max_length=4)
    description: str = Field("", max_length=6000)
    second_assistant: bool = False
    audit_ai: bool = True
    reps: int = Field(2, ge=1, le=3)
    max_calls: int = Field(80, ge=5, le=200)


class AuditIn(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    rivals: list[str] = Field([], max_length=5)
    ai: bool = False


def _keys(*names: str) -> None:
    from visibility.llm import load_key

    try:
        for name in names:
            load_key(name)
    except ValueError as exc:
        raise HTTPException(503, f"{exc}. Anahtarı .env dosyasına ekleyin.") from exc


def options_from(plan: PlanIn) -> service.Options:
    return service.Options(
        brand=plan.brand.strip(),
        sector=plan.sector.strip(),
        language=plan.language if plan.language in {"tr", "en"} else "tr",
        brand_aliases=tuple(a.strip() for a in plan.aliases if a.strip()),
        custom_queries=tuple(q.strip() for q in plan.discovery if q.strip()),
        named_queries=tuple(q.strip() for q in plan.named if q.strip()),
        queries=len(plan.discovery),
        reps=plan.reps,
        max_calls=plan.max_calls,
        # A step that failed before (a 503, say) is asked again instead of blocking the run;
        # the receipts still cap every step at three attempts.
        retry_failed=True,
        description=plan.description.strip(),
        audit_ai=plan.audit_ai,
        assistants=(
            (assistants.PRIMARY, "cerebras") if plan.second_assistant else (assistants.PRIMARY,)
        ),
    )


def _client(http: httpx.AsyncClient):
    from advisor.clients import AdvisorClient

    return AdvisorClient(http)


@app.post("/api/profile")
async def build_profile(body: ProfileIn) -> dict:
    """Read the site (or search the name), extract the profile, write the questions."""
    _keys("GEMINI_API_KEY", "SERPER_API_KEY")
    folder = PROFILES / digest(body.value.strip())[:16]
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15)) as http:
        receipts = Receipts(folder, _client(http), retry_failed=True)
        try:
            found = await profile.build_profile(
                body.value.strip(), receipts, http, refresh=body.refresh
            )
            # When and whether the source was read describe this request, not the brand;
            # they stay out of the question payload so its receipt still matches.
            reading = {key: found.pop(key, None) for key in ("read_at", "changed")}
            questions = await profile.plan_queries(
                found, receipts, n_discovery=body.discovery, n_named=body.named
            )
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return {
        "profile": found | reading | {"description": profile.description(found)},
        "queries": questions,
    }


@app.post("/api/queries")
async def rewrite_queries(body: QueriesIn) -> dict:
    _keys("GEMINI_API_KEY")
    if not body.profile.get("brand") or not body.profile.get("sector"):
        raise HTTPException(422, "Profilde marka ve sektör olmalı.")
    folder = PROFILES / digest(body.profile["brand"])[:16]
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15)) as http:
        receipts = Receipts(folder, _client(http), retry_failed=True)
        try:
            return await profile.plan_queries(
                body.profile, receipts, n_discovery=body.discovery, n_named=body.named
            )
        except (ValueError, httpx.HTTPError) as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post("/api/estimate")
def estimate(plan: PlanIn) -> dict:
    options = options_from(plan)
    return {
        "calls": service.estimated_calls(options),
        "budget": options.max_calls,
        "cached": (service.run_folder(options) / "report.md").exists(),
        "run_id": service.run_id(options),
    }


def result_of(state: dict) -> dict:
    """What the page needs from a finished state, plain JSON."""
    out = {key: state.get(key) for key in RESULT_KEYS}
    diagnosis = state.get("diagnosis", "thin")
    out["diagnosis_title"] = DIAGNOSES.get(diagnosis, diagnosis)
    out["diagnosis_note"] = DIAGNOSIS_NOTE.get(diagnosis, "")
    out["assistant_labels"] = {name: assistants.label(name) for name in assistants.ASSISTANTS}
    out["answers"] = [
        {
            key: row.get(key)
            for key in (
                "query",
                "condition",
                "rep",
                "assistant",
                "kind",
                "mentioned",
                "first",
                "named_brands",
                "picked",
                "answer",
            )
        }
        for row in state.get("observations") or []
        if row.get("condition") == "search_on" and row.get("rep") == 0
    ]
    record = state.get("description_audit") or {}
    description = state.get("description_text") or ""
    if record.get("findings") is not None and description:
        out["description_marks"] = audit.marked(description, record)
    return out


async def _execute(job: str, options: service.Options) -> None:
    global _boosters
    run = RUNS[job]
    try:
        if _boosters is None:
            _boosters = await asyncio.to_thread(model.load)
        final: dict = {}
        async for node, state in service.stream(options, _boosters):
            run["steps"].append(node)
            final = state
        service.save(options, final)
        final["description_text"] = options.description
        run["result"] = result_of(final)
        run["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - the page shows every failure
        run["status"] = "error"
        run["error"] = (
            f"{str(exc) or type(exc).__name__} Analizi yeniden başlatırsanız tamamlanan adımlar "
            "yeniden ödenmez; kaldığı yerden devam eder."
        )


@app.post("/api/runs")
async def start_run(plan: PlanIn) -> dict:
    options = options_from(plan)
    calls = service.estimated_calls(options)
    if calls > options.max_calls:
        raise HTTPException(422, f"Tahmini {calls} çağrı, {options.max_calls} bütçesini aşıyor.")
    _keys(
        "GEMINI_API_KEY",
        "SERPER_API_KEY",
        *(["CEREBRAS_API_KEY"] if "cerebras" in options.assistants else []),
    )
    job = uuid.uuid4().hex[:12]
    RUNS[job] = {
        "status": "running",
        "steps": [],
        "result": None,
        "error": None,
        "calls": calls,
        "aliases": list(options.brand_aliases),
    }
    RUNS[job]["task"] = asyncio.create_task(_execute(job, options))
    return {"id": job, "calls": calls}


@app.get("/api/runs/{job}")
def run_status(job: str) -> dict:
    run = RUNS.get(job)
    if run is None:
        raise HTTPException(404, "Böyle bir analiz yok.")
    return {
        "status": run["status"],
        "steps": [{"node": n, "label": service.STEP_LABELS.get(n, n)} for n in run["steps"]],
        "error": run["error"],
        "result": run["result"],
    }


def attachment(name: str) -> dict[str, str]:
    """A download header that survives any brand name: headers are Latin-1 only."""
    plain = unicodedata.normalize("NFKD", name.translate(str.maketrans("ıİ", "iI")))
    plain = re.sub(r"[^A-Za-z0-9._-]+", "-", plain.encode("ascii", "ignore").decode()).strip("-")
    return {
        "Content-Disposition": f"attachment; filename=\"{plain or 'rapor'}\"; "
        f"filename*=UTF-8''{quote(name)}"
    }


def _finished(job: str) -> dict:
    run = RUNS.get(job)
    if run is None or not run.get("result"):
        raise HTTPException(404, "Rapor henüz hazır değil.")
    return run


@app.get("/api/runs/{job}/report.md")
def run_report(job: str) -> PlainTextResponse:
    run = _finished(job)
    return PlainTextResponse(
        run["result"]["report"] or "",
        media_type="text/markdown; charset=utf-8",
        headers=attachment(f"{run['result']['brand']} görünürlük raporu.md"),
    )


@app.get("/api/runs/{job}/report.pdf")
async def run_report_pdf(job: str) -> Response:
    run = _finished(job)
    try:
        data = await asyncio.to_thread(pdf.report_pdf, run["result"], run.get("aliases") or [])
    except ImportError as exc:
        raise HTTPException(
            503, "PDF için weasyprint gerekli; uygulamayı make advisor-ui ile başlatın."
        ) from exc
    return Response(
        data,
        media_type="application/pdf",
        headers=attachment(f"{run['result']['brand']} görünürlük raporu.pdf"),
    )


@app.post("/api/audit")
async def audit_text(body: AuditIn) -> dict:
    rivals = [r for r in body.rivals if r.strip()]
    method = "rules"
    found = rival_found = None
    note = None
    if body.ai:
        _keys("GEMINI_API_KEY")
        texts = {"brand": body.text, **{f"rival_{i}": r for i, r in enumerate(rivals, 1)}}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as http:
            try:
                classified = await audit.classify(
                    texts, Receipts(audit.AUDIT_OUTPUT, _client(http), retry_failed=True)
                )
            except (ValueError, httpx.HTTPError) as exc:
                note = f"Yapay zekâ sınıflandırması alınamadı ({exc}); kurallar kullanıldı."
            else:
                found = classified["brand"]
                rival_found = [classified[f"rival_{i}"] for i in range(1, len(rivals) + 1)]
                method = "ai"
    record = audit.as_record(
        audit.audit(body.text, rivals, found=found, rival_found=rival_found, method=method)
    )
    return {
        "record": record,
        "marks": audit.marked(body.text, record),
        "method_note": audit.METHOD_NOTE[record["method"]],
        "note": note,
        "verdicts": audit.VERDICT_TR,
    }


@app.get("/api/evidence")
def evidence() -> dict:
    """The measured win shares behind the audit, for the page's reference chart."""
    return {
        "types": [
            {
                "key": key,
                "label": audit.LABELS[key],
                "verdict": audit.verdict(key),
                "gemini": audit.shares(key)[0],
                "cerebras": audit.shares(key)[1],
            }
            for key in audit.WINS
        ],
        "chance": audit.CHANCE,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
