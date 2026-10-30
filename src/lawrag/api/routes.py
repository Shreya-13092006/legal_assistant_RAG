"""HTTP endpoints (SDD 3.7.1). No retrieval or verification logic lives here."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from lawrag.api.deps import AppState, get_state
from lawrag.api.schemas import AskRequest, CompareRequest, SearchHit, SearchRequest, VerifyRequest
from lawrag.errors import LLMTimeout
from lawrag.models.qa import AskOptions, QAResponse

router = APIRouter()


@router.get("/health")
def health(st: AppState = Depends(get_state)):
    from fastapi.responses import JSONResponse
    d = st.deps
    graph = d.store.health()
    llm_ok = None
    gen = d.llm.get("generator")
    ping = getattr(gen.adapter, "ping", None)
    if ping:
        llm_ok = "reachable" if ping() else "unreachable"
    else:
        llm_ok = gen.provider
    body = {"status": "ok" if graph.get("status") == "ok" else "degraded",
            "graph": graph, "neo4j": graph["status"] if graph.get("backend") == "neo4j" else "not used",
            "embedder": d.embedder.model_id, "bm25": "loaded" if d.bm25.ids else "empty", "llm": llm_ok,
            "graph_snapshot": d.snapshot_hash, "corpus_hash": d.corpus_hash,
            "cutoff": d.cfg.project.cutoff, "budget_remaining_usd": round(d.llm.budget.remaining(), 4)}
    return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)


@router.post("/ask", response_model=QAResponse, response_model_exclude_none=False)
async def ask(req: AskRequest, st: AppState = Depends(get_state)):
    opts = req.options or AskOptions()
    try:
        resp = await asyncio.wait_for(run_in_threadpool(st.qa.ask, req.question, opts),
                                      timeout=st.deps.cfg.api.timeout_s)
    except asyncio.TimeoutError as exc:
        raise LLMTimeout(f"request exceeded {st.deps.cfg.api.timeout_s}s") from exc
    if not opts.debug:
        resp.trace = None
    return resp


@router.get("/sections/{uid}")
def get_section(uid: str, include_qualifiers: bool = True, st: AppState = Depends(get_state)):
    return st.law.get_section(uid, include_qualifiers)


@router.get("/mappings/{uid}")
def get_mappings(uid: str, direction: str | None = Query(None, pattern="^(ipc_to_bns|bns_to_ipc)$"),
                 include_unverified: bool = False, st: AppState = Depends(get_state)):
    return st.law.get_mappings(uid, direction, include_unverified)


@router.post("/compare")
def compare(req: CompareRequest, st: AppState = Depends(get_state)):
    return st.law.compare(req.left_uid, req.right_uid)


@router.post("/search")
def search(req: SearchRequest, st: AppState = Depends(get_state)):
    hits = st.law.search(req.query, req.mode, req.k, req.act)
    return {"candidates": [SearchHit(uid=c.uid, score=c.fused_score, ranks=c.ranks, via=c.via) for c in hits]}


@router.post("/verify")
def verify(req: VerifyRequest, st: AppState = Depends(get_state)):
    return {"deterministic": st.qa.verify(req.answer, req.context_uids, req.question)}
