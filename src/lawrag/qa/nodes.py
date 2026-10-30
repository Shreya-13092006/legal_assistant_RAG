"""Workflow nodes (SDD 3.5.2). Plain functions `(state, deps) -> partial state update`, so they are
testable without LangGraph. The LLM never chooses control flow."""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from lawrag import DISCLAIMER
from lawrag.errors import BudgetExceeded, LLMError, LLMOutputError, LLMTimeout, RetrievalError
from lawrag.legal import uid as U
from lawrag.llm.client import Message, load_prompt
from lawrag.logging_setup import get_logger
from lawrag.models.qa import (CitationView, Claim, ClaimView, ContextPack, Draft, QAResponse, TraceEvent,
                              VerificationReport)
from lawrag.qa.retry import plan_retry as _plan_retry
from lawrag.retrieval.context_builder import build_context as _build_context
from lawrag.retrieval.fusion import rrf, single
from lawrag.retrieval.graph_expand import graph_expand as _graph_expand
from lawrag.retrieval.mapping_facts import facts_for
from lawrag.retrieval.retrievers import BM25Retriever, IdLookupRetriever, VectorRetriever
from lawrag.services.container import Deps
from lawrag.verify.claims import ensure_claims
from lawrag.verify.decide import decide as _decide
from lawrag.verify.deterministic import deterministic_checks, norm_citation, premise_claim_indexes
from lawrag.verify.judge import judge_claims

log = get_logger("qa.nodes")


class DraftOut(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def traced(name: str):
    def wrap(fn):
        def inner(state, deps):
            t0 = time.perf_counter() * 1000
            usage = state.get("usage")
            n0 = len(usage) if usage is not None else 0
            update = fn(state, deps) or {}
            t1 = time.perf_counter() * 1000
            calls = usage[n0:] if usage is not None else []
            ev = TraceEvent(node=name, start_ms=t0, end_ms=t1, tokens_in=sum(c["in"] for c in calls),
                            tokens_out=sum(c["out"] for c in calls), cost_usd=sum(c["cost"] for c in calls),
                            cache_hits=sum(1 for c in calls if c["cached"]), info=update.pop("_info", {}))
            update["trace"] = list(state.get("trace", [])) + [ev]
            log.info(f"node {name}", extra={"data": {"ms": round(t1 - t0, 1), **ev.info}})
            return update
        inner.__name__ = name
        return inner
    return wrap


# -- analysis & retrieval -----------------------------------------------------------------------
@traced("analyze_query")
def analyze_query(state, deps: Deps):
    a = deps.analyzer.analyze(state["question"])
    opts = state.get("options")
    if opts and opts.act and a.ambiguous_act:
        a.section_ids = [u for u in a.section_ids if U.act_of(u) == opts.act]
        a.ambiguous_act = False
    return {"analysis": a, "_info": {"intent": a.intent, "ids": a.section_ids, "nonexistent": a.nonexistent_ids,
                                     "out_of_scope": a.out_of_scope}}


@traced("retrieve")
def retrieve(state, deps: Deps):
    cfg, a = deps.cfg, state["analysis"]
    plan = state.get("retry_plan")
    include_sec = (state.get("options") and state["options"].include_secondary) or cfg.corpus.include_secondary
    retrieved, facts, degraded = {}, {}, list(state.get("degraded", []))
    extra_ids = plan.add_ids if plan else []
    if cfg.retrieval.id_lookup or extra_ids:
        retrieved["id_lookup"] = IdLookupRetriever(deps.store).retrieve(a, extra_ids=extra_ids)
        if a.intent in ("mapping", "comparison") or extra_ids:
            for uid in list(dict.fromkeys(a.section_ids + extra_ids)):
                for f in facts_for(uid, deps.store, a.direction if uid in a.section_ids else None):
                    facts[f.mapping_id] = f
    if cfg.retrieval.vector.enabled:
        k = (plan.vector_k if plan and plan.vector_k else cfg.retrieval.vector.k)
        try:
            retrieved["vector"] = VectorRetriever(deps.store, deps.embedder, cfg.retrieval.act_filter,
                                                  include_sec).retrieve(a, k)
        except RetrievalError as exc:  # degrade, continue with the remaining retrievers
            degraded.append(f"vector: {exc}")
    if cfg.retrieval.bm25.enabled:
        k = (plan.bm25_k if plan and plan.bm25_k else cfg.retrieval.bm25.k)
        retrieved["bm25"] = BM25Retriever(deps.store, deps.bm25, cfg.retrieval.act_filter, include_sec).retrieve(a, k)
    prev = {f.mapping_id: f for f in state.get("mapping_facts", [])} if plan else {}
    return {"retrieved": retrieved, "mapping_facts": list({**prev, **facts}.values()), "degraded": degraded,
            "_info": {k: len(v) for k, v in retrieved.items()}}


@traced("graph_expand")
def graph_expand(state, deps: Deps):
    cfg = deps.cfg
    if not cfg.retrieval.graph.enabled:
        return {"_info": {"skipped": True}}
    first = {k: v for k, v in state["retrieved"].items() if k != "graph"}
    prelim = rrf(first, cfg.fusion.k_rrf) if first else []
    pinned = [c for c in prelim if c.pinned]
    others = [c for c in prelim if not c.pinned][: cfg.retrieval.graph.seeds]
    seeds = pinned + others
    for i, s in enumerate(seeds, 1):
        s.ranks["seed"] = i
    plan = state.get("retry_plan")
    expanded, facts = _graph_expand(seeds, state["analysis"], deps.store, cfg.retrieval.graph,
                                    extra_rels={"REFERS_TO"} if plan and plan.add_refers_to else None,
                                    extra_hops=plan.extra_hops if plan else 0)
    merged = {f.mapping_id: f for f in state.get("mapping_facts", [])}
    merged.update({f.mapping_id: f for f in facts})
    retrieved = dict(state["retrieved"])
    retrieved["graph"] = expanded
    return {"retrieved": retrieved, "mapping_facts": sorted(merged.values(), key=lambda f: f.mapping_id),
            "_info": {"expanded": len(expanded), "mapping_facts": len(merged)}}


@traced("fuse")
def fuse(state, deps: Deps):
    cfg = deps.cfg
    lists = {k: v for k, v in state["retrieved"].items() if v}
    pinned = [c.uid for c in lists.get("id_lookup", [])]
    if cfg.fusion.method == "none" and len(lists) <= 1:
        fused = single(next(iter(lists.values()), []), cfg.fusion.top_n)
    else:
        fused = rrf(lists, cfg.fusion.k_rrf, cfg.fusion.weights, pinned, cfg.fusion.top_n,
                    cfg.fusion.illustration_weight, state["analysis"].intent)
    if cfg.rerank.enabled:
        from lawrag.retrieval.rerank import CrossEncoderReranker, LLMReranker
        rr = deps.extras.get("reranker")
        if rr is None:
            rr = LLMReranker(deps.llm.get("reranker"), deps.store) if cfg.rerank.kind == "llm" else \
                CrossEncoderReranker(deps.store)
            deps.extras["reranker"] = rr
        pool = rrf(lists, cfg.fusion.k_rrf, cfg.fusion.weights, pinned, cfg.rerank.top_in,
                   cfg.fusion.illustration_weight, state["analysis"].intent)
        fused = rr.rerank(state["analysis"].rewritten_query, pool, cfg.rerank.top_out)
    return {"fused": fused, "_info": {"fused": [c.uid for c in fused[:10]]}}


@traced("build_context")
def build_context(state, deps: Deps):
    include_sec = (state.get("options") and state["options"].include_secondary) or deps.cfg.corpus.include_secondary
    ctx = _build_context(state["fused"], state["analysis"], deps.store, deps.cfg.context.budget_tokens,
                         state.get("mapping_facts", []), bool(include_sec))
    return {"context": ctx, "_info": {"blocks": len(ctx.blocks), "tokens": ctx.token_count, "dropped": ctx.dropped}}


def evidence_gate(state, deps: Deps) -> str:
    a, ctx = state["analysis"], state.get("context") or ContextPack()
    if a.out_of_scope:
        return "abstain"
    if a.nonexistent_ids:
        return "generate"
    gate = deps.cfg.workflow.gate.min_retriever_rank
    has = any(c.pinned for c in state.get("fused", [])) or bool(ctx.mapping_facts) or any(
        min(c.ranks.values() or [10**6]) <= gate for c in state.get("fused", []))
    if has and ctx.blocks:
        return "generate"
    if state.get("attempt", 0) < deps.cfg.workflow.max_retries:
        return "plan_retry"
    return "abstain"


# -- generation & verification -------------------------------------------------------------------
def _user_message(state) -> str:
    a = state["analysis"]
    mode = "premise_correction" if a.nonexistent_ids else "normal"
    lines = [f"QUESTION: {state['question']}", f"MODE: {mode}",
             f"NONEXISTENT SECTIONS: {', '.join(a.nonexistent_ids) or 'none'}",
             f"INTENT: {a.intent}"]
    if a.offence_date:
        lines.append(f"OFFENCE DATE: {a.offence_date.isoformat()}")
    if a.ambiguous_act:
        lines.append("NOTE: the section number exists in both acts; answer for each act separately.")
    fb = state.get("feedback") or []
    if fb:
        lines.append("FEEDBACK FROM VERIFICATION (fix these):")
        lines += [f"- {f}" for f in fb]
    lines.append("SOURCES:")
    lines.append(state["context"].render())
    return "\n".join(lines)


@traced("generate")
def generate(state, deps: Deps):
    a = state["analysis"]
    mode = "premise_correction" if a.nonexistent_ids else "normal"
    llm = deps.llm.get("generator")
    msg = [Message(role="user", content=_user_message(state))]
    for attempt in range(2):
        try:
            res = llm.complete(load_prompt("generate"), msg, schema=DraftOut, seed=deps.cfg.seed)
            d: DraftOut = res.parsed
            draft = Draft(answer=d.answer, claims=d.claims, caveats=d.caveats, mode=mode)
            return {"draft": draft, "_info": {"claims": len(draft.claims), "cached": res.cached}}
        except LLMTimeout as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return {"draft": None, "system_error": f"generator timed out: {exc}"}
        except LLMOutputError as exc:  # treated as a verification failure, never a crash
            return {"draft": Draft(answer="", claims=[], mode=mode), "_info": {"output_error": str(exc)}}
        except BudgetExceeded:
            raise
        except LLMError as exc:
            return {"draft": None, "system_error": f"generator unavailable: {exc}"}
    return {"draft": None, "system_error": "generator failed"}


def after_generate(state, deps: Deps) -> str:
    return "abstain" if state.get("system_error") else "verify_deterministic"


@traced("verify_deterministic")
def verify_deterministic(state, deps: Deps):
    draft = state["draft"]
    try:
        draft = ensure_claims(draft, deps.llm.get("decomposer"))
    except (LLMError, LLMOutputError) as exc:
        log.warning(f"decomposer failed: {exc}")
    rep = VerificationReport(judge_enabled=False)
    if deps.cfg.verification.deterministic:
        rep.deterministic = deterministic_checks(draft, state["context"], state["analysis"], deps.store,
                                                 deps.commencement, list(deps.cfg.acts))
    rep.claims = [c.model_copy() for c in draft.claims]
    for i in premise_claim_indexes(draft, state["analysis"]):
        rep.claims[i].verdict = "SUPPORTED"
        rep.claims[i].evidence_span = "deterministic:D7 (section absent from the graph)"
    return {"draft": draft, "verification": rep,
            "_info": {"failed": [c.check for c in rep.deterministic.checks if not c.passed]}}


@traced("verify_claims")
def verify_claims(state, deps: Deps):
    rep: VerificationReport = state["verification"].model_copy(deep=True)
    cfg = deps.cfg
    hard = rep.deterministic.hard_failures
    if not cfg.verification.claim_judge or (hard and state.get("attempt", 0) >= cfg.workflow.max_retries):
        rep.notes.append("claim judge skipped" if not cfg.verification.claim_judge else
                         "claim judge skipped: hard deterministic failure with no retry left")
        return {"verification": rep, "_info": {"skipped": True}}
    skip = {i for i, c in enumerate(rep.claims) if c.verdict == "SUPPORTED"}
    rep.claims = judge_claims(rep.claims, state["context"], deps.llm.get("judge"), skip)
    rep.judge_enabled = True
    return {"verification": rep, "_info": {"verdicts": [c.verdict for c in rep.claims]}}


@traced("decide")
def decide(state, deps: Deps):
    rep: VerificationReport = state["verification"].model_copy(deep=True)
    cfg = deps.cfg
    if not cfg.verification.enforce:  # V0 baseline: results recorded, not enforced
        rep.decision, rep.decision_rule = "accept", "V0"
        return {"verification": rep, "decision": "accept", "drop_claims": [], "decision_reason": "verification off"}
    d = _decide(rep, state["draft"], state["context"], state["analysis"], state.get("attempt", 0),
                cfg.workflow.max_retries, cfg.verification.tau_partial)
    rep.decision, rep.decision_rule = d.decision, d.rule
    rep.notes.append(f"{d.rule}: {d.reason}")
    return {"verification": rep, "decision": d.decision, "drop_claims": d.drop_claims, "decision_reason": d.reason,
            "_info": {"decision": d.decision, "rule": d.rule}}


def route_decision(state, deps: Deps) -> str:
    return {"accept": "finalize", "partial": "finalize", "retry": "plan_retry"}.get(state["decision"], "abstain")


@traced("plan_retry")
def plan_retry(state, deps: Deps):
    attempt = state.get("attempt", 0)
    assert attempt < deps.cfg.workflow.max_retries, "retry budget exceeded"  # impossible by construction
    plan = _plan_retry(state.get("verification"), state["analysis"], state.get("context"), deps.store, deps.cfg)
    actions = list(state.get("retry_actions", [])) + [f"{plan.action}: {plan.reason}"]
    return {"retry_plan": plan, "attempt": attempt + 1, "feedback": plan.feedback, "retry_actions": actions,
            "draft": None, "verification": None, "_info": {"action": plan.action, "add_ids": plan.add_ids}}


def route_retry(state, deps: Deps) -> str:
    return "generate" if state["retry_plan"].action == "regenerate" else "retrieve"


# -- outcomes ----------------------------------------------------------------------------------
def _citation_views(uids: list[str], deps: Deps) -> list[CitationView]:
    out = []
    for u in dict.fromkeys(uids):
        if U.MAPPING_ID_RE.match(u) or "#sec#" in u or not deps.store.exists(u):
            continue
        sec = deps.store.get_section(U.section_of(u))
        p = deps.store.get_provision(u)
        src = deps.store.get_source(sec.source_id) if sec else None
        text = p.text if p else (sec.text if sec else "")
        out.append(CitationView(uid=u, act=U.act_of(u), section_title=sec.title if sec else "",
                                excerpt=text[:400] + ("…" if len(text) > 400 else ""),
                                source_title=src.title if src else None,
                                page=(p.page if p and p.page else (sec.page_start if sec else None)),
                                url=src.url if src else None, status=sec.status if sec else None))
    return out


def _base_response(state, deps: Deps, outcome: str) -> QAResponse:
    usage = state.get("usage") or []
    trace = state.get("trace", [])
    timings: dict[str, float] = {}
    for ev in trace:
        timings[ev.node] = round(timings.get(ev.node, 0.0) + ev.end_ms - ev.start_ms, 1)
    timings["total"] = round(sum(v for k, v in timings.items()), 1)
    ctx = state.get("context")
    opts = state.get("options")
    return QAResponse(request_id=state["request_id"], question=state["question"], outcome=outcome,
                      answer_markdown="", disclaimer=DISCLAIMER, config_hash=state.get("config_hash", ""),
                      timings_ms=timings, cost_estimate_usd=round(sum(u["cost"] for u in usage), 6),
                      analysis=state.get("analysis"), retry_actions=state.get("retry_actions", []),
                      mapping_facts=ctx.mapping_facts if ctx else [],
                      context_uids=sorted(ctx.uids()) if ctx else [],
                      retrieved_ranking=[c.uid for c in state.get("fused", [])],
                      verification=state.get("verification") or VerificationReport(),
                      trace=trace if (opts and opts.debug) or state.get("run_id") else None)


@traced("abstain")
def abstain(state, deps: Deps):
    a = state["analysis"]
    ctx = state.get("context")
    if state.get("system_error"):
        kind, reason = "system_error", f"A system error prevented an answer: {state['system_error']}."
    elif a.out_of_scope:
        kind, reason = "scope", (f"This question is outside the system's scope: {a.out_of_scope_reason}. "
                                 "The system covers the substantive offences of the IPC and the BNS only.")
    else:
        kind = "premise" if a.nonexistent_ids else "evidence"
        searched = [r for r in ("id_lookup", "vector", "bm25", "graph") if r in deps.cfg.enabled_retrievers()]
        found = [c.uid for c in state.get("fused", [])[:5]]
        parts = [f"Searched with {', '.join(searched)}"
                 + (f" for {', '.join(a.section_ids)}" if a.section_ids else "") + "."]
        parts.append(f"Top evidence found: {', '.join(found) or 'none'}.")
        if a.nonexistent_ids:
            parts.append(f"The section(s) {', '.join(a.nonexistent_ids)} do not exist in the indexed acts, and "
                         "the premise could not be confirmed with verified facts.")
        if state.get("decision_reason"):
            parts.append(f"Why insufficient: {state['decision_reason']}.")
        if state.get("retry_actions"):
            parts.append(f"Retry attempted: {'; '.join(state['retry_actions'])}.")
        reason = " ".join(parts)
    resp = _base_response(state, deps, "abstained")
    related = [f for f in (ctx.mapping_facts if ctx else []) if f.status == "verified"]
    md = "**The system could not give a verified answer.**\n\n" + reason
    if related:
        md += "\n\nRelated verified facts:\n" + "\n".join(
            f"- [{f.mapping_id}] {', '.join(f.from_uids) or '—'} → {', '.join(f.to_uids) or '—'} ({f.type})"
            for f in related)
    resp.answer_markdown, resp.abstention_reason, resp.abstention_kind = md, reason, kind
    if state.get("draft") and state["draft"].claims:
        resp.unverified_claims = [c.text for c in state["draft"].claims]
    return {"final": resp, "outcome": "abstained", "abstention_kind": kind}


@traced("finalize")
def finalize(state, deps: Deps):
    draft: Draft = state["draft"]
    rep: VerificationReport = state["verification"]
    drop = set(state.get("drop_claims", []))
    claims = rep.claims if rep and rep.claims else draft.claims
    outcome = "partial" if state.get("decision") == "partial" else "answered"
    kept = [c for i, c in enumerate(claims) if i not in drop]
    removed = [c.text for i, c in enumerate(claims) if i in drop]
    if outcome == "partial":
        md = "\n\n".join(f"{c.text} " + " ".join(f"[{norm_citation(x)}]" for x in c.citations) for c in kept)
        md += "\n\n_Some statements could not be verified against the cited text and were removed (see " \
              "\"Could not verify\")._"
    else:
        md = draft.answer
    if draft.caveats:
        md += "\n\n" + "\n".join(f"> {c}" for c in draft.caveats)
    resp = _base_response(state, deps, outcome)
    resp.answer_markdown = md
    resp.claims = [ClaimView(text=c.text, citations=[norm_citation(x) for x in c.citations], verdict=c.verdict,
                             core=c.core, evidence_span=c.evidence_span) for c in kept]
    resp.unverified_claims = removed
    resp.citations = _citation_views([norm_citation(x) for c in kept for x in c.citations], deps)
    return {"final": resp, "outcome": outcome}
