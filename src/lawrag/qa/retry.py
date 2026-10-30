"""Deterministic, bounded retry planning (SDD 3.5.4). First applicable action wins."""
from __future__ import annotations

from lawrag.config import RunConfig
from lawrag.graph.store import GraphStore
from lawrag.legal import uid as U
from lawrag.models.qa import ContextPack, QueryAnalysis, RetryPlan, VerificationReport
from lawrag.verify.decide import effective_verdicts, retrievable_gap


def _feedback(report: VerificationReport | None) -> list[str]:
    if report is None:
        return []
    fb = [f"{chk.check}: {d}" for chk in report.deterministic.checks if not chk.passed for d in chk.details]
    if report.judge_enabled:
        for i, c in enumerate(report.claims):
            if c.verdict == "CONTRADICTED":
                fb.append(f"Claim {i} ('{c.text[:120]}') is contradicted by its cited text; correct or remove it.")
            elif c.verdict == "NOT_ENOUGH_INFO" and c.core:
                fb.append(f"Claim {i} ('{c.text[:120]}') is not supported by its cited text; support it or remove it.")
    return fb


def plan_retry(report: VerificationReport | None, analysis: QueryAnalysis, ctx: ContextPack | None,
               store: GraphStore, cfg: RunConfig) -> RetryPlan:
    ctx = ctx or ContextPack()
    feedback = _feedback(report)
    # 1. targeted lookup: uids named in failed / NEI claims or in the question but missing from the context
    gap: list[str] = []
    if report is not None:
        v = effective_verdicts(report)
        bad = {i for chk in report.deterministic.hard_failures for i in chk.claim_indexes}
        bad |= {i for i, x in enumerate(v) if x != "SUPPORTED"}
        gap = retrievable_gap(report, ctx, analysis, sorted(bad))
    else:
        have = ctx.uids() | ctx.section_uids()
        gap = [u for u in analysis.section_ids if u not in have]
    ids = []
    for g in gap:
        if U.MAPPING_ID_RE.match(g):
            m = store.get_mapping(g)
            ids += (m.from_uids + m.to_uids) if m else []
        elif store.exists(g):
            ids.append(g)
    ids = list(dict.fromkeys(ids))
    if ids:
        return RetryPlan(action="targeted_lookup", add_ids=ids, feedback=feedback,
                         reason=f"look up {ids} named in failed claims or the question")
    # 3 before 2: a citation/contradiction failure with the evidence already present only needs regeneration
    if report is not None and (report.deterministic.hard_failures or any(
            c.verdict == "CONTRADICTED" for c in report.claims)):
        return RetryPlan(action="regenerate", feedback=feedback, reason="evidence present; regenerate with feedback")
    # 2. widen
    return RetryPlan(action="widen", vector_k=cfg.retrieval.vector.k * 2, bm25_k=cfg.retrieval.bm25.k * 2,
                     extra_hops=1, add_refers_to=True, feedback=feedback,
                     reason="insufficient evidence; widen k and graph hops")
