"""Decision policy: accept / partial / retry / abstain (SDD 3.5.3). Rules are evaluated in order."""
from __future__ import annotations

from dataclasses import dataclass, field

from lawrag.legal import uid as U
from lawrag.models.qa import ContextPack, Draft, QueryAnalysis, VerificationReport
from lawrag.verify.deterministic import mentioned_sections, norm_citation


@dataclass
class DecisionResult:
    decision: str
    rule: str
    drop_claims: list[int] = field(default_factory=list)
    reason: str = ""


def effective_verdicts(report: VerificationReport) -> list[str]:
    soft_bad = {i for chk in report.deterministic.soft_failures for i in chk.claim_indexes}
    out = []
    for i, c in enumerate(report.claims):
        v = c.verdict or ("SUPPORTED" if not report.judge_enabled else "NOT_ENOUGH_INFO")
        if i in soft_bad and v == "SUPPORTED":
            v = "NOT_ENOUGH_INFO"
        out.append(v)
    return out


def _core(report: VerificationReport) -> list[int]:
    idx = [i for i, c in enumerate(report.claims) if c.core]
    return idx or list(range(len(report.claims)))


def retrievable_gap(report: VerificationReport, ctx: ContextPack, analysis: QueryAnalysis,
                    idxs: list[int]) -> list[str]:
    """uids named by the given claims (or by the question) that are not in the context."""
    have = ctx.uids() | ctx.section_uids() | ctx.mapping_ids()
    gap: list[str] = []
    for i in idxs:
        c = report.claims[i]
        for x in [norm_citation(x) for x in c.citations] + sorted(mentioned_sections(c.text, ["IPC", "BNS"])):
            if x not in have and (U.MAPPING_ID_RE.match(x) or U.is_valid(x)) and (
                    U.MAPPING_ID_RE.match(x) or U.section_of(x) not in have):
                gap.append(x)
    gap += [u for u in analysis.section_ids if u not in have and U.section_of(u) not in have]
    return list(dict.fromkeys(gap))


def decide(report: VerificationReport, draft: Draft, ctx: ContextPack, analysis: QueryAnalysis, attempt: int,
           max_retries: int, tau_partial: float) -> DecisionResult:
    can_retry = attempt < max_retries
    v = effective_verdicts(report)
    n = len(v)
    core = _core(report)
    hard = report.deterministic.hard_failures
    supported = [i for i in range(n) if v[i] == "SUPPORTED"]
    contradicted = [i for i in range(n) if v[i] == "CONTRADICTED"]
    core_supported = bool(n) and all(v[i] == "SUPPORTED" for i in core)
    ratio = len(supported) / n if n else 0.0
    report.support_ratio, report.core_supported = ratio, core_supported

    def partial_ok(excluded: set[int]) -> DecisionResult | None:
        keep = [i for i in range(n) if i not in excluded]
        if not keep:
            return None
        core_k = [i for i in core if i in keep] or keep
        sup_k = [i for i in keep if v[i] == "SUPPORTED"]
        if all(v[i] == "SUPPORTED" for i in core_k) and len(sup_k) / n >= tau_partial:
            drop = sorted(excluded | {i for i in keep if v[i] != "SUPPORTED"})
            return DecisionResult("partial", "R5", drop, "unsupported non-core claims removed")
        return None

    # R1 premise correction
    if draft.mode == "premise_correction" and not hard and core_supported:
        return DecisionResult("accept", "R1", reason="premise corrected with supported facts")
    # R2 everything supported
    if not hard and not contradicted and n and len(supported) == n:
        return DecisionResult("accept", "R2", reason="all claims supported, no hard failures")
    # R3 hard deterministic failure
    if hard:
        if can_retry:
            return DecisionResult("retry", "R3", reason="; ".join(d for h in hard for d in h.details[:2]))
        failing = {i for h in hard for i in h.claim_indexes}
        if failing and len(failing) < n and not any(h.check in ("D7", "D6") and not h.claim_indexes for h in hard):
            p = partial_ok(failing | set(contradicted))
            if p:
                p.rule = "R3->R5"
                return p
        return DecisionResult("abstain", "R3", reason="hard verification failure after the retry")
    # R4 contradicted core claim
    if any(i in core for i in contradicted):
        return DecisionResult("retry" if can_retry else "abstain", "R4", reason="a core claim is contradicted")
    # R5 partial
    p = partial_ok(set(contradicted))
    if p and any(v[i] != "SUPPORTED" for i in range(n)):
        return p
    # R6 retrievable gap
    unsupported_core = [i for i in core if i < n and v[i] != "SUPPORTED"]
    gap = retrievable_gap(report, ctx, analysis, unsupported_core)
    if (unsupported_core or n == 0) and gap:
        return DecisionResult("retry" if can_retry else "abstain", "R6",
                              reason=f"evidence gap that may be retrievable: {gap}")
    # R7
    return DecisionResult("abstain", "R7", reason="core claims unsupported" if n else "no claims produced")
