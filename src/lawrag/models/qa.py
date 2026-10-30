"""Online QA data models (SDD 3.4-3.5)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from lawrag.models.mapping import MappingFact

Intent = Literal["lookup", "mapping", "comparison", "scenario", "unknown"]
Direction = Literal["ipc_to_bns", "bns_to_ipc"]
Verdict = Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
Decision = Literal["accept", "partial", "retry", "abstain"]
Outcome = Literal["answered", "partial", "abstained"]


class QueryAnalysis(BaseModel):
    intent: Intent = "unknown"
    section_ids: list[str] = Field(default_factory=list)       # canonical uids found and existing
    nonexistent_ids: list[str] = Field(default_factory=list)   # well-formed but absent (false premise)
    ambiguous_act: bool = False
    acts: list[str] = Field(default_factory=list)
    direction: Direction | None = None
    offence_date: date | None = None
    rewritten_query: str = ""
    analyzer: Literal["rules", "rules+llm"] = "rules"
    out_of_scope: bool = False
    out_of_scope_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    uid: str                         # provision uid (or section uid for section_summary hits)
    section_uid: str
    act: str
    kind: str = "body"
    chunk_ids: list[str] = Field(default_factory=list)
    via: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    pinned: bool = False
    hops: int = 0
    fused_score: float = 0.0
    secondary: bool = False


class ContextBlock(BaseModel):
    label: str
    kind: Literal["provision", "qualifier", "mapping_fact", "secondary", "status_note"]
    text: str
    uids: list[str]
    source_id: str
    page: int | None = None
    excerpt: bool = False
    section_uid: str | None = None


class ContextPack(BaseModel):
    blocks: list[ContextBlock] = Field(default_factory=list)
    mapping_facts: list[MappingFact] = Field(default_factory=list)
    token_count: int = 0
    dropped: list[str] = Field(default_factory=list)

    def uids(self) -> set[str]:
        return {u for b in self.blocks for u in b.uids}

    def section_uids(self) -> set[str]:
        return {b.section_uid for b in self.blocks if b.section_uid}

    def mapping_ids(self) -> set[str]:
        return {m.mapping_id for m in self.mapping_facts}

    def render(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    core: bool = True
    verdict: Verdict | None = None
    evidence_span: str | None = None
    rationale: str | None = None


class Draft(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    mode: Literal["normal", "premise_correction"] = "normal"


class CheckResult(BaseModel):
    check: str                   # D1..D7
    passed: bool
    severity: Literal["hard", "soft"] = "hard"
    details: list[str] = Field(default_factory=list)
    claim_indexes: list[int] = Field(default_factory=list)


class DeterministicReport(BaseModel):
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def soft_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "soft"]

    def get(self, check: str) -> CheckResult | None:
        return next((c for c in self.checks if c.check == check), None)


class VerificationReport(BaseModel):
    deterministic: DeterministicReport = Field(default_factory=DeterministicReport)
    claims: list[Claim] = Field(default_factory=list)
    judge_enabled: bool = False
    support_ratio: float | None = None
    core_supported: bool | None = None
    decision: Decision | None = None
    decision_rule: str | None = None
    notes: list[str] = Field(default_factory=list)


class RetryPlan(BaseModel):
    action: Literal["targeted_lookup", "widen", "regenerate"]
    add_ids: list[str] = Field(default_factory=list)
    vector_k: int | None = None
    bm25_k: int | None = None
    extra_hops: int = 0
    add_refers_to: bool = False
    feedback: list[str] = Field(default_factory=list)
    reason: str = ""


class TraceEvent(BaseModel):
    node: str
    start_ms: float
    end_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    info: dict = Field(default_factory=dict)


class ClaimView(BaseModel):
    text: str
    citations: list[str]
    verdict: Verdict | None
    core: bool
    evidence_span: str | None = None


class CitationView(BaseModel):
    uid: str
    act: str
    section_title: str
    excerpt: str
    source_title: str | None = None
    page: int | None = None
    url: str | None = None
    status: str | None = None


class AskOptions(BaseModel):
    include_secondary: bool | None = None
    act: Literal["IPC", "BNS"] | None = None
    debug: bool = False


class QAResponse(BaseModel):
    request_id: str
    question: str = ""
    outcome: Outcome
    answer_markdown: str
    claims: list[ClaimView] = Field(default_factory=list)
    citations: list[CitationView] = Field(default_factory=list)
    mapping_facts: list[MappingFact] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    abstention_kind: Literal["evidence", "scope", "premise", "system_error"] | None = None
    verification: VerificationReport = Field(default_factory=VerificationReport)
    retry_actions: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    cost_estimate_usd: float = 0.0
    config_hash: str = ""
    analysis: QueryAnalysis | None = None
    context_uids: list[str] = Field(default_factory=list)
    retrieved_ranking: list[str] = Field(default_factory=list)   # fused uid order (for eval)
    trace: list[TraceEvent] | None = None
