"""Typed workflow state (SDD 3.5.1)."""
from __future__ import annotations

from typing import Any, TypedDict

from lawrag.models.mapping import MappingFact
from lawrag.models.qa import (AskOptions, Candidate, ContextPack, Draft, QAResponse, QueryAnalysis, RetryPlan,
                              TraceEvent, VerificationReport)


class QAState(TypedDict, total=False):
    # identity and configuration
    request_id: str
    run_id: str | None
    config_hash: str
    question: str
    options: AskOptions
    # analysis and retrieval
    analysis: QueryAnalysis
    retrieved: dict[str, list[Candidate]]
    mapping_facts: list[MappingFact]
    fused: list[Candidate]
    context: ContextPack
    degraded: list[str]
    # generation and verification
    draft: Draft | None
    verification: VerificationReport | None
    drop_claims: list[int]
    feedback: list[str]
    # control
    attempt: int
    retry_plan: RetryPlan | None
    retry_actions: list[str]
    route: str
    decision: str | None
    decision_reason: str
    outcome: str | None
    abstention_kind: str | None
    system_error: str | None
    final: QAResponse | None
    trace: list[TraceEvent]
    usage: list[dict[str, Any]]
