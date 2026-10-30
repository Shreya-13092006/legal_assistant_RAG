"""Claim-level judge (SDD 3.6.2). The judge sees ONLY the cited blocks. It checks consistency with
cited text; it is never a legal authority."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from lawrag.logging_setup import get_logger
from lawrag.models.qa import Claim, ContextPack
from lawrag.verify.deterministic import cited_text

log = get_logger("verify.judge")


class JudgeOut(BaseModel):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    evidence_span: str = ""
    rationale: str = ""


def _norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def span_in(span: str, text: str) -> bool:
    return bool(span.strip()) and _norm(span) in _norm(text)


def judge_claims(claims: list[Claim], ctx: ContextPack, llm, skip: set[int] | None = None) -> list[Claim]:
    from lawrag.llm.client import Message, load_prompt
    system = load_prompt("judge")
    out: list[Claim] = []
    for i, c in enumerate(claims):
        c = c.model_copy()
        if skip and i in skip:
            out.append(c)
            continue
        if not c.citations:
            c.verdict, c.rationale = "NOT_ENOUGH_INFO", "claim has no citation"
            out.append(c)
            continue
        evidence = cited_text(c, ctx)
        if not evidence:
            c.verdict, c.rationale = "NOT_ENOUGH_INFO", "cited labels are not in the context"
            out.append(c)
            continue
        try:
            res = llm.complete(system, [Message(role="user", content=f"CLAIM: {c.text}\nCITED TEXT:\n{evidence}")],
                               schema=JudgeOut)
            j: JudgeOut = res.parsed
            c.verdict, c.evidence_span, c.rationale = j.verdict, j.evidence_span or None, j.rationale
            if j.verdict == "SUPPORTED" and not span_in(j.evidence_span, evidence):
                c.verdict = "NOT_ENOUGH_INFO"
                c.rationale = f"downgraded: evidence span not found verbatim in cited text ({j.rationale})"
        except Exception as exc:  # judge failure -> conservative NEI
            log.warning(f"judge failed on claim {i}: {exc}")
            c.verdict, c.rationale = "NOT_ENOUGH_INFO", f"judge error: {exc}"
        out.append(c)
    return out
