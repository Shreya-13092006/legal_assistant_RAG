"""Claim decomposition fallback (SDD 3.6.2 step 1)."""
from __future__ import annotations

from pydantic import BaseModel

from lawrag.models.qa import Claim, Draft


class _Claims(BaseModel):
    claims: list[Claim]


def ensure_claims(draft: Draft, llm) -> Draft:
    """If the generator returned no usable claim list, split the answer with the decomposer prompt."""
    if draft.claims and all(c.text.strip() for c in draft.claims):
        return draft
    if not draft.answer.strip():
        return draft
    from lawrag.llm.client import Message, load_prompt
    res = llm.complete(load_prompt("decompose"), [Message(role="user", content=f"ANSWER:\n{draft.answer}")],
                       schema=_Claims)
    return draft.model_copy(update={"claims": [c for c in res.parsed.claims if c.text.strip()]})
