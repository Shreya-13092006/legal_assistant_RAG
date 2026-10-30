"""Evaluation correctness judge (SDD 7.5.2). Configured separately (`llm.roles.eval_judge`) from the
pipeline judge. Calibrated against the human grading sample before its numbers are reported."""
from __future__ import annotations

from pydantic import BaseModel, Field

from lawrag.models.eval import GoldItem


class EvalVerdict(BaseModel):
    score: int = Field(ge=0, le=2)
    rationale: str = ""


def grade(pred: dict, item: GoldItem, llm) -> EvalVerdict:
    from lawrag.llm.client import Message, load_prompt
    user = (f"QUESTION: {item.question}\nEXPECTED BEHAVIOUR: {item.expected_behavior}\n"
            f"REFERENCE ANSWER: {item.gold_answer}\nKEY FACTS:\n" + "\n".join(f"- {k}" for k in item.key_facts) +
            f"\n\nSYSTEM OUTCOME: {pred['outcome']}\nSYSTEM ANSWER:\n{pred['answer_markdown']}")
    return llm.complete(load_prompt("eval_judge"), [Message(role="user", content=user)], schema=EvalVerdict).parsed
