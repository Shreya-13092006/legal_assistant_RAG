"""Gold-dataset schema (SDD 7.2.2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ExpectedBehavior = Literal["answer", "partial_ok", "abstain", "correct_premise", "state_dispute",
                           "state_unresolved"]


class GoldSection(BaseModel):
    uid: str
    relevance: Literal[1, 2] = 2


class GoldMapping(BaseModel):
    mapping_id: str
    from_: list[str] = Field(default_factory=list, alias="from")
    to: list[str] = Field(default_factory=list)
    type: str
    status: str

    model_config = {"populate_by_name": True}


class GoldEvidence(BaseModel):
    source_id: str
    page: int | None = None
    row: int | None = None


class GoldItem(BaseModel):
    id: str
    version: int = 1
    split: Literal["dev", "test", "smoke"]
    category: str
    subtype: str | None = None
    question: str
    expected_behavior: ExpectedBehavior
    gold_answer: str = ""
    key_facts: list[str] = Field(default_factory=list)
    gold_sections: list[GoldSection] = Field(default_factory=list)
    gold_mappings: list[GoldMapping] = Field(default_factory=list)
    evidence: list[GoldEvidence] = Field(default_factory=list)
    trap_type: Literal["nonexistent_section", "wrong_act", "struck_down_or_repealed",
                       "nonexistent_mapping"] | None = None
    abstain_reason: Literal["out_of_scope", "not_in_corpus", "insufficient_evidence"] | None = None
    temporal: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    authored_by: str = "developer"
    drafting_aid: Literal["none", "llm_draft"] = "none"
    reviewed_by: str | None = None
    review_status: Literal["draft", "verified", "rejected"] = "draft"
    review_date: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _rules(self) -> "GoldItem":
        if self.expected_behavior == "abstain" and not self.abstain_reason:
            raise ValueError(f"{self.id}: abstain items need abstain_reason")
        if self.expected_behavior == "correct_premise" and not self.trap_type:
            raise ValueError(f"{self.id}: correct_premise items need trap_type")
        if self.expected_behavior in {"answer", "correct_premise", "state_dispute", "state_unresolved"} \
                and not self.evidence and self.split != "smoke":
            raise ValueError(f"{self.id}: {self.expected_behavior} items must list evidence")
        return self
