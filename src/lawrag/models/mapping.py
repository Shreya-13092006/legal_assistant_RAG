"""Mapping-side data models (SDD 3.1.4, 3.2.4-3.2.6)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MappingType = Literal["IDENTICAL", "MODIFIED", "SPLIT", "MERGED", "PARTIAL", "OMITTED", "NEW", "UNDETERMINED"]
MappingStatus = Literal["proposed", "verified", "disputed", "unresolved", "rejected"]
FINAL_STATUSES = {"verified", "disputed", "unresolved", "rejected"}


class RawMappingRow(BaseModel):
    source_id: str
    page: int
    row_idx: int
    raw_cells: list[str]
    ipc_uids: list[str] = Field(default_factory=list)
    bns_uids: list[str] = Field(default_factory=list)
    marker: Literal["new", "deleted", "omitted", "none"] = "none"
    remark: str = ""
    range_expanded: bool = False
    parse_error: str | None = None


class Evidence(BaseModel):
    source_id: str
    page: int | None = None
    row: int | None = None
    raw_row: str | None = None
    extractor: str = "manual"


class Endpoint(BaseModel):
    uid: str
    role: Literal["primary", "partial"] = "primary"


class MappingRecord(BaseModel):
    """A :Mapping hub with its edges, as stored in the graph."""
    mapping_id: str
    type: MappingType
    status: MappingStatus
    granularity: Literal["section", "provision"] = "section"
    sources_agree: bool = True
    created_from: str = "seed"
    from_: list[Endpoint] = Field(default_factory=list, alias="from")
    to: list[Endpoint] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    change_summary: str | None = None
    text_similarity: float | None = None
    reviewed_by: str | None = None
    reviewed_on: str | None = None
    review_note: str | None = None
    dispute_note: str | None = None

    model_config = {"populate_by_name": True}

    @property
    def from_uids(self) -> list[str]:
        return [e.uid for e in self.from_]

    @property
    def to_uids(self) -> list[str]:
        return [e.uid for e in self.to]


class MappingCandidate(BaseModel):
    """Machine-proposed mapping awaiting review (status = proposed)."""
    candidate_id: str
    ipc_uids: list[str]
    bns_uids: list[str]
    proposed_type: MappingType
    type_basis: str
    sources_agree: bool
    text_similarity: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class MappingFact(BaseModel):
    """What the context builder, generator, verifier and UI see about a hub."""
    mapping_id: str
    type: MappingType
    status: MappingStatus
    granularity: str = "section"
    from_uids: list[str] = Field(default_factory=list)
    to_uids: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)   # source_id, title, page, row, raw_row, authority_rank
    change_summary: str | None = None
    review_note: str | None = None
    dispute_note: str | None = None
    reviewed_by: str | None = None
    reviewed_on: str | None = None
    sources_agree: bool = True
    authoritative: bool = False
    caveat: str | None = None
