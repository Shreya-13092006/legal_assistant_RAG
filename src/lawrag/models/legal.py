"""Statute-side data models (SDD 3.1.2, 2.7)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SectionStatus = Literal["in_force", "repealed", "struck_down", "omitted_by_amendment", "not_yet_in_force"]
ProvisionKind = Literal["body", "subsection", "clause", "subclause", "explanation", "exception", "proviso",
                        "illustration"]
QUALIFIER_KINDS = {"explanation", "exception", "proviso", "illustration"}
SourceKind = Literal["gazette", "india_code", "concordance_official", "judgment", "commentary", "other"]


class SourceRecord(BaseModel):
    source_id: str
    title: str
    kind: SourceKind
    authority_rank: int = Field(ge=1, le=4)
    sha256: str | None = None
    retrieved_on: str | None = None
    publisher: str | None = None
    url: str | None = None
    local_path: str | None = None
    citation: str | None = None
    license_note: str | None = None


class Act(BaseModel):
    code: str
    name: str
    year: int
    text_as_of: str
    act_number: str | None = None
    commenced_on: str | None = None
    repealed_on: str | None = None
    jurisdiction_note: str | None = None


class Chapter(BaseModel):
    uid: str
    act: str
    number: str
    title: str
    order: int


class Section(BaseModel):
    uid: str
    act: str
    number: str
    sort_key: list = Field(default_factory=list)
    title: str
    chapter_uid: str | None = None
    status: SectionStatus = "in_force"
    source_id: str
    page_start: int | None = None
    page_end: int | None = None
    text: str = ""
    amendment_notes: list[str] = Field(default_factory=list)
    status_note: str | None = None
    status_source_id: str | None = None


class Provision(BaseModel):
    uid: str
    section_uid: str
    parent_uid: str               # section uid or provision uid
    kind: ProvisionKind
    label: str                    # "(1)", "(a)", "Explanation 1", "Exception 2", "body"
    text: str
    order: int
    in_force: bool = True
    status_note: str | None = None
    qualifies: list[str] = Field(default_factory=list)   # QUALIFIES targets (qualifiers only)
    page: int | None = None

    @property
    def is_qualifier(self) -> bool:
        return self.kind in QUALIFIER_KINDS


class ActTree(BaseModel):
    act: Act
    chapters: list[Chapter] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    provisions: list[Provision] = Field(default_factory=list)


class Block(BaseModel):
    source_id: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float | None = None
    bold: bool = False
    kind: Literal["text", "footnote", "header", "devanagari"] = "text"


class CrossRef(BaseModel):
    from_uid: str
    to_uid: str
    raw_text: str
    resolved_by: Literal["regex", "manual"] = "regex"


class Chunk(BaseModel):
    chunk_id: str
    provision_uid: str | None = None
    section_uid: str | None = None
    source_id: str
    act: str | None = None
    embed_text: str
    display_text: str
    kind: str
    tokens: int
    corpus_variant: Literal["text", "scan", "sec"] = "text"
    label: Literal["TextChunk", "ScanChunk", "SecondaryChunk"] = "TextChunk"
    embedding_model: str | None = None
    text_sha256: str
    embedding: list[float] | None = None
    part: int = 0
    n_parts: int = 1
    para_range: str | None = None
    discusses: list[str] = Field(default_factory=list)
    ocr_confidence: float | None = None
