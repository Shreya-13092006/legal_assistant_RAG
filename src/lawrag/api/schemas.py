"""Request/response schemas for the HTTP API (SDD 3.7.1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lawrag.models.qa import AskOptions


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    options: AskOptions | None = None


class CompareRequest(BaseModel):
    left_uid: str
    right_uid: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["vector", "bm25", "graph", "hybrid"] = "hybrid"
    k: int = Field(10, ge=1, le=100)
    act: Literal["IPC", "BNS"] | None = None


class SearchHit(BaseModel):
    uid: str
    score: float
    ranks: dict[str, int]
    via: list[str]


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1)
    context_uids: list[str]
    question: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict = Field(default_factory=dict)
