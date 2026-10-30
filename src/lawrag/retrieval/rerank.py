"""Optional reranker (controlled experiment only, SDD 3.4.6 / 7.6). Disabled by default; on any
failure the RRF order is kept."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from lawrag.graph.store import GraphStore
from lawrag.logging_setup import get_logger
from lawrag.models.qa import Candidate

log = get_logger("retrieval.rerank")


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...


def _text(c: Candidate, store: GraphStore) -> str:
    p = store.get_provision(c.uid)
    if p:
        return p.text
    s = store.get_section(c.section_uid)
    return s.text if s else ""


class _Order(BaseModel):
    order: list[str]


class LLMReranker:
    def __init__(self, llm, store: GraphStore):
        self.llm, self.store = llm, store

    def rerank(self, query, candidates, top_n):
        from lawrag.llm.client import Message, load_prompt
        pinned = [c for c in candidates if c.pinned]
        pool = [c for c in candidates if not c.pinned]
        listing = "\n".join(f"- {c.uid}: {_text(c, self.store)[:400]}" for c in pool)
        try:
            res = self.llm.complete(load_prompt("rerank"), [Message(role="user", content=f"QUESTION: {query}\n"
                                                                    f"CANDIDATES:\n{listing}")], schema=_Order)
            pos = {u: i for i, u in enumerate(res.parsed.order)}
            pool = sorted(pool, key=lambda c: (pos.get(c.uid, len(pos)), -c.fused_score))
        except Exception as exc:
            log.warning(f"reranker failed, keeping RRF order: {exc}")
        return pinned + pool[:top_n]


class CrossEncoderReranker:
    def __init__(self, store: GraphStore, model: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder  # optional dependency
        self.model, self.store = CrossEncoder(model, device="cpu"), store

    def rerank(self, query, candidates, top_n):
        pinned = [c for c in candidates if c.pinned]
        pool = [c for c in candidates if not c.pinned]
        try:
            scores = self.model.predict([(query, _text(c, self.store)) for c in pool])
            pool = [c for _, c in sorted(zip(scores, pool), key=lambda x: (-float(x[0]), x[1].uid))]
        except Exception as exc:
            log.warning(f"cross-encoder failed, keeping RRF order: {exc}")
        return pinned + pool[:top_n]
