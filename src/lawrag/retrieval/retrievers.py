"""First-stage retrievers: exact-ID lookup, vector, BM25 (SDD 3.4.2-3.4.4).

All retrievers return provision-level `Candidate`s; chunk parts roll up to their provision.
"""
from __future__ import annotations

from typing import Protocol

from lawrag.graph.store import GraphStore
from lawrag.index.bm25 import BM25Index
from lawrag.index.embedder import CachedEmbedder
from lawrag.legal import uid as U
from lawrag.models.legal import Chunk
from lawrag.models.qa import Candidate, QueryAnalysis


class Retriever(Protocol):
    name: str

    def retrieve(self, analysis: QueryAnalysis, k: int) -> list[Candidate]: ...


def _chunk_candidate(c: Chunk, via: str, rank: int, score: float) -> Candidate:
    if c.label == "SecondaryChunk":
        return Candidate(uid=c.chunk_id, section_uid=(c.discusses[0] if c.discusses else c.source_id),
                         act=c.act or "", kind="secondary", chunk_ids=[c.chunk_id], via=[via],
                         ranks={via: rank}, scores={via: score}, secondary=True)
    uid = c.provision_uid or c.section_uid
    return Candidate(uid=uid, section_uid=c.section_uid or U.section_of(uid), act=c.act or U.act_of(uid),
                     kind=c.kind, chunk_ids=[c.chunk_id], via=[via], ranks={via: rank}, scores={via: score})


def rollup(cands: list[Candidate]) -> list[Candidate]:
    """Collapse chunk parts of the same provision; keep best rank; re-rank 1..n."""
    out: dict[str, Candidate] = {}
    for c in cands:
        if c.uid in out:
            out[c.uid].chunk_ids = sorted(set(out[c.uid].chunk_ids + c.chunk_ids))
        else:
            out[c.uid] = c
    ranked = list(out.values())
    for i, c in enumerate(ranked, 1):
        for v in c.ranks:
            c.ranks[v] = i
    return ranked


class IdLookupRetriever:
    name = "id_lookup"

    def __init__(self, store: GraphStore):
        self.store = store

    def retrieve(self, analysis: QueryAnalysis, k: int = 0, extra_ids: list[str] | None = None) -> list[Candidate]:
        out: list[Candidate] = []
        rank = 1
        for uid in list(dict.fromkeys(analysis.section_ids + (extra_ids or []))):
            if not self.store.exists(uid):
                continue
            sec_uid = U.section_of(uid)
            provs = self.store.provisions_of(sec_uid)
            if uid != sec_uid:  # a specific provision was named: it plus its own qualifiers
                provs = [p for p in provs if p.uid == uid]
            if not provs:  # status stub (e.g. struck-down section with no text)
                out.append(Candidate(uid=sec_uid, section_uid=sec_uid, act=U.act_of(sec_uid), kind="status",
                                     via=["id_lookup"], ranks={"id_lookup": rank}, pinned=True))
                rank += 1
            for p in provs:
                if p.is_qualifier and uid == sec_uid:
                    continue  # qualifiers are attached by the context builder
                out.append(Candidate(uid=p.uid, section_uid=sec_uid, act=U.act_of(sec_uid), kind=p.kind,
                                     via=["id_lookup"], ranks={"id_lookup": rank}, scores={"id_lookup": 1.0},
                                     pinned=True))
                rank += 1
        return out


class VectorRetriever:
    name = "vector"

    def __init__(self, store: GraphStore, embedder: CachedEmbedder, act_filter: bool = True,
                 include_secondary: bool = False):
        self.store, self.embedder = store, embedder
        self.act_filter, self.include_secondary = act_filter, include_secondary

    def retrieve(self, analysis: QueryAnalysis, k: int) -> list[Candidate]:
        vec = self.embedder.encode([analysis.rewritten_query], is_query=True)[0].tolist()
        act = analysis.acts[0] if self.act_filter and len(analysis.acts) == 1 and analysis.intent not in (
            "mapping", "comparison") else None
        hits = self.store.vector_search(vec, k, "TextChunk", act=act, model=self.embedder.model_id)
        if self.include_secondary:
            hits += self.store.vector_search(vec, max(3, k // 5), "SecondaryChunk", model=self.embedder.model_id)
            hits.sort(key=lambda h: -h[1])
        return rollup([_chunk_candidate(c, "vector", i, s) for i, (c, s) in enumerate(hits, 1)])


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: GraphStore, index: BM25Index, act_filter: bool = True, include_secondary: bool = False):
        self.store, self.index = store, index
        self.act_filter, self.include_secondary = act_filter, include_secondary
        self._chunks = {c.chunk_id: c for c in store.all_chunks()}

    def retrieve(self, analysis: QueryAnalysis, k: int) -> list[Candidate]:
        act = analysis.acts[0] if self.act_filter and len(analysis.acts) == 1 and analysis.intent not in (
            "mapping", "comparison") else None
        allowed = {cid for cid, c in self._chunks.items()
                   if (self.include_secondary or c.label != "SecondaryChunk") and (act is None or c.act in (act, None))}
        hits = self.index.search(analysis.rewritten_query, k * 2, allowed)
        cands = [_chunk_candidate(self._chunks[cid], "bm25", i, s) for i, (cid, s) in enumerate(hits, 1)
                 if cid in self._chunks]
        return rollup(cands)[:k]
