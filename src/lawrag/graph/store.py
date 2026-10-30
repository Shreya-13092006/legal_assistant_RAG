"""GraphStore interface and the in-memory backend.

Upper layers depend on the `GraphStore` protocol only (SDD 2.3 layering rule). `MemoryGraphStore`
serves the same queries from a `GraphData` snapshot; it is used in tests and when Neo4j is not
running. `Neo4jGraphStore` (neo4j_store.py) runs the named Cypher queries in queries.cypher.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

import numpy as np

from lawrag.graph.data import GraphData, snapshot_hash
from lawrag.legal import uid as U
from lawrag.models.legal import Act, Chunk, Provision, Section, SourceRecord
from lawrag.models.mapping import MappingRecord


@runtime_checkable
class GraphStore(Protocol):
    backend: str

    def section_ids(self) -> set[str]: ...
    def exists(self, uid: str) -> bool: ...
    def get_act(self, code: str) -> Act | None: ...
    def get_section(self, uid: str) -> Section | None: ...
    def get_provision(self, uid: str) -> Provision | None: ...
    def provisions_of(self, section_uid: str) -> list[Provision]: ...
    def qualifiers_of(self, uid: str) -> list[Provision]: ...
    def mappings_for(self, uid: str, direction: str | None = None) -> list[MappingRecord]: ...
    def get_mapping(self, mapping_id: str) -> MappingRecord | None: ...
    def refers_to(self, uid: str) -> list[str]: ...
    def succeeded_by(self, uid: str) -> list[tuple[str, str, str]]: ...
    def get_source(self, source_id: str) -> SourceRecord | None: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def chunks_for(self, uid: str) -> list[Chunk]: ...
    def all_chunks(self, label: str | None = None) -> list[Chunk]: ...
    def secondary_discussing(self, uid: str) -> list[Chunk]: ...
    def vector_search(self, vec: list[float], k: int, label: str = "TextChunk", act: str | None = None,
                      model: str | None = None) -> list[tuple[Chunk, float]]: ...
    def write(self, data: GraphData) -> None: ...
    def export(self) -> GraphData: ...
    def snapshot_hash(self) -> str: ...
    def health(self) -> dict: ...


def _hub_matches(uid: str, endpoint: str) -> bool:
    """A hub endpoint matches a uid if equal, or one is the section of the other."""
    if uid == endpoint:
        return True
    try:
        return U.section_of(endpoint) == uid or U.section_of(uid) == endpoint
    except ValueError:
        return False


class MemoryGraphStore:
    backend = "memory"

    def __init__(self, data: GraphData | None = None):
        self.write(data or GraphData())

    # -- loading -------------------------------------------------------------------------------
    def write(self, data: GraphData) -> None:
        self.data = data
        self._acts = {a.code: a for a in data.acts}
        self._sections = {s.uid: s for s in data.sections}
        self._provisions = {p.uid: p for p in data.provisions}
        self._by_section: dict[str, list[Provision]] = defaultdict(list)
        for p in sorted(data.provisions, key=lambda p: p.order):
            self._by_section[p.section_uid].append(p)
        self._qual_index: dict[str, list[Provision]] = defaultdict(list)
        for p in data.provisions:
            for t in p.qualifies:
                self._qual_index[t].append(p)
        self._mappings = {m.mapping_id: m for m in data.mappings}
        self._sources = {s.source_id: s for s in data.sources}
        self._chunks = {c.chunk_id: c for c in data.chunks}
        self._chunks_by_uid: dict[str, list[Chunk]] = defaultdict(list)
        self._discusses: dict[str, list[Chunk]] = defaultdict(list)
        for c in data.chunks:
            if c.provision_uid:
                self._chunks_by_uid[c.provision_uid].append(c)
            elif c.section_uid:
                self._chunks_by_uid[c.section_uid].append(c)
            for d in c.discusses:
                self._discusses[d].append(c)
        self._xrefs: dict[str, list[str]] = defaultdict(list)
        for x in data.xrefs:
            self._xrefs[x.from_uid].append(x.to_uid)
        self._succ: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for a, b, mid, t in data.succeeded_by():
            self._succ[a].append((b, mid, t))
        self._matrix_cache: dict[tuple, tuple[list[Chunk], np.ndarray]] = {}

    # -- reads ---------------------------------------------------------------------------------
    def section_ids(self) -> set[str]:
        return set(self._sections)

    def exists(self, uid: str) -> bool:
        return uid in self._sections or uid in self._provisions

    def get_act(self, code: str) -> Act | None:
        return self._acts.get(code)

    def get_section(self, uid: str) -> Section | None:
        return self._sections.get(uid)

    def get_provision(self, uid: str) -> Provision | None:
        return self._provisions.get(uid)

    def provisions_of(self, section_uid: str) -> list[Provision]:
        return list(self._by_section.get(section_uid, []))

    def qualifiers_of(self, uid: str) -> list[Provision]:
        targets = [uid]
        if uid in self._provisions:
            p = self._provisions[uid]
            if p.is_qualifier:
                return []
            targets += U.ancestors(uid) if U.is_valid(uid) else []
            targets.append(p.section_uid)
        out: dict[str, Provision] = {}
        for t in targets:
            for q in self._qual_index.get(t, []):
                # a qualifier aimed at the whole section applies to every rule in it; one aimed at a
                # specific sub-section applies only to that sub-section and its clauses
                out[q.uid] = q
        return sorted(out.values(), key=lambda p: p.order)

    def mappings_for(self, uid: str, direction: str | None = None) -> list[MappingRecord]:
        out = []
        for m in self._mappings.values():
            in_from = any(_hub_matches(uid, e) for e in m.from_uids)
            in_to = any(_hub_matches(uid, e) for e in m.to_uids)
            if direction == "ipc_to_bns" and not in_from:
                continue
            if direction == "bns_to_ipc" and not in_to:
                continue
            if in_from or in_to:
                out.append(m)
        return sorted(out, key=lambda m: m.mapping_id)

    def get_mapping(self, mapping_id: str) -> MappingRecord | None:
        return self._mappings.get(mapping_id)

    def refers_to(self, uid: str) -> list[str]:
        return sorted(set(self._xrefs.get(uid, [])))

    def succeeded_by(self, uid: str) -> list[tuple[str, str, str]]:
        out = list(self._succ.get(uid, []))
        if uid in self._provisions:
            out += self._succ.get(self._provisions[uid].section_uid, [])
        return sorted(set(out))

    def get_source(self, source_id: str) -> SourceRecord | None:
        return self._sources.get(source_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunks_for(self, uid: str) -> list[Chunk]:
        return sorted(self._chunks_by_uid.get(uid, []), key=lambda c: c.part)

    def all_chunks(self, label: str | None = None) -> list[Chunk]:
        cs = sorted(self._chunks.values(), key=lambda c: c.chunk_id)
        return [c for c in cs if label is None or c.label == label]

    def secondary_discussing(self, uid: str) -> list[Chunk]:
        out = self._discusses.get(uid, [])
        if uid in self._provisions:
            out = out + self._discusses.get(self._provisions[uid].section_uid, [])
        return sorted({c.chunk_id: c for c in out}.values(), key=lambda c: c.chunk_id)

    def vector_search(self, vec, k, label="TextChunk", act=None, model=None):
        key = (label, model)
        if key not in self._matrix_cache:
            cs = [c for c in self.all_chunks(label) if c.embedding is not None
                  and (model is None or c.embedding_model == model)]
            mat = np.array([c.embedding for c in cs], dtype=np.float32) if cs else np.zeros((0, 1), np.float32)
            self._matrix_cache[key] = (cs, mat)
        cs, mat = self._matrix_cache[key]
        if not cs:
            return []
        q = np.asarray(vec, dtype=np.float32)
        if mat.shape[1] != q.shape[0]:
            from lawrag.errors import RetrievalError
            raise RetrievalError(f"vector dimension mismatch: index {mat.shape[1]} vs query {q.shape[0]}")
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-12)
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order:
            c = cs[int(i)]
            if act and c.act != act:
                continue
            out.append((c, float(sims[int(i)])))
            if len(out) >= k:
                break
        return out

    # -- whole-graph ---------------------------------------------------------------------------
    def export(self) -> GraphData:
        return self.data

    def snapshot_hash(self) -> str:
        return snapshot_hash(self.data)

    def health(self) -> dict:
        return {"status": "ok", "backend": "memory", "sections": len(self._sections),
                "chunks": len(self._chunks), "mappings": len(self._mappings)}
