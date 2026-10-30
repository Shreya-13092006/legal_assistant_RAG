"""LawService: get_section, get_mappings, compare, search (SDD 3.8.7). Shared by API, CLI, eval and MCP."""
from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from lawrag.errors import BadIdentifier, Conflict, NotFound
from lawrag.legal import uid as U
from lawrag.models.mapping import MappingFact
from lawrag.models.qa import Candidate, QueryAnalysis
from lawrag.retrieval.fusion import rrf, single
from lawrag.retrieval.graph_expand import graph_expand
from lawrag.retrieval.mapping_facts import facts_for
from lawrag.retrieval.retrievers import BM25Retriever, IdLookupRetriever, VectorRetriever
from lawrag.services.container import Deps


class ProvisionView(BaseModel):
    uid: str
    kind: str
    label: str
    text: str
    in_force: bool
    status_note: str | None = None
    qualifies: list[str] = Field(default_factory=list)


class SectionView(BaseModel):
    uid: str
    act: str
    number: str
    title: str
    status: str
    status_note: str | None = None
    provisions: list[ProvisionView]
    amendment_notes: list[str] = Field(default_factory=list)
    source: dict


class ComparePair(BaseModel):
    left: SectionView
    right: SectionView
    mapping: MappingFact | None = None
    diff_segments: list[dict]


class Comparison(BaseModel):
    pairs: list[ComparePair]
    note: str = ""


class MappingsView(BaseModel):
    uid: str
    direction: str | None
    mappings: list[MappingFact]
    coverage: str


def _check_uid(uid: str) -> str:
    uid = uid.strip()
    if not U.is_valid(uid) and not U.is_section_uid(uid):
        raise BadIdentifier(f"malformed uid {uid!r}; expected e.g. IPC:302 or BNS:103(1)")
    return uid


def text_diff(a: str, b: str) -> list[dict]:
    """Deterministic word-level diff: [{op: equal|insert|delete|replace, left, right}]."""
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        out.append({"op": op, "left": " ".join(aw[i1:i2]), "right": " ".join(bw[j1:j2])})
    return out


class LawService:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.store = deps.store

    def get_section(self, uid: str, include_qualifiers: bool = True) -> SectionView:
        uid = _check_uid(uid)
        sec = self.store.get_section(U.section_of(uid))
        if sec is None:
            act = U.act_of(uid)
            raise NotFound(f"{uid} does not exist in the {act} as indexed (cutoff "
                           f"{self.deps.cfg.project.cutoff.get(act, 'n/a')})")
        provs = self.store.provisions_of(sec.uid)
        if uid != sec.uid:
            if not self.store.exists(uid):
                raise NotFound(f"{uid} does not exist")
            keep = {uid} | ({q.uid for q in self.store.qualifiers_of(uid)} if include_qualifiers else set())
            provs = [p for p in provs if p.uid in keep]
        elif not include_qualifiers:
            provs = [p for p in provs if not p.is_qualifier]
        src = self.store.get_source(sec.source_id)
        return SectionView(
            uid=sec.uid, act=sec.act, number=sec.number, title=sec.title, status=sec.status,
            status_note=sec.status_note, amendment_notes=sec.amendment_notes,
            provisions=[ProvisionView(uid=p.uid, kind=p.kind, label=p.label, text=p.text, in_force=p.in_force,
                                      status_note=p.status_note, qualifies=p.qualifies) for p in provs],
            source={"source_id": sec.source_id, "title": src.title if src else None, "page": sec.page_start,
                    "url": src.url if src else None})

    def get_mappings(self, uid: str, direction: str | None = None, include_unverified: bool = False) -> MappingsView:
        uid = _check_uid(uid)
        if not self.store.exists(uid):
            raise NotFound(f"{uid} does not exist")
        direction = direction or ("ipc_to_bns" if U.act_of(uid) == "IPC" else "bns_to_ipc")
        # disputed/unresolved hubs are always shown as caveats; include_unverified adds nothing beyond
        # final statuses because proposed rows never leave the review queue
        facts = facts_for(uid, self.store, direction)
        if not include_unverified:
            facts = [f for f in facts if f.status in ("verified", "disputed", "unresolved")]
        return MappingsView(uid=uid, direction=direction, mappings=facts,
                            coverage="reviewed" if facts else "unreviewed")

    def compare(self, left_uid: str, right_uid: str | None = None) -> Comparison:
        left = self.get_section(left_uid)
        pairs = []
        if right_uid:
            right = self.get_section(right_uid)
            mf = next((f for f in facts_for(left.uid, self.store)
                       if any(U.section_of(t) == right.uid for t in f.to_uids + f.from_uids)), None)
            pairs.append(ComparePair(left=left, right=right, mapping=mf,
                                     diff_segments=text_diff(_body(left), _body(right))))
            return Comparison(pairs=pairs)
        facts = [f for f in facts_for(left.uid, self.store) if f.status == "verified" and f.to_uids + f.from_uids]
        targets = []
        for f in facts:
            side = f.to_uids if any(U.section_of(u) == left.uid for u in f.from_uids) else f.from_uids
            targets += [(t, f) for t in side]
        if not targets:
            raise Conflict(f"no verified mapping for {left.uid}; pass right_uid to compare explicitly")
        for t, f in targets:
            right = self.get_section(t)
            pairs.append(ComparePair(left=left, right=right, mapping=f, diff_segments=text_diff(_body(left), _body(right))))
        return Comparison(pairs=pairs, note="Targets taken from verified mappings. The diff is a mechanical word "
                                            "diff, not a legal analysis.")

    def search(self, query: str, mode: str = "hybrid", k: int = 10, act: str | None = None) -> list[Candidate]:
        d = self.deps
        a: QueryAnalysis = d.analyzer.analyze(query)
        if act:
            a.acts = [act]
        lists: dict[str, list[Candidate]] = {}
        if mode in ("vector", "hybrid", "graph"):
            lists["vector"] = VectorRetriever(d.store, d.embedder, True).retrieve(a, k)
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = BM25Retriever(d.store, d.bm25, True).retrieve(a, k)
        if mode in ("hybrid", "graph"):
            lists["id_lookup"] = IdLookupRetriever(d.store).retrieve(a)
            prelim = rrf({kk: v for kk, v in lists.items() if v}, d.cfg.fusion.k_rrf)
            seeds = [c for c in prelim if c.pinned] + [c for c in prelim if not c.pinned][: d.cfg.retrieval.graph.seeds]
            lists["graph"], _ = graph_expand(seeds, a, d.store, d.cfg.retrieval.graph)
        lists = {kk: v for kk, v in lists.items() if v}
        if len(lists) == 1:
            return single(next(iter(lists.values())), k)
        pinned = [c.uid for c in lists.get("id_lookup", [])]
        return rrf(lists, d.cfg.fusion.k_rrf, d.cfg.fusion.weights, pinned, k, d.cfg.fusion.illustration_weight,
                   a.intent)


def _body(s: SectionView) -> str:
    return "\n".join(p.text for p in s.provisions if p.kind not in ("illustration",))
