"""GraphData: the complete, serializable content of the knowledge graph.

Both graph backends load from and export to this structure, so the loader order (SDD 3.2.8), the
audit (G1-G12) and the snapshot hash are implemented once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from lawrag.models.legal import Act, Chapter, Chunk, CrossRef, Provision, Section, SourceRecord
from lawrag.models.mapping import MappingRecord


class GraphData(BaseModel):
    acts: list[Act] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    provisions: list[Provision] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    mappings: list[MappingRecord] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    xrefs: list[CrossRef] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    def succeeded_by(self) -> list[tuple[str, str, str, str]]:
        """Derived SUCCEEDED_BY edges (from, to, mapping_id, type): verified hubs only (SDD 3.1.4)."""
        out = []
        for m in self.mappings:
            if m.status != "verified":
                continue
            for a in m.from_uids:
                for b in m.to_uids:
                    out.append((a, b, m.mapping_id, m.type))
        return sorted(out)

    def corpus_hash(self) -> str:
        rows = sorted(f"{c.chunk_id}|{c.text_sha256}" for c in self.chunks)
        return hashlib.sha256("\n".join(rows).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(by_alias=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GraphData":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def snapshot_hash(data: GraphData) -> str:
    """sha256 of a canonical, sorted export of node keys, key properties and edges."""
    rows: list[str] = []
    for a in data.acts:
        rows.append(f"Act|{a.code}|{a.text_as_of}")
    for c in data.chapters:
        rows.append(f"Chapter|{c.uid}|{c.title}|HAS_CHAPTER<-{c.act}")
    for s in data.sections:
        rows.append(f"Section|{s.uid}|{s.title}|{s.status}|{s.source_id}|HAS_SECTION<-{s.chapter_uid}")
    for p in data.provisions:
        th = hashlib.sha256(p.text.encode()).hexdigest()[:16]
        rows.append(f"Provision|{p.uid}|{p.kind}|{p.in_force}|{th}|HAS_PROVISION<-{p.parent_uid}")
        for t in sorted(p.qualifies):
            rows.append(f"QUALIFIES|{p.uid}->{t}")
    for c in data.chunks:
        rows.append(f"Chunk|{c.chunk_id}|{c.text_sha256}|{c.embedding_model}")
    for m in data.mappings:
        rows.append(f"Mapping|{m.mapping_id}|{m.type}|{m.status}|{m.reviewed_by}|{m.reviewed_on}")
        rows += [f"MAPPED_IN|{e.uid}->{m.mapping_id}|{e.role}" for e in m.from_]
        rows += [f"MAPS_TO|{m.mapping_id}->{e.uid}|{e.role}" for e in m.to]
        rows += [f"EVIDENCED_BY|{m.mapping_id}->{e.source_id}|{e.page}|{e.row}" for e in m.evidence]
    for s in data.sources:
        rows.append(f"Source|{s.source_id}|{s.sha256}")
    for x in data.xrefs:
        rows.append(f"REFERS_TO|{x.from_uid}->{x.to_uid}")
    for a, b, mid, t in data.succeeded_by():
        rows.append(f"SUCCEEDED_BY|{a}->{b}|{mid}|{t}")
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
