"""Neo4j backend: named queries over Bolt; batched MERGE loading in the SDD 3.2.8 order."""
from __future__ import annotations

import re
from pathlib import Path

from lawrag.errors import GraphUnavailable, RetrievalError
from lawrag.graph.data import GraphData, snapshot_hash
from lawrag.legal import uid as U
from lawrag.logging_setup import get_logger
from lawrag.models.legal import Act, Chapter, Chunk, CrossRef, Provision, Section, SourceRecord
from lawrag.models.mapping import MappingRecord

log = get_logger("graph.neo4j")
_HERE = Path(__file__).parent
BATCH = 500


def load_named_queries(path: Path = _HERE / "queries.cypher") -> dict[str, str]:
    queries: dict[str, str] = {}
    name, buf = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^//\s*name:\s*(\w+)", line)
        if m:
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = m.group(1), []
        elif name and not line.startswith("//"):
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


def vector_index_name(label: str, model: str) -> str:
    return f"{label.lower()}_vec_{re.sub(r'[^a-z0-9]', '', model.lower())}"[:60]


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class Neo4jGraphStore:
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 embedding_model: str | None = None):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover
            raise GraphUnavailable("the 'neo4j' package is not installed (pip install lawrag[neo4j])") from exc
        self.database = database
        self.embedding_model = embedding_model
        self.q = load_named_queries()
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
        except Exception as exc:
            raise GraphUnavailable(f"cannot connect to Neo4j at {uri}: {exc}") from exc

    def close(self) -> None:
        self.driver.close()

    def run(self, query_name: str, **params) -> list[dict]:
        """Run a named query. Only names from queries.cypher are accepted."""
        if query_name not in self.q:
            raise KeyError(f"unknown named query {query_name}")
        return self._exec(self.q[query_name], params)

    def _exec(self, cypher: str, params: dict | None = None) -> list[dict]:
        try:
            with self.driver.session(database=self.database) as s:
                return [r.data() for r in s.run(cypher, params or {})]
        except Exception as exc:
            if "ServiceUnavailable" in type(exc).__name__:
                raise GraphUnavailable(str(exc)) from exc
            raise

    # -- reads ---------------------------------------------------------------------------------
    def section_ids(self) -> set[str]:
        return {r["uid"] for r in self.run("section_ids")}

    def exists(self, uid: str) -> bool:
        return bool(self.run("node_exists", uid=uid)[0]["exists"])

    def get_act(self, code: str) -> Act | None:
        r = self.run("get_act", code=code)
        return Act(**r[0]["act"]) if r else None

    @staticmethod
    def _section(d: dict) -> Section:
        d = dict(d)
        d["sort_key"] = list(U.sort_key(d.get("number", "")))
        return Section(**d)

    def get_section(self, uid: str) -> Section | None:
        r = self.run("get_section", uid=uid)
        return self._section(r[0]["section"]) if r else None

    def get_provision(self, uid: str) -> Provision | None:
        r = self.run("get_provision", uid=uid)
        return Provision(**r[0]["provision"]) if r else None

    def provisions_of(self, section_uid: str) -> list[Provision]:
        return [Provision(**r["provision"]) for r in self.run("provisions_of", section_uid=section_uid)]

    def qualifiers_of(self, uid: str) -> list[Provision]:
        p = self.get_provision(uid)
        if p and p.is_qualifier:
            return []
        targets = [uid] + (U.ancestors(uid) if U.is_valid(uid) else [])
        if p:
            targets.append(p.section_uid)
        return [Provision(**r["provision"]) for r in self.run("qualifiers_of", targets=sorted(set(targets)))]

    @staticmethod
    def _mapping(r: dict) -> MappingRecord:
        m = dict(r["mapping"])
        m["from"] = [x for x in r["froms"] if x.get("uid")]
        m["to"] = [x for x in r["tos"] if x.get("uid")]
        m["evidence"] = [_clean(e) for e in r["evidence"] if e.get("source_id")]
        return MappingRecord.model_validate(m)

    def mappings_for(self, uid: str, direction: str | None = None) -> list[MappingRecord]:
        uids = {uid, U.section_of(uid)} if U.is_valid(uid) else {uid}
        if U.is_section_uid(uid):
            uids |= {p.uid for p in self.provisions_of(uid)}
        out = [self._mapping(r) for r in self.run("mappings_for", uids=sorted(uids))]
        if direction == "ipc_to_bns":
            out = [m for m in out if any(u in uids for u in m.from_uids)]
        elif direction == "bns_to_ipc":
            out = [m for m in out if any(u in uids for u in m.to_uids)]
        return out

    def get_mapping(self, mapping_id: str) -> MappingRecord | None:
        r = self.run("get_mapping", mapping_id=mapping_id)
        return self._mapping(r[0]) if r and r[0]["mapping"] else None

    def refers_to(self, uid: str) -> list[str]:
        return [r["uid"] for r in self.run("refers_to", uid=uid)]

    def succeeded_by(self, uid: str) -> list[tuple[str, str, str]]:
        uids = [uid] + ([U.section_of(uid)] if U.is_valid(uid) else [])
        return sorted({(r["uid"], r["mapping_id"], r["type"]) for r in self.run("succeeded_by", uids=uids)})

    def get_source(self, source_id: str) -> SourceRecord | None:
        r = self.run("get_source", source_id=source_id)
        return SourceRecord(**_clean(r[0]["source"])) if r else None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        r = self.run("get_chunk", chunk_id=chunk_id)
        return Chunk(**_clean(r[0]["chunk"])) if r else None

    def chunks_for(self, uid: str) -> list[Chunk]:
        return [Chunk(**_clean(r["chunk"])) for r in self.run("chunks_for", uid=uid)]

    def all_chunks(self, label: str | None = None) -> list[Chunk]:
        return [Chunk(**_clean(r["chunk"])) for r in self.run("all_chunks", label=label)]

    def secondary_discussing(self, uid: str) -> list[Chunk]:
        uids = [uid] + ([U.section_of(uid)] if U.is_valid(uid) else [])
        return [Chunk(**_clean(r["chunk"])) for r in self.run("secondary_discussing", uids=uids)]

    def vector_search(self, vec, k, label="TextChunk", act=None, model=None):
        index = vector_index_name(label, model or self.embedding_model or "default")
        try:
            rows = self.run("vector_search", index=index, k=int(k * (3 if act else 1)), vec=list(vec), act=act)
        except Exception as exc:
            raise RetrievalError(f"vector index query failed ({index}): {exc}") from exc
        return [(Chunk(**_clean(r["chunk"])), float(r["score"])) for r in rows][:k]

    # -- writes (loader) ---------------------------------------------------------------------
    def apply_schema(self, dim: int | None = None, model: str | None = None) -> None:
        for stmt in (_HERE / "schema.cypher").read_text(encoding="utf-8").split(";"):
            body = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("//")).strip()
            if body:
                self._exec(body)
        if dim and model:
            for label in ("TextChunk", "SecondaryChunk"):
                name = vector_index_name(label, model)
                self._exec(
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (c:{label}) ON c.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dim)}, "
                    f"`vector.similarity_function`: 'cosine'}}}}")

    def _batched(self, cypher: str, rows: list[dict]) -> None:
        for i in range(0, len(rows), BATCH):
            self._exec(cypher, {"rows": rows[i:i + BATCH]})

    def write(self, data: GraphData, wipe: bool = True) -> None:
        if wipe:
            self.run("wipe")
        dim = next((len(c.embedding) for c in data.chunks if c.embedding), None)
        model = next((c.embedding_model for c in data.chunks if c.embedding_model), None)
        self.apply_schema(dim, model)
        # 1. sources
        self._batched("UNWIND $rows AS r MERGE (s:Source {source_id: r.source_id}) SET s += r",
                      [_clean(s.model_dump()) for s in data.sources])
        # 2. acts, chapters, sections, provisions
        self._batched("UNWIND $rows AS r MERGE (a:Act {code: r.code}) SET a += r",
                      [_clean(a.model_dump()) for a in data.acts])
        self._batched("UNWIND $rows AS r MERGE (c:Chapter {uid: r.uid}) SET c += r "
                      "WITH c, r MATCH (a:Act {code: r.act}) MERGE (a)-[:HAS_CHAPTER {order: r.order}]->(c)",
                      [c.model_dump() for c in data.chapters])
        secs = []
        for s in data.sections:
            d = _clean(s.model_dump())
            d["sort_key"] = f"{int(s.sort_key[0]):06d}{s.sort_key[1]}" if s.sort_key else s.number
            secs.append(d)
        self._batched(
            "UNWIND $rows AS r MERGE (s:Section {uid: r.uid}) SET s += r "
            "WITH s, r MATCH (c:Chapter {uid: r.chapter_uid}) MERGE (c)-[:HAS_SECTION]->(s) "
            "WITH s, r MATCH (src:Source {source_id: r.source_id}) "
            "MERGE (s)-[:EXTRACTED_FROM {page_start: coalesce(r.page_start, -1), page_end: coalesce(r.page_end, -1)}]->(src)",
            secs)
        self._batched(
            "UNWIND $rows AS r MATCH (s:Section {uid: r.uid}) MATCH (src:Source {source_id: r.status_source_id}) "
            "MERGE (s)-[:STATUS_EVIDENCED_BY {note: coalesce(r.status_note, '')}]->(src)",
            [d for d in secs if d.get("status_source_id")])
        self._batched(
            "UNWIND $rows AS r MERGE (p:Provision {uid: r.uid}) SET p += r "
            "WITH p, r MATCH (parent {uid: r.parent_uid}) WHERE parent:Section OR parent:Provision "
            "MERGE (parent)-[:HAS_PROVISION {order: r.order}]->(p)",
            [_clean(p.model_dump()) for p in data.provisions])
        # 3. qualifier edges
        self._batched(
            "UNWIND $rows AS r MATCH (q:Provision {uid: r.q}) MATCH (t {uid: r.t}) "
            "WHERE t:Section OR t:Provision MERGE (q)-[:QUALIFIES {kind: r.kind}]->(t)",
            [{"q": p.uid, "t": t, "kind": p.kind} for p in data.provisions for t in p.qualifies])
        # 4. chunks and embeddings
        for label in ("TextChunk", "ScanChunk", "SecondaryChunk"):
            rows = [_clean(c.model_dump()) for c in data.chunks if c.label == label]
            self._batched(
                f"UNWIND $rows AS r MERGE (c:Chunk {{chunk_id: r.chunk_id}}) SET c += r, c:{label} "
                "WITH c, r OPTIONAL MATCH (p:Provision {uid: r.provision_uid}) "
                "FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (p)-[:HAS_CHUNK]->(c)) "
                "WITH c, r MATCH (src:Source {source_id: r.source_id}) MERGE (c)-[:EXTRACTED_FROM]->(src)",
                rows)
        self._batched(
            "UNWIND $rows AS r MATCH (c:Chunk {chunk_id: r.c}) MATCH (n {uid: r.n}) MERGE (c)-[:DISCUSSES {basis: 'manual'}]->(n)",
            [{"c": c.chunk_id, "n": d} for c in data.chunks for d in c.discusses])
        # 5. cross-references
        self._batched(
            "UNWIND $rows AS r MATCH (a {uid: r.from_uid}) MATCH (b {uid: r.to_uid}) "
            "MERGE (a)-[:REFERS_TO {raw_text: r.raw_text, resolved_by: r.resolved_by}]->(b)",
            [x.model_dump() for x in data.xrefs])
        # 6. mappings and evidence
        hubs = []
        for m in data.mappings:
            d = _clean(m.model_dump(exclude={"from_", "to", "evidence"}))
            hubs.append(d)
        self._batched("UNWIND $rows AS r MERGE (m:Mapping {mapping_id: r.mapping_id}) SET m += r", hubs)
        self._batched(
            "UNWIND $rows AS r MATCH (m:Mapping {mapping_id: r.m}) MATCH (n {uid: r.uid}) "
            "MERGE (n)-[:MAPPED_IN {role: r.role}]->(m)",
            [{"m": m.mapping_id, "uid": e.uid, "role": e.role} for m in data.mappings for e in m.from_])
        self._batched(
            "UNWIND $rows AS r MATCH (m:Mapping {mapping_id: r.m}) MATCH (n {uid: r.uid}) "
            "MERGE (m)-[:MAPS_TO {role: r.role}]->(n)",
            [{"m": m.mapping_id, "uid": e.uid, "role": e.role} for m in data.mappings for e in m.to])
        self._batched(
            "UNWIND $rows AS r MATCH (m:Mapping {mapping_id: r.m}) MATCH (s:Source {source_id: r.source_id}) "
            "MERGE (m)-[e:EVIDENCED_BY {row: coalesce(r.row, -1), page: coalesce(r.page, -1)}]->(s) "
            "SET e.raw_row = r.raw_row, e.extractor = r.extractor",
            [{"m": m.mapping_id, **e.model_dump()} for m in data.mappings for e in m.evidence])
        # 7. derive SUCCEEDED_BY
        self.run("derive_succeeded_by")
        log.info("neo4j load complete", extra={"data": {"sections": len(data.sections),
                                                       "chunks": len(data.chunks)}})

    # -- whole-graph ---------------------------------------------------------------------------
    def export(self) -> GraphData:
        nodes = self.run("export_nodes")
        edges = self.run("export_edges")
        data = GraphData()
        by_type: dict[str, list[dict]] = {}
        for e in edges:
            by_type.setdefault(e["type"], []).append(e)
        mappings: dict[str, dict] = {}
        for n in nodes:
            labels, p = set(n["labels"]), _clean(n["props"])
            if "Act" in labels:
                data.acts.append(Act(**p))
            elif "Chapter" in labels:
                data.chapters.append(Chapter(**p))
            elif "Section" in labels:
                data.sections.append(self._section(p))
            elif "Provision" in labels:
                data.provisions.append(Provision(**p))
            elif "Chunk" in labels:
                data.chunks.append(Chunk(**p))
            elif "Mapping" in labels:
                mappings[p["mapping_id"]] = {**p, "from": [], "to": [], "evidence": []}
            elif "Source" in labels:
                data.sources.append(SourceRecord(**p))
        for e in by_type.get("MAPPED_IN", []):
            mappings[e["dst"]]["from"].append({"uid": e["src"], "role": e["props"].get("role", "primary")})
        for e in by_type.get("MAPS_TO", []):
            mappings[e["src"]]["to"].append({"uid": e["dst"], "role": e["props"].get("role", "primary")})
        for e in by_type.get("EVIDENCED_BY", []):
            pr = e["props"]
            mappings[e["src"]]["evidence"].append({
                "source_id": e["dst"], "page": None if pr.get("page") == -1 else pr.get("page"),
                "row": None if pr.get("row") == -1 else pr.get("row"), "raw_row": pr.get("raw_row"),
                "extractor": pr.get("extractor") or "manual"})
        data.mappings = [MappingRecord.model_validate(m) for _, m in sorted(mappings.items())]
        data.xrefs = [CrossRef(from_uid=e["src"], to_uid=e["dst"], raw_text=e["props"].get("raw_text", ""),
                               resolved_by=e["props"].get("resolved_by", "regex"))
                      for e in by_type.get("REFERS_TO", [])]
        data.meta["stored_succeeded_by"] = sorted(
            (e["src"], e["dst"], e["props"].get("mapping_id"), e["props"].get("type"))
            for e in by_type.get("SUCCEEDED_BY", []))
        data.meta["has_section_edges"] = [(e["src"], e["dst"]) for e in by_type.get("HAS_SECTION", [])]
        data.meta["extracted_from"] = [(e["src"], e["dst"]) for e in by_type.get("EXTRACTED_FROM", [])]
        return data

    def snapshot_hash(self) -> str:
        return snapshot_hash(self.export())

    def health(self) -> dict:
        try:
            self.run("ping")
            return {"status": "ok", "backend": "neo4j"}
        except Exception as exc:
            return {"status": "down", "backend": "neo4j", "error": str(exc)}
