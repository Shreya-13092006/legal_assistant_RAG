"""Assemble GraphData from act trees, reviewed mappings, chunks and sources, then write it to a
store (SDD 3.2.8). Loading is deterministic: the same inputs give the same snapshot hash."""
from __future__ import annotations

from pathlib import Path

from lawrag.config import RunConfig
from lawrag.errors import ValidationError
from lawrag.graph.data import GraphData
from lawrag.index.chunker import chunk_secondary, chunk_tree
from lawrag.index.embedder import CachedEmbedder
from lawrag.ingest import seed as seed_io
from lawrag.ingest.xref_extractor import extract_all
from lawrag.logging_setup import get_logger
from lawrag.models.legal import ActTree, SourceRecord
from lawrag.models.mapping import FINAL_STATUSES, MappingRecord

log = get_logger("graph.loader")


def validate_mappings(mappings: list[MappingRecord], known_uids: set[str], known_sources: set[str],
                      allow_proposed: bool = True) -> None:
    errors = []
    for m in mappings:
        for u in m.from_uids + m.to_uids:
            if u not in known_uids:
                errors.append(f"{m.mapping_id}: unknown uid {u}")
        for e in m.evidence:
            if e.source_id not in known_sources:
                errors.append(f"{m.mapping_id}: unknown source {e.source_id}")
        if m.status in FINAL_STATUSES and not (m.reviewed_by and m.reviewed_on):
            errors.append(f"{m.mapping_id}: status {m.status} needs reviewed_by and reviewed_on")
        if m.status == "verified" and not m.evidence:
            errors.append(f"{m.mapping_id}: verified mapping without evidence")
        if m.status == "proposed" and not allow_proposed:
            errors.append(f"{m.mapping_id}: proposed rows are not allowed in this graph")
    if errors:
        raise ValidationError("; ".join(errors), errors=errors)


def assemble(trees: list[ActTree], mappings: list[MappingRecord], sources: list[SourceRecord],
             cfg: RunConfig, embedder: CachedEmbedder | None, secondary: list[dict] | None = None) -> GraphData:
    data = GraphData(sources=sorted(sources, key=lambda s: s.source_id))
    unresolved: list[dict] = []
    for t in trees:
        data.acts.append(t.act)
        data.chapters += t.chapters
        data.sections += t.sections
        data.provisions += t.provisions
        refs, unres = extract_all(t)
        data.xrefs += refs
        unresolved += unres
        data.chunks += chunk_tree(t, cfg.chunking)
    if secondary:
        data.chunks += chunk_secondary(secondary, cfg.chunking)
    known = {s.uid for s in data.sections} | {p.uid for p in data.provisions}
    validate_mappings(mappings, known, {s.source_id for s in sources})
    data.mappings = sorted(mappings, key=lambda m: m.mapping_id)
    if embedder is not None and data.chunks:
        vecs = embedder.encode([c.embed_text for c in data.chunks])
        for c, v in zip(data.chunks, vecs):
            c.embedding = [round(float(x), 6) for x in v]
            c.embedding_model = embedder.model_id
    data.meta = {"unresolved_xrefs": unresolved, "corpus_hash": data.corpus_hash(),
                 "embedding_model": embedder.model_id if embedder else None,
                 "embedding_dim": embedder.dim if embedder else None}
    return data


def build_from_seed(cfg: RunConfig, embedder: CachedEmbedder | None) -> GraphData:
    seed_dir = cfg.path(cfg.graph.seed_dir)
    return assemble(seed_io.load_seed_trees(seed_dir), seed_io.load_seed_mappings(seed_dir),
                    seed_io.load_seed_sources(seed_dir), cfg, embedder, seed_io.load_seed_secondary(seed_dir))


def build_from_processed(cfg: RunConfig, embedder: CachedEmbedder | None, processed: Path,
                         approved_csv: Path, sources_json: Path) -> GraphData:
    """M2 path: act trees from the PDF pipeline + approved_mappings.csv + sources.json."""
    import json

    from lawrag.ingest.review_io import read_approved

    trees = [ActTree.model_validate_json(p.read_text(encoding="utf-8"))
             for p in sorted(processed.glob("*.tree.json"))]
    sources = [SourceRecord(**s) for s in json.loads(sources_json.read_text(encoding="utf-8"))]
    known = {s.uid for t in trees for s in t.sections} | {p.uid for t in trees for p in t.provisions}
    mappings = read_approved(approved_csv, known, {s.source_id for s in sources})
    return assemble(trees, mappings, sources, cfg, embedder)
