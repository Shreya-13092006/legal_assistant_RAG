"""Dependency container: builds the store, embedder, BM25 index, analyzer and LLM clients once per
process (SDD 2.12 process model) and hands them to services, the API, CLI and evaluation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from lawrag.config import PROJECT_ROOT, RunConfig
from lawrag.graph.data import GraphData
from lawrag.graph.store import GraphStore, MemoryGraphStore
from lawrag.index.bm25 import BM25Index
from lawrag.index.embedder import CachedEmbedder, make_embedder
from lawrag.index.tokenizer import TOKENIZER_VERSION
from lawrag.llm.client import LLMRegistry
from lawrag.logging_setup import get_logger
from lawrag.retrieval.query_analyzer import QueryAnalyzer

log = get_logger("services.container")


@dataclass
class Deps:
    cfg: RunConfig
    store: GraphStore
    embedder: CachedEmbedder
    bm25: BM25Index
    llm: LLMRegistry
    analyzer: QueryAnalyzer
    corpus_hash: str
    snapshot_hash: str
    commencement: date
    extras: dict = field(default_factory=dict)

    def with_config(self, cfg: RunConfig) -> "Deps":
        """Same loaded resources, different run config (used by the evaluation runner)."""
        if cfg.embedding.model != self.cfg.embedding.model:
            raise ValueError("changing the embedding model needs a rebuilt graph; build new Deps instead")
        d = Deps(cfg=cfg, store=self.store, embedder=self.embedder, bm25=self.bm25,
                 llm=LLMRegistry(cfg.llm, PROJECT_ROOT, self.llm._override or None), analyzer=self.analyzer,
                 corpus_hash=self.corpus_hash, snapshot_hash=self.snapshot_hash, commencement=self.commencement,
                 extras=dict(self.extras))
        d.analyzer = QueryAnalyzer(cfg, self.analyzer.section_ids, self.analyzer.provision_ids, d.llm.get("analyzer_fallback"))
        return d


def load_graph_data(cfg: RunConfig, embedder: CachedEmbedder, rebuild: bool = False) -> GraphData:
    from lawrag.graph.loader import build_from_seed
    snap = cfg.path(cfg.graph.snapshot_path)
    if snap.exists() and not rebuild:
        data = GraphData.load(snap)
        if data.meta.get("embedding_model") == embedder.model_id:
            return data
        log.warning("graph snapshot was embedded with a different model; rebuilding from seed")
    data = build_from_seed(cfg, embedder)
    data.save(snap)
    return data


def make_store(cfg: RunConfig, embedder: CachedEmbedder, rebuild: bool = False) -> GraphStore:
    if cfg.graph.backend == "neo4j":
        from lawrag.graph.neo4j_store import Neo4jGraphStore
        return Neo4jGraphStore(cfg.neo4j.uri, cfg.neo4j.user, os.environ.get("NEO4J_PASSWORD", ""),
                               cfg.neo4j.database, embedding_model=embedder.model_id)
    return MemoryGraphStore(load_graph_data(cfg, embedder, rebuild))


def build_bm25(store: GraphStore, cache_dir: Path, corpus_hash: str) -> BM25Index:
    d = cache_dir / f"{corpus_hash[:16]}-{TOKENIZER_VERSION}"
    idx = BM25Index.load(d)
    if idx is not None and idx.key == corpus_hash:
        return idx
    idx = BM25Index().build([(c.chunk_id, c.embed_text) for c in store.all_chunks()], key=corpus_hash)
    idx.save(d)
    return idx


def build_deps(cfg: RunConfig, *, store: GraphStore | None = None, llm_adapters: dict | None = None,
               rebuild: bool = False) -> Deps:
    embedder = make_embedder(cfg.embedding, cfg.path(cfg.embedding.cache_path))
    store = store or make_store(cfg, embedder, rebuild)
    data = store.export()
    corpus_hash = data.corpus_hash()
    bm25 = build_bm25(store, cfg.path("cache/bm25"), corpus_hash)
    llm = LLMRegistry(cfg.llm, PROJECT_ROOT, llm_adapters)
    analyzer = QueryAnalyzer(cfg, store.section_ids(), {p.uid for p in data.provisions}, llm.get("analyzer_fallback"))
    return Deps(cfg=cfg, store=store, embedder=embedder, bm25=bm25, llm=llm, analyzer=analyzer,
                corpus_hash=corpus_hash, snapshot_hash=store.snapshot_hash(),
                commencement=date.fromisoformat(cfg.project.bns_commencement))
