"""Loader idempotency, SUCCEEDED_BY derivation, audit rules detect seeded violations (SDD 5.1 #6-#7)."""
import os

import pytest

from lawrag.graph.audit import audit
from lawrag.graph.data import snapshot_hash
from lawrag.graph.loader import build_from_seed
from lawrag.index.embedder import make_embedder
from lawrag.models.mapping import Endpoint


def test_reload_gives_identical_hash(cfg, seed_data):
    again = build_from_seed(cfg, make_embedder(cfg.embedding))
    assert snapshot_hash(again) == snapshot_hash(seed_data)


def test_succeeded_by_only_from_verified(seed_data):
    edges = seed_data.succeeded_by()
    mids = {e[2] for e in edges}
    statuses = {m.mapping_id: m.status for m in seed_data.mappings}
    assert mids and all(statuses[m] == "verified" for m in mids)
    assert ("IPC:498A", "BNS:85", "MAP-0004", "SPLIT") in edges
    assert not any(e[2] in ("MAP-0008", "MAP-0010") for e in edges)  # disputed / rejected


def test_seed_audit_passes(seed_data):
    rep = audit(seed_data, active_model="hashing-512")
    assert rep.status == "pass", [(r.rule, r.details) for r in rep.rules if not r.passed]
    assert any("SEED-DRAFT" in w for r in rep.rules for w in r.warnings)  # unreviewed seed flagged


@pytest.mark.parametrize("mutate,rule", [
    (lambda d: setattr(d.mappings[0], "evidence", []), "G5"),
    (lambda d: d.mappings[3].to.pop(), "G6"),                       # SPLIT with one target
    (lambda d: setattr(d.sections[0], "chapter_uid", "IPC:CH-NONE"), "G2"),
    (lambda d: setattr(d.provisions[0], "parent_uid", "IPC:9999"), "G3"),
    (lambda d: d.chunks.clear(), "G4"),
    (lambda d: setattr(next(s for s in d.sections if s.status == "struck_down"), "status_source_id", None), "G10"),
    (lambda d: setattr(d.chunks[0], "embedding_model", "other"), "G11"),
    (lambda d: d.meta.__setitem__("stored_succeeded_by", []), "G7"),
])
def test_audit_detects_violations(seed_data, mutate, rule):
    d = seed_data.model_copy(deep=True)
    mutate(d)
    rep = audit(d, active_model="hashing-512")
    failing = {r.rule for r in rep.rules if not r.passed}
    assert rule in failing


def test_g1_expected_counts(seed_data):
    rep = audit(seed_data, expected={"BNS": ["85", "86", "100", "999"]}, active_model="hashing-512")
    g1 = next(r for r in rep.rules if r.rule == "G1")
    assert not g1.passed and "999" in g1.details[0]


@pytest.mark.neo4j
def test_neo4j_smoke_and_roundtrip(seed_data):
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        pytest.skip("NEO4J_PASSWORD not set")
    from lawrag.errors import GraphUnavailable
    from lawrag.graph.neo4j_store import Neo4jGraphStore
    try:
        s = Neo4jGraphStore(os.environ.get("NEO4J_URI", "bolt://localhost:7687"), "neo4j", pw)
    except GraphUnavailable:
        pytest.skip("Neo4j is not running")
    assert s.run("ping")[0]["ok"] == 1
    s.write(seed_data)
    assert s.section_ids() == {x.uid for x in seed_data.sections}
    assert s.snapshot_hash() == snapshot_hash(seed_data)
    s.write(seed_data)
    assert s.snapshot_hash() == snapshot_hash(seed_data)  # idempotent reload
    hits = s.vector_search(seed_data.chunks[0].embedding, 3, model=seed_data.chunks[0].embedding_model)
    assert hits and hits[0][0].chunk_id == seed_data.chunks[0].chunk_id
    rep = audit(s.export(), active_model=seed_data.chunks[0].embedding_model)
    assert rep.status == "pass"
    s.close()


def test_endpoint_model():
    assert Endpoint(uid="IPC:302").role == "primary"
