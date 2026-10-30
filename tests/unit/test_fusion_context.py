"""RRF (hand-computed), pinning, dedup; context builder budget/qualifiers/determinism."""
import pytest

from lawrag.models.qa import Candidate, QueryAnalysis
from lawrag.retrieval.context_builder import build_context
from lawrag.retrieval.fusion import rrf
from lawrag.retrieval.mapping_facts import facts_for


def C(uid, via, rank, pinned=False, kind="body"):
    return Candidate(uid=uid, section_uid=uid.split(":body")[0].split("(")[0], act=uid[:3], kind=kind, via=[via],
                     ranks={via: rank}, pinned=pinned)


def test_rrf_hand_computed():
    lists = {"vector": [C("BNS:1:body", "vector", 1), C("BNS:2:body", "vector", 2)],
             "bm25": [C("BNS:2:body", "bm25", 1), C("BNS:3:body", "bm25", 2)]}
    out = rrf(lists, k_rrf=60)
    scores = {c.uid: c.fused_score for c in out}
    assert scores["BNS:2:body"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["BNS:1:body"] == pytest.approx(1 / 61)
    assert [c.uid for c in out] == ["BNS:2:body", "BNS:1:body", "BNS:3:body"]
    assert out[0].via == ["vector", "bm25"]


def test_pinned_first_and_truncation():
    lists = {"id_lookup": [C("IPC:9:body", "id_lookup", 1, pinned=True)],
             "vector": [C(f"BNS:{i}:body", "vector", i) for i in range(1, 20)]}
    out = rrf(lists, pinned=["IPC:9:body"], top_n=3)
    assert out[0].uid == "IPC:9:body" and len(out) == 4


def test_illustration_downweight():
    lists = {"vector": [C("BNS:1:Ill(a)", "vector", 1, kind="illustration"), C("BNS:1:body", "vector", 2)]}
    out = rrf(lists, illustration_weight=0.5)
    assert out[0].uid == "BNS:1:body"
    out = rrf(lists, illustration_weight=0.5, intent="scenario")
    assert out[0].uid == "BNS:1:Ill(a)"


def test_context_qualifiers_mapping_budget_determinism(store):
    a = QueryAnalysis(intent="mapping", section_ids=["IPC:498A"])
    fused = [C("IPC:498A:body", "id_lookup", 1, pinned=True), C("BNS:100:body", "vector", 1),
             C("BNS:318(4)", "vector", 2)]
    for c in fused:
        c.section_uid = c.uid.split(":body")[0].split("(")[0]
    facts = facts_for("IPC:498A", store)
    ctx = build_context(fused, a, store, 3500, facts)
    labels = [b.label for b in ctx.blocks]
    assert labels[0] == "MAPPING MAP-0004"
    assert "IPC:498A:Expl1" in labels  # qualifier attached to its rule
    assert ctx.render() == build_context(fused, a, store, 3500, facts).render()
    small = build_context(fused, a, store, 260, facts)
    kept = [b.label for b in small.blocks]
    assert "IPC:498A:body" in kept and "IPC:498A:Expl1" in kept and "MAPPING MAP-0004" in kept
    assert small.dropped  # unpinned material dropped first


def test_context_marks_not_in_force(store):
    a = QueryAnalysis(intent="lookup")
    ctx = build_context([C("BNS:106(2)", "vector", 1)], a, store, 3500)
    assert "NOT IN FORCE" in ctx.render()
