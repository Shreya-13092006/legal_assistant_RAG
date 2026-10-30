"""ID-extraction table (precision = recall = 1.0 required) and intent rules (SDD 5.1 #11)."""
import pytest

from lawrag.retrieval.query_analyzer import QueryAnalyzer

SECTIONS = {f"IPC:{n}" for n in ["34", "299", "302", "304A", "376", "420", "498A", "120B", "497", "124A", "377", "417"]} | \
           {f"BNS:{n}" for n in ["3", "85", "86", "100", "103", "106", "152", "304", "318", "61"]}
PROVS = {"BNS:103(1)", "BNS:103(2)", "BNS:106(1)", "BNS:106(2)", "IPC:376(2)(a)"}

ID_TABLE = [
    ("What is section 302 IPC?", ["IPC:302"], []),
    ("s. 302 of the Indian Penal Code", ["IPC:302"], []),
    ("sec 302 IPC", ["IPC:302"], []),
    ("u/s 302 IPC", ["IPC:302"], []),
    ("IPC 302 punishment", ["IPC:302"], []),
    ("302 IPC punishment", ["IPC:302"], []),
    ("What replaced IPC 498 A?", ["IPC:498A"], []),
    ("Explain section 120-B IPC", ["IPC:120B"], []),
    ("sections 302 and 34 IPC", ["IPC:302", "IPC:34"], []),
    ("302 r/w 34 IPC", ["IPC:302", "IPC:34"], []),
    ("BNS section 103(1)", ["BNS:103(1)"], []),
    ("section 103 of the Bharatiya Nyaya Sanhita", ["BNS:103"], []),
    ("What is BNS 999?", [], ["BNS:999"]),
    ("What replaced 498A?", ["IPC:498A"], []),
    ("punishment for 7 years under BNS", [], []),
    ("five or more persons", [], []),
    ("Compare IPC 304A with BNS 106", ["IPC:304A", "BNS:106"], []),
    ("IPC 302 A pushes B", ["IPC:302"], []),
    ("Punishment for adultery under BNS 497?", [], ["BNS:497"]),
    ("u/s 376(2)(a) IPC", ["IPC:376(2)(a)"], []),
]


@pytest.fixture(scope="module")
def qa(cfg):
    return QueryAnalyzer(cfg, SECTIONS, PROVS)


@pytest.mark.parametrize("q,ids,missing", ID_TABLE)
def test_id_table(qa, q, ids, missing):
    a = qa.analyze(q)
    assert a.section_ids == ids
    assert a.nonexistent_ids == missing


def test_ambiguous_bare_number(qa):
    a = qa.analyze("What does section 304 say?")  # 304 exists in BNS only in this fixture
    assert a.section_ids == ["BNS:304"]
    a = qa.analyze("What does section 3 say?")
    assert a.section_ids == ["BNS:3"]


@pytest.mark.parametrize("q,intent,direction", [
    ("What replaced IPC 498A?", "mapping", "ipc_to_bns"),
    ("Which IPC section did BNS 318 come from?", "mapping", "bns_to_ipc"),
    ("How does BNS 106 differ from IPC 304A?", "comparison", None),
    ("What is the punishment for murder under the BNS?", "lookup", None),
    ("A pushes B, B dies. Which sections may be relevant?", "scenario", None),
])
def test_intents(qa, q, intent, direction):
    a = qa.analyze(q)
    assert a.intent == intent
    if direction:
        assert a.direction == direction


def test_out_of_scope_and_dates(qa):
    assert qa.analyze("Can I get anticipatory bail?").out_of_scope
    a = qa.analyze("A cheated B in March 2023, which IPC section applies?")
    assert a.offence_date.isoformat() == "2023-03-01"
    assert qa.analyze("offence committed before July 2024").offence_date.isoformat() == "2024-06-30"
