"""Section splitter, concordance normalisation, reconciler, review gate, cross-references (SDD 5.1 #2-#5)."""
import csv

import pytest

from lawrag.errors import ValidationError
from lawrag.ingest.concordance_extractor import build_row, normalise_refs
from lawrag.ingest.mapping_reconciler import propose_type, reconcile, similarity, write_queue
from lawrag.ingest.review_io import read_approved
from lawrag.ingest.section_splitter import parse_act, split_provisions
from lawrag.ingest.xref_extractor import extract_all
from lawrag.models.legal import Block
from lawrag.models.mapping import RawMappingRow


def blocks(lines):
    return [Block(source_id="SRC-T", page=i // 5 + 1, bbox=(0, i * 10, 100, i * 10 + 9), text=t)
            for i, t in enumerate(lines)]


SAMPLE = [
    "CHAPTER VI", "OF OFFENCES AFFECTING THE HUMAN BODY",
    "100. Culpable homicide.—Whoever causes death by doing an act with the intention of causing death commits "
    "the offence of culpable homicide.",
    "Illustrations (a) A lays sticks over a pit. A has committed the offence. (b) A shoots at B.",
    "Explanation 1.—A person who causes bodily injury shall be deemed to have caused death.",
    "Explanation 2.—Where death is caused by bodily injury the person is deemed to have caused it.",
    "103. Punishment for murder.—(1) Whoever commits murder shall be punished with death. (2) When a group of "
    "five or more persons commits murder as referred to in sub-section (1), each shall be punished with death.",
    "105. Punishment for culpable homicide.—Whoever commits culpable homicide not amounting to murder shall be "
    "punished as provided in section 100 and section 103.",
    "106. Causing death by negligence.—(1) Whoever causes death by negligence shall be punished— (a) with "
    "imprisonment; or (b) with fine: Provided that a registered medical practitioner shall be punished less.",
]


def test_parse_act_structure():
    tree, _ = parse_act(blocks(SAMPLE), {"code": "BNS", "name": "BNS", "year": 2023, "text_as_of": "2024-07-01"},
                        "SRC-T")
    assert [s.number for s in tree.sections] == ["100", "103", "105", "106"]
    assert tree.chapters[0].uid == "BNS:CH-VI" and "HUMAN BODY" in tree.chapters[0].title
    uids = {p.uid for p in tree.provisions}
    assert {"BNS:100:body", "BNS:100:Ill(a)", "BNS:100:Ill(b)", "BNS:100:Expl1", "BNS:100:Expl2",
            "BNS:103(1)", "BNS:103(2)", "BNS:106(1)", "BNS:106(1)(a)", "BNS:106(1)(b)", "BNS:106:Prov1"} <= uids
    prov = next(p for p in tree.provisions if p.uid == "BNS:106:Prov1")
    assert prov.qualifies == ["BNS:106(1)"]
    refs, _ = extract_all(tree)
    pairs = {(r.from_uid, r.to_uid) for r in refs}
    assert ("BNS:105:body", "BNS:100") in pairs and ("BNS:105:body", "BNS:103") in pairs
    assert ("BNS:103(2)", "BNS:103(1)") in pairs  # sub-section reference resolved in the same section


def test_inline_reference_not_split():
    out = split_provisions("Whoever does the act referred to in sub-section (2) shall be punished.")
    assert len(out) == 1 and out[0]["kind"] == "body"


def test_missing_section_detected():
    from lawrag.ingest.tree_builder import validate_tree
    tree, _ = parse_act(blocks(SAMPLE), {"code": "BNS", "name": "BNS", "year": 2023, "text_as_of": "2024"}, "SRC-T")
    with pytest.raises(ValidationError):
        validate_tree(tree, ["100", "101", "103", "105", "106"])


@pytest.mark.parametrize("cell,act,uids,marker,rng,err", [
    ("Sec. 376(2)(a) & (b)", "IPC", ["IPC:376(2)(a)", "IPC:376(2)(b)"], "none", False, False),
    ("299 to 304", "IPC", ["IPC:299", "IPC:300", "IPC:301", "IPC:302", "IPC:303", "IPC:304"], "none", True, False),
    ("120-B", "IPC", ["IPC:120B"], "none", False, False),
    ("85, 86", "BNS", ["BNS:85", "BNS:86"], "none", False, False),
    ("New", "BNS", [], "new", False, False),
    ("Deleted", "IPC", [], "deleted", False, False),
    ("—", "BNS", [], "none", False, False),
    ("see note 4", "IPC", [], "none", False, True),
])
def test_normalise_refs(cell, act, uids, marker, rng, err):
    r = normalise_refs(cell, act)
    assert r["uids"] == uids and r["marker"] == marker and r["range_expanded"] == rng
    assert bool(r["parse_error"]) == err


def test_column_validity():
    r = normalise_refs("999", "BNS", valid={"1", "2"})
    assert r["parse_error"] and "not valid" in r["parse_error"]


def _row(ipc, bns, src="SRC-A", remark="", row=1):
    return RawMappingRow(source_id=src, page=1, row_idx=row, raw_cells=[";".join(ipc), ";".join(bns)],
                         ipc_uids=ipc, bns_uids=bns, remark=remark)


def test_reconciler_types_and_agreement():
    rows = [_row(["IPC:498A"], ["BNS:85", "BNS:86"]), _row(["IPC:417"], ["BNS:318"]), _row(["IPC:420"], ["BNS:318"]),
            _row(["IPC:497"], [], remark="Deleted"), _row([], ["BNS:304"]),
            _row(["IPC:124A"], ["BNS:152"], src="SRC-A"), _row(["IPC:124A"], ["BNS:150"], src="SRC-B")]
    cands = {tuple(c.ipc_uids): c for c in reconcile(rows, [])}
    assert cands[("IPC:498A",)].proposed_type == "SPLIT"
    assert cands[("IPC:417", "IPC:420")].proposed_type == "MERGED"
    assert cands[("IPC:497",)].proposed_type == "OMITTED"
    assert cands[()].proposed_type == "NEW"
    assert cands[("IPC:124A",)].sources_agree is False


def test_similarity_rule():
    t = "Whoever commits murder shall be punished with death."
    assert propose_type(["IPC:302"], ["BNS:103"], [], {"IPC:302": t, "BNS:103": t})[0] == "IDENTICAL"
    assert propose_type(["IPC:302"], ["BNS:103"], [], {"IPC:302": t, "BNS:103": "Something else entirely."})[0] == "MODIFIED"
    assert similarity("section 302 of this Code", "section 103 of this Sanhita") == 1.0


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


BASE = {"mapping_id": "MAP-0001", "ipc_uids": "IPC:302", "bns_uids": "BNS:103(1)", "type": "IDENTICAL",
        "status": "verified", "reviewed_by": "R1", "reviewed_on": "2026-11-01",
        "evidence": "SRC-A|p12|r7|302 | 103(1)", "granularity": "provision"}


def test_review_gate_accepts_valid(tmp_path):
    p = tmp_path / "a.csv"
    _write(p, [BASE])
    recs = read_approved(p, {"IPC:302", "BNS:103(1)"}, {"SRC-A"})
    assert recs[0].evidence[0].page == 12 and recs[0].evidence[0].row == 7


@pytest.mark.parametrize("change", [{"reviewed_by": ""}, {"reviewed_on": ""}, {"status": "maybe"},
                                    {"bns_uids": "BNS:999"}, {"evidence": ""}, {"evidence": "SRC-Z|p1|r1|x"},
                                    {"reviewed_on": "1 Nov 2026"}])
def test_review_gate_rejects(tmp_path, change):
    p = tmp_path / "a.csv"
    _write(p, [{**BASE, **change}])
    with pytest.raises(ValidationError):
        read_approved(p, {"IPC:302", "BNS:103(1)"}, {"SRC-A"})


def test_queue_written(tmp_path):
    cands = reconcile([_row(["IPC:302"], ["BNS:103"])], [])
    write_queue(cands, [], tmp_path / "q.csv")
    rows = list(csv.DictReader((tmp_path / "q.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "proposed" and rows[0]["reviewed_by"] == ""


def test_build_row_markers():
    r = build_row("S", 1, 0, ["497", "—", ""], 0, 1, 2)
    assert r.marker == "deleted" and r.ipc_uids == ["IPC:497"]
    r = build_row("S", 1, 0, ["—", "304", "New"], 0, 1, 2)
    assert r.marker == "new"
