"""Mapping reconciliation (SDD 3.2.5): group concordance rows from all sources into connected
components, propose a type deterministically, check source agreement, write the review queue.

The LLM never proposes pairs, types or statuses.
"""
from __future__ import annotations

import csv
import difflib
import re
from collections import defaultdict
from pathlib import Path

from lawrag.legal import uid as U
from lawrag.models.legal import ActTree
from lawrag.models.mapping import Evidence, MappingCandidate, RawMappingRow

IDENTICAL_THRESHOLD = 0.97  # PROPOSED


def _norm(t: str) -> str:
    t = re.sub(r"\b(sanhita|code)\b", "ACT", t.lower())
    t = re.sub(r"\bsection \d+[a-z]*\b", "section N", t)
    return re.sub(r"\W+", " ", t).strip()


def similarity(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(a=_norm(a), b=_norm(b), autojunk=False).ratio(), 4)


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _section_text(uid: str, texts: dict[str, str]) -> str:
    return texts.get(uid) or texts.get(U.section_of(uid), "")


def reconcile(rows: list[RawMappingRow], trees: list[ActTree]) -> list[MappingCandidate]:
    texts = {s.uid: s.text for t in trees for s in t.sections}
    texts.update({p.uid: p.text for t in trees for p in t.provisions})
    uf = _UF()
    singles: list[RawMappingRow] = []
    for r in rows:
        if r.parse_error:
            continue
        nodes = [f"I|{u}" for u in r.ipc_uids] + [f"B|{u}" for u in r.bns_uids]
        if not nodes:
            continue
        for n in nodes[1:]:
            uf.union(nodes[0], n)
        uf.find(nodes[0])
        if not (r.ipc_uids and r.bns_uids):
            singles.append(r)
    comps: dict[str, dict] = defaultdict(lambda: {"ipc": set(), "bns": set(), "rows": []})
    for r in rows:
        if r.parse_error:
            continue
        nodes = [f"I|{u}" for u in r.ipc_uids] + [f"B|{u}" for u in r.bns_uids]
        if not nodes:
            continue
        c = comps[uf.find(nodes[0])]
        c["ipc"].update(r.ipc_uids)
        c["bns"].update(r.bns_uids)
        c["rows"].append(r)
    out = []
    for i, (_, c) in enumerate(sorted(comps.items(), key=lambda kv: (sorted(kv[1]["ipc"]), sorted(kv[1]["bns"])))):
        ipc, bns = sorted(c["ipc"]), sorted(c["bns"])
        typ, basis, sim = propose_type(ipc, bns, c["rows"], texts)
        pairs_by_source: dict[str, set] = defaultdict(set)
        for r in c["rows"]:
            pairs_by_source[r.source_id].add((tuple(sorted(r.ipc_uids)), tuple(sorted(r.bns_uids))))
        agree = len({frozenset(v) for v in pairs_by_source.values()}) <= 1
        ev = [Evidence(source_id=r.source_id, page=r.page, row=r.row_idx, raw_row=" | ".join(r.raw_cells),
                       extractor="pdfplumber") for r in c["rows"]]
        out.append(MappingCandidate(candidate_id=f"CAND-{i + 1:04d}", ipc_uids=ipc, bns_uids=bns, proposed_type=typ,
                                    type_basis=basis, sources_agree=agree, text_similarity=sim, evidence=ev))
    return out


def propose_type(ipc: list[str], bns: list[str], rows: list[RawMappingRow], texts: dict[str, str]):
    remarks = " ".join(r.remark for r in rows).lower()
    markers = {r.marker for r in rows}
    if ipc and not bns:
        return "OMITTED", "cardinality: IPC side only" + (" + marker deleted" if "deleted" in markers else ""), None
    if bns and not ipc:
        return "NEW", "cardinality: BNS side only" + (" + marker new" if "new" in markers else ""), None
    if len(ipc) == 1 and len(bns) >= 2:
        return "SPLIT", f"cardinality 1:{len(bns)}", None
    if len(ipc) >= 2 and len(bns) == 1:
        return "MERGED", f"cardinality {len(ipc)}:1", None
    if len(ipc) >= 2 and len(bns) >= 2:
        return "PARTIAL", f"cardinality {len(ipc)}:{len(bns)} (many-to-many)", None
    if re.search(r"\bpart(ly|ially)?\b", remarks):
        return "PARTIAL", "remark keyword 'partly'", None
    if re.search(r"\bmodif|amend|change", remarks):
        return "MODIFIED", "remark keyword 'modified'", None
    a, b = _section_text(ipc[0], texts), _section_text(bns[0], texts)
    if a and b:
        sim = similarity(a, b)
        return ("IDENTICAL" if sim >= IDENTICAL_THRESHOLD else "MODIFIED"), f"text similarity {sim:.3f}", sim
    return "MODIFIED", "1:1, no text available for similarity", None


QUEUE_FIELDS = ["mapping_id", "candidate_id", "ipc_uids", "bns_uids", "proposed_type", "type", "type_basis",
                "sources_agree", "text_similarity", "evidence", "ipc_text", "bns_text", "diff",
                "status", "reviewed_by", "reviewed_on", "review_note", "change_summary", "dispute_note", "granularity"]


def write_queue(cands: list[MappingCandidate], trees: list[ActTree], path: Path, start_id: int = 1) -> None:
    texts = {s.uid: s.text for t in trees for s in t.sections}
    texts.update({p.uid: p.text for t in trees for p in t.provisions})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=QUEUE_FIELDS)
        w.writeheader()
        for i, c in enumerate(cands):
            a = " / ".join(_section_text(u, texts) for u in c.ipc_uids)
            b = " / ".join(_section_text(u, texts) for u in c.bns_uids)
            diff = " ".join(f"[{op}:{' '.join(a.split()[i1:i2])}→{' '.join(b.split()[j1:j2])}]"
                            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a.split(), b=b.split()).get_opcodes()
                            if op != "equal")[:2000]
            gran = "provision" if any("(" in u for u in c.ipc_uids + c.bns_uids) else "section"
            w.writerow({"mapping_id": f"MAP-{start_id + i:04d}", "candidate_id": c.candidate_id,
                        "ipc_uids": ";".join(c.ipc_uids), "bns_uids": ";".join(c.bns_uids),
                        "proposed_type": c.proposed_type, "type": c.proposed_type, "type_basis": c.type_basis,
                        "sources_agree": c.sources_agree, "text_similarity": c.text_similarity,
                        "evidence": " || ".join(f"{e.source_id}|p{e.page}|r{e.row}|{e.raw_row}" for e in c.evidence),
                        "ipc_text": a[:3000], "bns_text": b[:3000], "diff": diff,
                        "status": "proposed" if c.sources_agree else "proposed (sources disagree: consider disputed)",
                        "reviewed_by": "", "reviewed_on": "", "review_note": "", "change_summary": "",
                        "dispute_note": "", "granularity": gran})
