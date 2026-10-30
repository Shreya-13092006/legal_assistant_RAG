"""Cross-reference extraction -> REFERS_TO edges (SDD 3.2.7).

Handles "section 34", "sections 299 to 304", "sub-section (2)" (same section), "the preceding
section"; act-level phrases ("this Sanhita") are not stored; references to other Acts are logged.
Unresolvable references are returned separately and never guessed.
"""
from __future__ import annotations

import re

from lawrag.legal import uid as U
from lawrag.models.legal import ActTree, CrossRef, Provision

_SECTION = re.compile(r"\bsections?\s+(\d{1,3}[A-Z]{0,2})(?:\s*(?:to|-)\s*(\d{1,3}[A-Z]{0,2}))?"
                      r"((?:\s*(?:,|and|or)\s*\d{1,3}[A-Z]{0,2})*)"
                      r"(?P<ext>\s+of\s+the\s+[A-Z][\w ,()]*?Act(?:,\s*\d{4})?)?", re.I)
_SUBSECTION = re.compile(r"\bsub-section\s*\((\d+[A-Z]?)\)(?!\s*of\s+section)", re.I)
_PRECEDING = re.compile(r"\bthe (?:last )?preceding section\b", re.I)


def extract_refs(prov: Provision, tree: ActTree, known: set[str]) -> tuple[list[CrossRef], list[dict]]:
    act = tree.act.code
    refs: list[CrossRef] = []
    unresolved: list[dict] = []
    text = prov.text
    numbers_sorted = sorted({s.number for s in tree.sections}, key=U.sort_key)
    for m in _SECTION.finditer(text):
        if m.group("ext"):
            unresolved.append({"from": prov.uid, "raw": m.group(0), "reason": "external act"})
            continue
        nums = [m.group(1).upper()]
        if m.group(2):
            lo, hi = U.sort_key(m.group(1).upper()), U.sort_key(m.group(2).upper())
            nums = [n for n in numbers_sorted if lo <= U.sort_key(n) <= hi]
        nums += [n.upper() for n in re.findall(r"\d{1,3}[A-Z]{0,2}", m.group(3) or "")]
        for n in nums:
            target = f"{act}:{n}"
            if target == prov.section_uid:
                continue
            if target in known:
                refs.append(CrossRef(from_uid=prov.uid, to_uid=target, raw_text=m.group(0).strip()))
            else:
                unresolved.append({"from": prov.uid, "raw": m.group(0), "target": target, "reason": "not in tree"})
    for m in _SUBSECTION.finditer(text):
        target = f"{prov.section_uid}({m.group(1)})"
        if target in known and target != prov.uid:
            refs.append(CrossRef(from_uid=prov.uid, to_uid=target, raw_text=m.group(0)))
        elif target not in known:
            unresolved.append({"from": prov.uid, "raw": m.group(0), "target": target, "reason": "not in tree"})
    for m in _PRECEDING.finditer(text):
        idx = numbers_sorted.index(U.parse_uid(prov.uid).number) if U.parse_uid(prov.uid).number in numbers_sorted else -1
        if idx > 0:
            refs.append(CrossRef(from_uid=prov.uid, to_uid=f"{act}:{numbers_sorted[idx - 1]}", raw_text=m.group(0)))
        else:
            unresolved.append({"from": prov.uid, "raw": m.group(0), "reason": "no preceding section"})
    dedup = {(r.from_uid, r.to_uid): r for r in refs}
    return list(dedup.values()), unresolved


def extract_all(tree: ActTree) -> tuple[list[CrossRef], list[dict]]:
    known = {s.uid for s in tree.sections} | {p.uid for p in tree.provisions}
    refs, unresolved = [], []
    for p in tree.provisions:
        r, u = extract_refs(p, tree, known)
        refs += r
        unresolved += u
    return refs, unresolved
