"""Build an ActTree from a nested section/provision description.

The seed YAML files and the PDF section splitter both produce this nested form, so uid assignment,
qualifier targeting and validation live in one place.

Section dict:   {number, title, page, status, status_note, status_source_id, amendment_notes,
                 provisions: [provision dict]}
Provision dict: {kind, label?, n?, text, target?, in_force?, status_note?, children?: [...]}
"""
from __future__ import annotations

from collections import Counter

from lawrag.errors import ValidationError
from lawrag.legal import uid as U
from lawrag.models.legal import QUALIFIER_KINDS, Act, ActTree, Chapter, Provision, Section

_QUAL_CODE = {"explanation": "Expl", "exception": "Exc", "proviso": "Prov"}
_KIND_LABEL = {"explanation": "Explanation", "exception": "Exception", "proviso": "Proviso",
               "illustration": "Illustration"}


def _provision_uid(sec_uid: str, parent_uid: str, item: dict, counters: Counter) -> tuple[str, str]:
    kind = item["kind"]
    if kind == "body":
        return f"{sec_uid}:body", "body"
    if kind in ("subsection", "clause", "subclause"):
        label = str(item["label"]).strip("() ")
        return f"{parent_uid}({label})", f"({label})"
    if kind == "illustration":
        label = str(item.get("label") or chr(ord("a") + counters["illustration"])).strip("() ").lower()
        counters["illustration"] += 1
        return f"{sec_uid}:Ill({label})", f"Illustration ({label})"
    code = _QUAL_CODE[kind]
    counters[kind] += 1
    n = int(item.get("n") or counters[kind])
    return f"{sec_uid}:{code}{n}", f"{_KIND_LABEL[kind]} {n}"


def _walk(sec_uid: str, parent_uid: str, items: list[dict], out: list[Provision], counters: Counter,
          default_in_force: bool, page: int | None) -> None:
    for item in items:
        puid, label = _provision_uid(sec_uid, parent_uid, item, counters)
        kind = item["kind"]
        qualifies: list[str] = []
        if kind in QUALIFIER_KINDS:
            target = item.get("target")
            if target:
                qualifies = [target if ":" in target else f"{sec_uid}{target}"]
            else:
                qualifies = [sec_uid]
        prov = Provision(
            uid=puid, section_uid=sec_uid,
            parent_uid=sec_uid if kind in QUALIFIER_KINDS or kind == "body" else parent_uid,
            kind=kind, label=label, text=str(item.get("text", "")).strip(), order=len(out),
            in_force=bool(item.get("in_force", default_in_force)), status_note=item.get("status_note"),
            qualifies=qualifies, page=item.get("page", page),
        )
        out.append(prov)
        if item.get("children"):
            _walk(sec_uid, puid, item["children"], out, counters, prov.in_force, page)


def build_tree(act: dict, chapters: list[dict], source_id: str) -> ActTree:
    act_m = Act(**act)
    tree = ActTree(act=act_m)
    for ci, ch in enumerate(chapters):
        ch_uid = f"{act_m.code}:CH-{ch['number']}"
        tree.chapters.append(Chapter(uid=ch_uid, act=act_m.code, number=str(ch["number"]),
                                     title=ch.get("title", ""), order=ci))
        for sec in ch.get("sections", []):
            number = U.normalize_number(str(sec["number"]))
            sec_uid = U.section_uid(act_m.code, number)
            status = sec.get("status", "in_force")
            provs: list[Provision] = []
            _walk(sec_uid, sec_uid, sec.get("provisions", []), provs, Counter(),
                  default_in_force=status in ("in_force",), page=sec.get("page"))
            text = "\n".join(f"{p.label} {p.text}" if p.kind != "body" else p.text for p in provs)
            tree.sections.append(Section(
                uid=sec_uid, act=act_m.code, number=number, sort_key=list(U.sort_key(number)),
                title=sec.get("title", ""), chapter_uid=ch_uid, status=status,
                source_id=sec.get("source_id", source_id), page_start=sec.get("page"),
                page_end=sec.get("page_end", sec.get("page")), text=text,
                amendment_notes=sec.get("amendment_notes", []), status_note=sec.get("status_note"),
                status_source_id=sec.get("status_source_id"),
            ))
            tree.provisions.extend(provs)
    validate_tree(tree)
    return tree


def validate_tree(tree: ActTree, expected_numbers: list[str] | None = None) -> list[str]:
    """Tree validation (SDD 3.2.9). Raises on hard errors, returns warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    acts = [tree.act.code]
    seen = Counter(p.uid for p in tree.provisions) + Counter(s.uid for s in tree.sections)
    dups = [u for u, n in seen.items() if n > 1]
    if dups:
        errors.append(f"duplicate uids: {dups[:10]}")
    for p in tree.provisions:
        if not U.is_valid(p.uid, acts):
            errors.append(f"malformed provision uid {p.uid}")
            continue
        if U.format_uid(U.parse_uid(p.uid)) != p.uid:
            errors.append(f"uid round-trip failed for {p.uid}")
        if p.is_qualifier and not p.qualifies:
            errors.append(f"orphan qualifier {p.uid}")
    sec_uids = {s.uid for s in tree.sections}
    prov_uids = {p.uid for p in tree.provisions}
    for p in tree.provisions:
        for t in p.qualifies:
            if t not in sec_uids and t not in prov_uids:
                errors.append(f"qualifier {p.uid} targets unknown node {t}")
    for s in tree.sections:
        if s.status == "in_force" and not any(p.section_uid == s.uid for p in tree.provisions):
            errors.append(f"in-force section {s.uid} has no provisions")
    if expected_numbers is not None:
        parsed = {s.number for s in tree.sections}
        missing = [n for n in expected_numbers if n not in parsed]
        extra = sorted(parsed - set(expected_numbers), key=U.sort_key)
        if missing:
            errors.append(f"missing sections: {missing[:30]}{'...' if len(missing) > 30 else ''}")
        if extra:
            warnings.append(f"unexpected sections: {extra[:30]}")
    lengths = sorted(len(s.text) for s in tree.sections)
    if len(lengths) >= 20:
        p99 = lengths[int(0.99 * (len(lengths) - 1))]
        for s in tree.sections:
            if len(s.text) > p99 and len(s.text) > 4000:
                warnings.append(f"abnormally long section {s.uid} (possible merged header)")
    if errors:
        raise ValidationError("; ".join(errors), errors=errors)
    return warnings
