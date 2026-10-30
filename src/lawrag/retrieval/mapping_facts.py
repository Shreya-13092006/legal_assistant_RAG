"""Turn :Mapping hubs into MappingFacts with the permitted phrasing (SDD 3.1.4 status table)."""
from __future__ import annotations

from lawrag.graph.store import GraphStore
from lawrag.models.mapping import MappingFact, MappingRecord


def _src_phrase(fact_evidence: list[dict]) -> str:
    if not fact_evidence:
        return "the reviewed mapping data"
    e = fact_evidence[0]
    parts = [e.get("title") or e["source_id"]]
    if e.get("page") is not None:
        parts.append(f"p.{e['page']}")
    if e.get("row") is not None:
        parts.append(f"row {e['row']}")
    return ", ".join(parts)


def statement(f: MappingFact) -> str:
    src = _src_phrase(f.evidence)
    frm, to = ", ".join(f.from_uids), ", ".join(f.to_uids)
    if f.status == "verified":
        if f.type == "OMITTED":
            return f"According to {src}, {frm} has no direct BNS equivalent (mapping type OMITTED, verified)."
        if f.type == "NEW":
            return f"According to {src}, {to} is new in the BNS and has no IPC predecessor (mapping type NEW, verified)."
        return f"According to {src}, {frm} corresponds to {to} (mapping type {f.type}, verified)."
    if f.status == "disputed":
        return (f"Sources differ on whether {frm} corresponds to {to or 'any BNS provision'}; this mapping is "
                f"disputed and this system does not resolve the conflict.")
    subject = frm or to
    return (f"No verified mapping is recorded for {subject}; the correspondence is unresolved in this system "
            f"(not established in this system's verified data).")


def to_fact(m: MappingRecord, store: GraphStore) -> MappingFact:
    ev = []
    for e in m.evidence:
        s = store.get_source(e.source_id)
        ev.append({"source_id": e.source_id, "title": s.title if s else e.source_id, "page": e.page, "row": e.row,
                   "raw_row": e.raw_row, "authority_rank": s.authority_rank if s else None,
                   "url": s.url if s else None})
    f = MappingFact(mapping_id=m.mapping_id, type=m.type, status=m.status, granularity=m.granularity,
                    from_uids=m.from_uids, to_uids=m.to_uids, evidence=ev, change_summary=m.change_summary,
                    review_note=m.review_note, dispute_note=m.dispute_note, reviewed_by=m.reviewed_by,
                    reviewed_on=m.reviewed_on, sources_agree=m.sources_agree, authoritative=m.status == "verified")
    f.caveat = statement(f) if m.status != "verified" else None
    return f


def facts_for(uid: str, store: GraphStore, direction: str | None = None,
              include_unverified: bool = True) -> list[MappingFact]:
    """Hubs of every final status are returned (for caveats); proposed/rejected never reach answers."""
    out = []
    for m in store.mappings_for(uid, direction):
        if m.status in ("rejected", "proposed"):
            continue
        if not include_unverified and m.status != "verified":
            continue
        out.append(to_fact(m, store))
    return out
