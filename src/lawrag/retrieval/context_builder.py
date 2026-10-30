"""Context construction with citation-preserving truncation (SDD 3.4.7).

The ContextPack is the single evidence object shared by the generator and the verifier. Identical
input gives byte-identical output.
"""
from __future__ import annotations

import re

from lawrag.graph.store import GraphStore
from lawrag.index.chunker import count_tokens
from lawrag.legal import uid as U
from lawrag.models.legal import Provision, Section
from lawrag.models.mapping import MappingFact
from lawrag.models.qa import Candidate, ContextBlock, ContextPack, QueryAnalysis
from lawrag.retrieval.mapping_facts import statement


def _label(p: Provision) -> str:
    return "Text" if p.kind == "body" else p.label


def provision_block(p: Provision, sec: Section, qualifier: bool = False, excerpt: str | None = None) -> ContextBlock:
    head = f"{sec.act} s.{sec.number} — {sec.title}"
    lab = _label(p)
    if qualifier and p.qualifies:
        lab += f" (qualifies {', '.join(p.qualifies)})"
    text = excerpt if excerpt is not None else p.text
    tag = f"{p.uid} | EXCERPT" if excerpt is not None else p.uid
    body = f"[{tag}] {head} | {lab}: {text}"
    if not p.in_force:
        body += f" [NOT IN FORCE: {p.status_note or sec.status_note or sec.status}]"
    return ContextBlock(label=p.uid, kind="qualifier" if qualifier else "provision", text=body, uids=[p.uid],
                        source_id=sec.source_id, page=p.page or sec.page_start, excerpt=excerpt is not None,
                        section_uid=sec.uid)


def status_block(sec: Section, store: GraphStore) -> ContextBlock:
    src = store.get_source(sec.status_source_id) if sec.status_source_id else None
    note = sec.status_note or ""
    srcs = f" (source: {src.title}, secondary)" if src else ""
    return ContextBlock(label=f"STATUS {sec.uid}", kind="status_note",
                        text=f"[STATUS {sec.uid}] {sec.act} s.{sec.number} — {sec.title}: status {sec.status}. {note}{srcs}",
                        uids=[sec.uid], source_id=sec.status_source_id or sec.source_id, page=sec.page_start,
                        section_uid=sec.uid)


def mapping_block(f: MappingFact) -> ContextBlock:
    e = f.evidence[0] if f.evidence else {}
    src = e.get("source_id", "?")
    if e.get("page") is not None:
        src += f", p.{e['page']}"
    if e.get("row") is not None:
        src += f" row {e['row']}"
    arrow = f"{', '.join(f.from_uids) or '—'} -> {', '.join(f.to_uids) or '—'}"
    text = f"[MAPPING {f.mapping_id} | {f.type} | {f.status} | Source: {src}] {arrow}. Statement: {statement(f)}"
    if f.change_summary:
        text += f" Change summary: {f.change_summary}"
    note = f.dispute_note or f.review_note
    if note:
        text += f" Note: {note}"
    return ContextBlock(label=f"MAPPING {f.mapping_id}", kind="mapping_fact", text=text, uids=[f.mapping_id],
                        source_id=e.get("source_id", ""), page=e.get("page"))


def _excerpt(text: str, budget: int) -> str:
    sents = re.split(r"(?<=[.;:])\s+", text)
    out = ""
    for s in sents:
        if count_tokens(out + " " + s) > budget and out:
            break
        out = f"{out} {s}".strip()
    return out


def build_context(fused: list[Candidate], analysis: QueryAnalysis, store: GraphStore, budget_tokens: int,
                  mapping_facts: list[MappingFact] | None = None, include_secondary: bool = False) -> ContextPack:
    facts = sorted({f.mapping_id: f for f in (mapping_facts or [])}.values(), key=lambda f: f.mapping_id)
    # 1. resolve candidates to rule provisions (section summaries / status hits expand to the section)
    units: list[tuple[str, bool, list[Provision]]] = []   # (section_uid, pinned, rule provisions)
    seen: set[str] = set()
    secondary: list[Candidate] = []
    for c in fused:
        if c.secondary:
            secondary.append(c)
            continue
        if c.uid == c.section_uid:  # summary / status candidates
            provs = [p for p in store.provisions_of(c.section_uid) if not p.is_qualifier]
        else:
            p = store.get_provision(c.uid)
            provs = [p] if p else []
            if p and p.is_qualifier:  # a qualifier hit brings in the rule it qualifies
                tgt = [store.get_provision(t) for t in p.qualifies if store.get_provision(t)]
                provs = [t for t in tgt if not t.is_qualifier] or [
                    q for q in store.provisions_of(p.section_uid) if not q.is_qualifier][:1]
                provs = provs + [p]
        provs = [p for p in provs if p.uid not in seen]
        seen.update(p.uid for p in provs)
        if provs or c.uid == c.section_uid:
            units.append((c.section_uid, c.pinned, provs))

    # 2. group into sections, keeping first-appearance (fused) order; pair IPC/BNS for comparisons
    order: list[str] = []
    pinned_secs: set[str] = set()
    by_sec: dict[str, list[tuple[bool, Provision]]] = {}
    for sec_uid, pinned, provs in units:
        if sec_uid not in by_sec:
            order.append(sec_uid)
            by_sec[sec_uid] = []
        if pinned:
            pinned_secs.add(sec_uid)
        by_sec[sec_uid] += [(pinned, p) for p in provs]
    if analysis.intent == "comparison" and facts:
        paired: list[str] = []
        for s in order:
            if s in paired:
                continue
            paired.append(s)
            for f in facts:
                if any(U.section_of(u) == s for u in f.from_uids):
                    paired += [U.section_of(t) for t in f.to_uids if U.section_of(t) in by_sec and U.section_of(t) not in paired]
        order = paired + [s for s in order if s not in paired]

    # 3. assemble droppable units: one unit = section header/status + rule provision + its qualifiers
    blocks_units: list[tuple[bool, str, list[ContextBlock]]] = []
    for sec_uid in order:
        sec = store.get_section(sec_uid)
        if sec is None:
            continue
        first = True
        provs = sorted(by_sec[sec_uid], key=lambda x: x[1].order)
        if not provs:
            blocks_units.append((sec_uid in pinned_secs, sec_uid, [status_block(sec, store)]))
            continue
        attached: set[str] = set()
        for pinned, p in provs:
            unit: list[ContextBlock] = []
            if first and (sec.status != "in_force" or sec.status_note):
                unit.append(status_block(sec, store))
            first = False
            if p.is_qualifier:
                if p.uid not in attached:
                    unit.append(provision_block(p, sec, qualifier=True))
                    attached.add(p.uid)
            else:
                unit.append(provision_block(p, sec))
                for q in store.qualifiers_of(p.uid):
                    if q.uid not in attached:
                        unit.append(provision_block(q, sec, qualifier=True))
                        attached.add(q.uid)
            if unit:
                blocks_units.append((pinned or sec_uid in pinned_secs, p.uid, unit))

    mapping_blocks = [mapping_block(f) for f in facts]
    sec_blocks: list[ContextBlock] = []
    if include_secondary:
        for c in secondary:
            ch = store.get_chunk(c.chunk_ids[0]) if c.chunk_ids else None
            if ch:
                src = store.get_source(ch.source_id)
                sec_blocks.append(ContextBlock(
                    label=ch.chunk_id, kind="secondary", uids=[ch.chunk_id], source_id=ch.source_id,
                    text=f"[{ch.chunk_id}] (SECONDARY SOURCE: {src.title if src else ch.source_id}, "
                         f"¶{ch.para_range}) {ch.display_text}"))

    # 4. budget: drop whole lowest-ranked non-pinned units first; never qualifiers of kept rules,
    #    pinned provisions or verified mapping facts
    def total(units_, extra):
        return sum(count_tokens(b.text) for _, _, u in units_ for b in u) + sum(count_tokens(b.text) for b in extra)

    dropped: list[str] = []
    kept = list(blocks_units)
    extra = mapping_blocks + sec_blocks
    while total(kept, extra) > budget_tokens and sec_blocks:
        b = sec_blocks.pop()
        extra.remove(b)
        dropped.append(b.label)
    while total(kept, extra) > budget_tokens:
        idx = next((i for i in range(len(kept) - 1, -1, -1) if not kept[i][0]), None)
        if idx is None:
            break
        dropped += [b.label for b in kept[idx][2]]
        kept.pop(idx)
    if total(kept, extra) > budget_tokens:  # only pinned material left: excerpt the longest provisions
        over = total(kept, extra) - budget_tokens
        for i, (pin, key, unit) in sorted(enumerate(kept), key=lambda x: -count_tokens(x[1][2][0].text)):
            head = unit[0]
            if head.kind != "provision" or over <= 0:
                continue
            p = store.get_provision(head.label)
            sec = store.get_section(p.section_uid)
            keep_tokens = max(40, count_tokens(p.text) - over)
            ex = _excerpt(p.text, keep_tokens)
            new = provision_block(p, sec, excerpt=ex)
            over -= count_tokens(head.text) - count_tokens(new.text)
            kept[i] = (pin, key, [new] + unit[1:])

    blocks = mapping_blocks + [b for _, _, u in kept for b in u] + sec_blocks
    return ContextPack(blocks=blocks, mapping_facts=facts, token_count=sum(count_tokens(b.text) for b in blocks),
                       dropped=dropped)
