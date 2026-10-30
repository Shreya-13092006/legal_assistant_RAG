"""Bounded graph expansion and mapping traversal (SDD 3.4.5)."""
from __future__ import annotations

from lawrag.config import GraphCfg
from lawrag.graph.store import GraphStore
from lawrag.legal import uid as U
from lawrag.models.mapping import MappingFact
from lawrag.models.qa import Candidate, QueryAnalysis
from lawrag.retrieval.mapping_facts import facts_for

ALLOW = {
    "lookup": ({"QUALIFIES"}, 1),
    "mapping": ({"MAPPING", "QUALIFIES"}, 2),
    "comparison": ({"MAPPING", "QUALIFIES", "REFERS_TO"}, 2),
    "scenario": ({"REFERS_TO", "QUALIFIES", "SUCCEEDED_BY"}, 2),
    "unknown": ({"QUALIFIES"}, 1),
}


def is_definitional(sec_uid: str) -> bool:
    """Definition hubs are not expanded through REFERS_TO (VERIFY ranges: BNS s.2, IPC ss.6-52A)."""
    p = U.parse_uid(sec_uid)
    n = U.sort_key(p.number)[0]
    return (p.act == "BNS" and n == 2) or (p.act == "IPC" and 6 <= n <= 52)


def graph_expand(seeds: list[Candidate], analysis: QueryAnalysis, store: GraphStore, cfg: GraphCfg,
                 extra_rels: set[str] | None = None, extra_hops: int = 0) -> tuple[list[Candidate], list[MappingFact]]:
    rels, hops_allowed = ALLOW.get(analysis.intent, ALLOW["unknown"])
    rels = set(rels) | (extra_rels or set())
    max_hops = min(hops_allowed + extra_hops, cfg.max_hops + extra_hops)
    out: dict[str, Candidate] = {}
    facts: dict[str, MappingFact] = {}
    visited = {s.uid for s in seeds}
    frontier = [(s, s.ranks.get("seed", i + 1)) for i, s in enumerate(seeds)]

    def add(uid: str, kind: str, rel: str, seed_rank: int, hops: int) -> bool:
        if uid in visited or len(out) >= cfg.max_expanded or not store.exists(uid):
            return False
        visited.add(uid)
        sec = U.section_of(uid)
        out[uid] = Candidate(uid=uid, section_uid=sec, act=U.act_of(uid), kind=kind, via=[f"graph:{rel}"],
                             ranks={"graph": seed_rank + cfg.hop_penalty * hops}, hops=hops)
        return True

    def section_rules(sec_uid: str) -> list[tuple[str, str]]:
        provs = store.provisions_of(sec_uid)
        rules = [(p.uid, p.kind) for p in provs if not p.is_qualifier]
        return rules or [(sec_uid, "status")]

    # false premise: the same number in the other act may carry the relevant fact (e.g. an OMITTED hub)
    for bad in analysis.nonexistent_ids:
        num = bad.split(":", 1)[1]
        for other in ("IPC", "BNS"):
            alt = f"{other}:{num}"
            if alt != bad and store.exists(alt):
                for f in facts_for(alt, store):
                    facts[f.mapping_id] = f
                for uid, kind in section_rules(alt)[: cfg.max_fanout]:
                    add(uid, kind, "PREMISE", 1, 1)

    # when the question names sections, hubs are traversed from those only (other seeds add noise)
    mapping_seeds = {s.uid for s in seeds if s.pinned} or {s.uid for s in seeds}
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        nxt = []
        for seed, srank in frontier:
            fan = 0
            if "QUALIFIES" in rels and not seed.secondary:
                for q in store.qualifiers_of(seed.uid) if seed.uid != seed.section_uid else []:
                    if fan >= cfg.max_fanout:
                        break
                    if add(q.uid, q.kind, "QUALIFIES", srank, hop):
                        fan += 1
            if "MAPPING" in rels and hop == 1 and not seed.secondary and seed.uid in mapping_seeds:
                for f in facts_for(seed.uid, store, analysis.direction if seed.pinned else None):
                    facts[f.mapping_id] = f
                    if f.status != "verified":
                        continue  # disputed/unresolved hubs never yield authoritative target text
                    seed_sec = U.section_of(seed.uid)
                    on_from = any(U.section_of(u) == seed_sec for u in f.from_uids)
                    other_side = f.to_uids if on_from else f.from_uids
                    for t in sorted(other_side):
                        targets = section_rules(t) if U.is_section_uid(t) else [(t, "subsection")]
                        for uid, kind in targets:
                            if fan >= cfg.max_fanout:
                                break
                            if add(uid, kind, "MAPS_TO", srank, hop):
                                fan += 1
                                nxt.append((out[uid], srank))
            if "REFERS_TO" in rels and not seed.secondary and not (
                    is_definitional(seed.section_uid) and analysis.intent != "lookup"):
                for t in store.refers_to(seed.uid):
                    if fan >= cfg.max_fanout:
                        break
                    if is_definitional(U.section_of(t)):
                        continue
                    for uid, kind in (section_rules(t) if U.is_section_uid(t) else [(t, "subsection")]):
                        if add(uid, kind, "REFERS_TO", srank, hop):
                            fan += 1
                            nxt.append((out[uid], srank))
            if "SUCCEEDED_BY" in rels and not seed.secondary:
                for t, _mid, _typ in store.succeeded_by(seed.uid):
                    for uid, kind in (section_rules(t) if U.is_section_uid(t) else [(t, "subsection")]):
                        if fan >= cfg.max_fanout:
                            break
                        if add(uid, kind, "SUCCEEDED_BY", srank, hop):
                            fan += 1
                            nxt.append((out[uid], srank))
        frontier = nxt
    return list(out.values()), sorted(facts.values(), key=lambda f: f.mapping_id)
